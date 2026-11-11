//! StemLoop -- an MPC Sample mod.
//!
//! Adds a "Stem Split" action: take the selected sample, split it into four
//! stems (drums / bass / other / vocals), and write each to its own (next free)
//! pad via the app's Extract command. The split runs the real engine
//! (`mpc_stemloop_engine`): host STFT + ONNX net + iSTFT. The net ONNX is loaded
//! from `STEMLOOP_MODEL_DIR` (set by the autorun), reading the model from the SD
//! card in place.
//!
//! Trigger: SHIFT + DATA on the Sample page (key 0x6f). Everything runs on an SDK
//! `Task` worker so the UI never blocks. On-screen notices (a banner + short
//! messages) are drawn in `on_paint` on the UI thread -- no background repaint
//! pump (an earlier pump raced the app's renderer at Extract and crashed it).
//!
//! Nothing here may panic -- a panic in a preloaded library aborts the sampler.

#![deny(clippy::panic, clippy::todo, clippy::unimplemented)]

use core::ffi::CStr;
use std::sync::Mutex;

use mpc_sample_sdk::platform::{clock, extract, page, pads, selection, status};
use mpc_sample_sdk::render::{RenderJob, RenderOutput};
use mpc_sample_sdk::task::{Task, TaskError};
use mpc_sample_sdk::{ButtonEvent, ButtonFlow, Mod, PaintEvent, draw, sites};
use mpc_stemloop_engine::{Engine, SR};

/// DATA / encoder-push key code (see SDK docs/02-buttons.md).
const KEY_DATA: u32 = 0x6f;
/// CHOP button key code; replaying it toggles the chop view back to Trim.
const KEY_CHOP: u8 = 0x4b;
/// Our render-job kind tag (single action for now).
const KIND_SPLIT: i32 = 0;
/// The four demucs stems, in order, as Extract pad names.
const STEMS: [&CStr; 4] = [c"Drums", c"Bass", c"Other", c"Vocals"];
/// Longest loop (seconds) we process; longer input is truncated to this.
const MAX_SECS: usize = 8;
/// Largest model bucket we ship = the chunk window for longer loops.
const WINDOW_SECS: usize = 4;
/// How long (ms) a one-shot notice (not the running banner) stays on screen.
const NOTICE_MS: i64 = 4000;
/// How long (ms) after a split we watch for the chop view, to leave it.
const CHOP_EXIT_MS: i64 = 4000;

/// A short on-screen notice (text, deadline-ms); drawn in `on_paint`.
static NOTICE: Mutex<(&'static str, i64)> = Mutex::new(("", 0));

/// The raw bytes of the trigger button event, captured for a later replay: the
/// app opens the chop view after an Extract, and replaying CHOP leaves it.
static TRIGGER: Mutex<[u8; 16]> = Mutex::new([0; 16]);

/// One ONNX engine kept alive across splits, keyed by its window size.
///
/// Loading the net costs ~30 s cold and ~335 MB resident. On the 2 GB device a
/// *second* cold load cannot allocate its weights alongside the first (the first
/// split's freed heap is retained by glibc), so a fresh engine per split made
/// the second split fail. Caching reuses the loaded net: the repeat split is
/// instant and allocates nothing new. A different loop length rebuilds, dropping
/// the old engine and trimming the heap first so the new net has room.
static ENGINE: Mutex<Option<(usize, Engine)>> = Mutex::new(None);

/// Return freed heap pages to the OS (glibc keeps them mapped otherwise).
fn trim_heap() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: `malloc_trim` only releases unused top-of-heap pages; it never
    // touches live allocations and is safe to call at any time on glibc.
    unsafe {
        malloc_trim(0);
    }
}

/// Show a one-shot notice for `NOTICE_MS` and ask for a repaint.
fn notice(text: &'static str) {
    if let Ok(mut n) = NOTICE.lock() {
        *n = (text, clock::now_ms() + NOTICE_MS);
    }
    draw::show_toast(text);
    draw::repaint_painted();
}

/// Draw the StemLoop banner with a line of text (320x240 screen).
fn draw_banner(g: draw::Graphics, line: &str) {
    g.colour(0xff10_1014);
    g.rect(28, 92, 264, 56);
    g.colour(0xffff_cc00);
    g.text("STEM SPLIT", 44, 100, 232, 18, 15.0);
    g.colour(0xffff_ffff);
    g.text(line, 44, 124, 232, 14, 11.0);
}

/// Directory holding the per-bucket net ONNX files (set by the autorun).
fn model_dir() -> std::path::PathBuf {
    std::env::var_os("STEMLOOP_MODEL_DIR")
        .map_or_else(|| std::path::PathBuf::from("."), std::path::PathBuf::from)
}

/// The mod. The one render in flight lives here; the instance persists across
/// callbacks, so no process statics are needed for it.
#[derive(Default)]
pub struct StemLoop {
    task: Option<Task<RenderJob>>,
    /// Deadline (ms) until which we replay CHOP once the chop view appears; 0 off.
    exit_chop_until: i64,
}

impl StemLoop {
    /// Start a Stem Split on the selected sample, on a worker thread.
    fn start(&mut self) {
        if self.task.is_some() {
            return; // one at a time
        }
        let screen = page::screen();
        let Some(source) = selection::current(screen) else {
            notice("select a sample first");
            return;
        };
        // Refuse (rather than overwrite) when there aren't 4 free pads.
        if !pads::has_room(selection::program(source.state), STEMS.len()) {
            notice("need 4 free pads");
            return;
        }
        let input = match source.read_audio() {
            Ok(audio) => audio,
            Err(_) => {
                notice("cannot read this sample");
                return;
            }
        };
        let channels = usize::from(source.channels);
        let total = usize::try_from(source.frames).unwrap_or(0);
        let frames = total.min(MAX_SECS * SR); // cap the processed length
        // Always the 4s window so there is ONE cached engine that is never
        // rebuilt: a second split (even of a different-length sample) reuses the
        // loaded net and its arena instead of allocating a second ~335 MB net,
        // which fails for want of RAM on the 2 GB device. Shorter loops zero-pad
        // to 4s and are trimmed back; longer loops chunk this 4s window.
        let window = WINDOW_SECS;
        let model = model_dir().join(format!("hdemucs_mmi_{window}s_net.onnx"));
        let job = RenderJob::new(source, KIND_SPLIT, input);
        let task = Task::spawn(screen, job, move |job| {
            split_worker(job, channels, frames, window, &model)
        });
        self.task = Some(task);
        status::set_status("Stem Split: separating...");
        draw::show_toast("Stem Split: separating...");
        draw::repaint_painted();
    }

    /// Take the worker's result (if ready) and write the stems to pads.
    fn finish(&mut self, screen: usize) {
        let Some(mut task) = self.task.take() else {
            return;
        };
        match task.take() {
            Err(TaskError::Running) => {
                self.task = Some(task); // not done yet; keep waiting
            }
            Err(TaskError::Render(code)) => {
                // Name where it failed on-screen (the device has no console).
                notice(match code {
                    2 => "failed: load model",
                    3 => "failed: separate",
                    4 => "failed: write stem",
                    _ => "processing failed",
                });
            }
            Err(_) => {
                notice("processing failed");
            }
            Ok(job) => {
                if finish_ok(&job, screen) {
                    // The app opens the chop view after an Extract; arm a replay
                    // of CHOP to return to the sample page once it appears.
                    self.exit_chop_until = clock::now_ms() + CHOP_EXIT_MS;
                }
            }
        }
    }

    /// Once the app's post-Extract chop view appears, toggle back to the sample
    /// page by replaying CHOP. Runs on the UI thread, bounded by a deadline so it
    /// never fires on a chop view the user opens later, and clears its own flag
    /// before the replay so a reentrant paint cannot replay twice.
    fn leave_chop(&mut self, screen: usize) {
        if self.exit_chop_until == 0 {
            return;
        }
        if clock::now_ms() >= self.exit_chop_until {
            self.exit_chop_until = 0; // the chop view never appeared; give up
        } else if page::chop_mode(screen) {
            // In chop now: clear first (so a reentrant paint cannot replay twice),
            // then replay CHOP to toggle back to the sample page.
            self.exit_chop_until = 0;
            if let Ok(bytes) = TRIGGER.lock() {
                let _ = sites::replay_button(&bytes, KEY_CHOP);
            }
        } else {
            // Not in chop yet: ask for another paint so we keep polling until the
            // app's post-Extract chop view appears (bounded by the deadline). This
            // is a UI-thread repaint request, not a background pump.
            draw::repaint_painted();
        }
    }
}

/// The worker: run the real separation engine and push the four stems.
fn split_worker(
    job: &mut RenderJob,
    channels: usize,
    frames: usize,
    window: usize,
    model: &std::path::Path,
) -> Result<(), i32> {
    // Reuse the cached engine when the window matches; otherwise (re)build it.
    let mut guard = ENGINE.lock().map_err(|_| 2_i32)?;
    if guard.as_ref().is_none_or(|(w, _)| *w != window) {
        *guard = None; // drop the old engine before loading the new net
        trim_heap();
        eprintln!("stemloop: worker: building engine (window={window}s)...");
        let engine = Engine::new(model, window).map_err(|e| {
            eprintln!("stemloop: worker: Engine::new FAILED (code 2): {e}");
            2_i32
        })?;
        *guard = Some((window, engine));
    } else {
        eprintln!("stemloop: worker: reusing cached engine (window={window}s)");
    }
    let (_, engine) = guard.as_mut().ok_or(2_i32)?;
    eprintln!("stemloop: worker: separating...");
    let stems = engine.separate(&job.input, channels, frames).map_err(|_| {
        eprintln!("stemloop: worker: separate FAILED (code 3)");
        3_i32
    })?;
    eprintln!("stemloop: worker: {} stems produced", stems.len());
    for stem in stems {
        let output = RenderOutput::new(stem.audio, stem.channels).ok_or(4_i32)?;
        job.push_output(output);
    }
    // Release the lock, then return this split's transient buffers (STFT/iSTFT,
    // deinterleave) to the OS so the next split starts from the same baseline.
    // The cached engine and its ORT arena stay put and are reused.
    drop(guard);
    trim_heap();
    eprintln!("stemloop: worker: done, requesting repaint");
    // The worker's completion is the only event that reaches an idle UI.
    draw::repaint_painted();
    Ok(())
}

/// Write each rendered stem to the next free pad via the app's Extract command.
///
/// Returns whether any stem was written (so the caller can arm the chop-exit).
fn finish_ok(job: &RenderJob, screen: usize) -> bool {
    if job.source.screen != screen || !job.source.is_current() {
        notice("cancelled");
        return false;
    }
    let mut wrote = 0_usize;
    for (i, output) in job.outputs.iter().enumerate() {
        let Some(&name) = STEMS.get(i) else { break };
        // The stem's own length (bucket-trimmed), so the pad is exactly the clean
        // separated audio -- not the full source length (which bled other audio).
        let n = u32::try_from(output.frames).unwrap_or(0);
        let ok = extract::run(&extract::Request {
            screen: job.source.screen,
            state: job.source.state,
            name,
            buffer: output.audio.get(..),
            source_frames: n,
            channels: output.channels,
            start: 0,
            end: n,
            destination: None,
        })
        .is_ok();
        if ok {
            wrote += 1;
        }
    }
    if wrote == STEMS.len() {
        notice("done: 4 stems created");
    } else if wrote > 0 {
        notice("partial: some stems failed");
    } else {
        notice("could not create pads");
    }
    wrote > 0
}

impl Mod for StemLoop {
    fn on_button(&mut self, event: ButtonEvent) -> ButtonFlow {
        // SHIFT + DATA on the Sample page launches the split.
        if event.code == KEY_DATA && event.pressed && event.shift {
            if let Ok(mut t) = TRIGGER.lock() {
                *t = event.bytes; // keep the raw event for a later CHOP replay
            }
            self.start();
            return ButtonFlow::Consume;
        }
        ButtonFlow::Forward
    }

    fn on_paint(&mut self, graphics: &mut draw::Graphics, event: PaintEvent) {
        if !event.sample_page {
            return;
        }
        // Draw on the UI thread only (no background pump). While a split runs,
        // show the progress banner; otherwise show a recent one-shot notice.
        if self.task.is_some() {
            draw_banner(*graphics, "separation in progress...");
        } else if let Ok(n) = NOTICE.lock() {
            if clock::now_ms() < n.1 && !n.0.is_empty() {
                draw_banner(*graphics, n.0);
            }
        }
        let screen = page::screen();
        self.finish(screen);
        self.leave_chop(screen);
    }
}

mpc_sample_sdk::export_mod!(StemLoop);
