/* ============================================================================
   app.js — Wingbeat Lab: a field instrument for mosquito acoustic surveillance.

   This is NOT a species identifier. It is the thing you need before one can
   exist: a calibrated wingbeat-frequency meter that records labelled audio and
   exports it, so a Tamil Nadu training set can be built.

   The honest reason for that choice is in the published work. Phone recordings
   only reach a mosquito at 5-10 cm, below about 50 dB of background noise, and
   a classifier trained elsewhere reached roughly 35% accuracy in the field
   without a location prior. A species model for Indian vectors needs Indian
   recordings, and nobody has collected them.

   Designed by Dr. Udayakumar Mani, SASTRA Deemed to be University.
   ========================================================================== */

(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    ctx: null, stream: null, source: null, proc: null,
    live: false, recording: false,
    chunks: [], sampleRate: 44100,
    lastResult: null, lastAudio: null, lastML: null,
    sessions: [],
  };

  /* Reference bands. These are PROVISIONAL, from the general statement in the
     literature that female wingbeat frequencies fall between 200 and 700 Hz and
     that a species distribution spans roughly 150-200 Hz. They are NOT
     per-species measurements. Replace them with distributions fitted from your
     own recordings, or from the Stanford Dryad dataset, before claiming any
     identification. The app says so on screen for exactly this reason. */
  var BANDS = [
    { name: "Below the wingbeat band", lo: 0,   hi: 200, note: "Usually room rumble, fans, traffic. Not a mosquito fundamental." },
    { name: "Lower wingbeat band",     lo: 200, hi: 400, note: "Overlaps several Anopheles and larger species. Genus-level at best." },
    { name: "Middle wingbeat band",    lo: 400, hi: 600, note: "Where most Aedes and Culex females fall. Species cannot be separated here without a fitted model." },
    { name: "Upper wingbeat band",     lo: 600, hi: 800, note: "Smaller individuals, and males of several species, which beat faster than females." },
    { name: "Above the usual band",    lo: 800, hi: 2000, note: "Probably a harmonic, an insect that is not a mosquito, or electrical noise." },
  ];

  function bandFor(f) {
    for (var i = 0; i < BANDS.length; i++) if (f >= BANDS[i].lo && f < BANDS[i].hi) return BANDS[i];
    return BANDS[BANDS.length - 1];
  }

  /* ------------------------------------------------------------ microphone */
  async function startLive() {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,      // all three would destroy a faint tone
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
    } catch (e) {
      say("Microphone refused: " + e.message + ". On a phone the page must be on https, and you must allow the microphone.", true);
      return;
    }
    state.ctx = new (window.AudioContext || window.webkitAudioContext)();
    state.sampleRate = state.ctx.sampleRate;
    state.source = state.ctx.createMediaStreamSource(state.stream);

    var size = 2048;
    state.proc = state.ctx.createScriptProcessor(size, 1, 1);
    state.source.connect(state.proc);
    state.proc.connect(state.ctx.destination);   // required in some browsers
    state.proc.onaudioprocess = onAudio;

    state.live = true;
    $("micBtn").textContent = "Stop microphone";
    $("recBtn").disabled = false;
    say("Microphone running at " + Math.round(state.sampleRate) + " Hz. " +
        "Hold the phone within 5 to 10 cm of the mosquito.");
  }

  function stopLive() {
    if (state.proc) { state.proc.disconnect(); state.proc.onaudioprocess = null; }
    if (state.source) state.source.disconnect();
    if (state.stream) state.stream.getTracks().forEach(function (t) { t.stop(); });
    if (state.ctx) state.ctx.close();
    state.live = false; state.recording = false;
    $("micBtn").textContent = "Start microphone";
    $("recBtn").disabled = true;
    $("recBtn").textContent = "Record 8 seconds";
    say("Microphone stopped.");
  }

  function onAudio(e) {
    var buf = e.inputBuffer.getChannelData(0);

    // live readout
    var f = DSP.fundamental(buf, state.sampleRate, { fmin: 150, fmax: 1200 });
    var rms = 0;
    for (var i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / buf.length);
    drawLevel(rms);

    if (f && f.snr >= 6 && rms > 2e-4) {
      $("liveHz").textContent = f.freq.toFixed(0);
      $("liveSnr").textContent = "signal " + f.snr.toFixed(1) + "x above noise";
      $("liveBand").textContent = bandFor(f.freq).name;
      $("liveHz").classList.add("on");
    } else {
      $("liveHz").textContent = "—";
      $("liveSnr").textContent = rms > 2e-4 ? "no clear tone" : "too quiet";
      $("liveBand").textContent = "";
      $("liveHz").classList.remove("on");
    }

    if (state.recording) {
      state.chunks.push(new Float32Array(buf));
      var secs = state.chunks.length * buf.length / state.sampleRate;
      $("recBtn").textContent = "Recording… " + secs.toFixed(1) + " s";
      if (secs >= 8) finishRecording();
    }
  }

  /* ------------------------------------------------------------- recording */
  function startRecording() {
    state.chunks = [];
    state.recording = true;
    $("recBtn").textContent = "Recording…";
    say("Recording. Keep the phone still and close.");
  }

  function finishRecording() {
    state.recording = false;
    $("recBtn").textContent = "Record 8 seconds";

    var total = state.chunks.reduce(function (n, c) { return n + c.length; }, 0);
    var audio = new Float32Array(total);
    var off = 0;
    state.chunks.forEach(function (c) { audio.set(c, off); off += c.length; });
    state.lastAudio = audio;

    analyseAndShow(audio, state.sampleRate);
  }

  function analyseAndShow(audio, sr) {
    var r = DSP.analyse(audio, sr, { windowMs: 20, hopMs: 10, minSnr: 6 });
    state.lastResult = r;

    /* The model panel is offered for every recording, including ones where the
       frequency measurement found nothing. The two methods disagree sometimes,
       and that disagreement is information: a network trained on real
       recordings can pick up a mosquito whose fundamental is too buried for a
       peak-picking algorithm to call. */
    state.lastML = null;
    $("mlPanel").hidden = false;
    $("mlBody").innerHTML = '<p class="say">Two small networks, 92 KB each, run ' +
      'entirely on this phone. Nothing is uploaded.</p>';
    mlSay("");

    if (!r.freqs.length) {
      $("result").hidden = false;
      $("stats").innerHTML = '<p class="warn">No wingbeat tone found in this recording. ' +
        'Usually that means the mosquito was too far away, the room was too noisy, ' +
        'or it was not flying. Get within 5 to 10 cm and try again.</p>';
      $("hist").innerHTML = "";
      $("saveBtn").disabled = true;
      return;
    }

    $("result").hidden = false;
    $("saveBtn").disabled = false;

    var band = bandFor(r.median);
    $("stats").innerHTML =
      '<div class="big"><b>' + r.median.toFixed(0) + '</b> Hz <span>median wingbeat</span></div>' +
      '<table class="st">' +
      row("Spread, 5th to 95th percentile", r.p5.toFixed(0) + " to " + r.p95.toFixed(0) + " Hz  (" + (r.p95 - r.p5).toFixed(0) + " Hz wide)") +
      row("Mean, standard deviation", r.mean.toFixed(0) + " Hz, " + (r.sd ? r.sd.toFixed(0) : "–") + " Hz") +
      row("Windows with a clear tone", r.windowsKept + " of " + r.windowsTotal +
          "  (" + (100 * r.windowsKept / r.windowsTotal).toFixed(0) + "%)") +
      row("Band", band.name) +
      '</table>' +
      '<p class="note">' + band.note + '</p>' +
      '<p class="caution"><b>This is a frequency measurement, not an identification.</b> ' +
      'Published field accuracy for species classification from phone audio is around 35% ' +
      'without a location prior, and about 65% with one. Record the true species from a ' +
      'microscope where you can, and use these recordings to train a model for your own sites.</p>';

    drawHistogram(r.freqs);
  }

  function row(k, v) { return "<tr><td>" + k + "</td><td>" + v + "</td></tr>"; }

  /* --------------------------------------------------------- trained model
     The frequency measurement above is arithmetic and always available. This
     is the learned part, and it is kept in its own panel on purpose: the two
     answer different questions, and one of them carries a caveat about where
     its training data came from.                                            */
  function pbar(p, hit) {
    return '<div class="pbar"><i class="' + (hit ? "" : "lo") +
           '" style="width:' + (100 * p).toFixed(1) + '%"></i></div>';
  }

  function showML(r) {
    state.lastML = r;
    var H = "";

    if (r.empty) {
      H = '<p class="warn">Every window in this recording was silent, so there was ' +
          'nothing to score. Check that the microphone was actually capturing.</p>';
      $("mlBody").innerHTML = H;
      return;
    }

    var detected = r.hitCount > 0;
    var pctWin = 100 * r.hitFraction;

    H += '<div class="verdict"><b class="' + (detected ? "" : "no") + '">' +
         (detected ? "Mosquito detected" : "No mosquito detected") + '</b>' +
         '<span>in ' + r.hitCount + ' of ' + r.windows.length +
         ' one-second windows (' + pctWin.toFixed(0) + '% of the recording)</span></div>';

    H += '<table class="st">' +
         row("Mosquito-seconds", r.mosquitoSeconds.toFixed(1) + " s" +
             (detected ? "" : " (none)")) +
         row("Strongest window", (100 * r.peakMosquito).toFixed(1) + "% confident" +
             pbar(r.peakMosquito, r.peakMosquito >= r.threshold)) +
         row("Typical window", (100 * r.medianMosquito).toFixed(1) + "% confident" +
             pbar(r.medianMosquito, r.medianMosquito >= r.threshold));

    if (r.culexOnHits !== null) {
      H += row("Culex quinquefasciatus, on the detected windows",
               (100 * r.culexOnHits).toFixed(1) + "% confident" +
               pbar(r.culexOnHits, r.culexOnHits >= r.threshold));
    } else {
      H += row("Culex quinquefasciatus",
               "not asked, because no window held a mosquito");
    }
    if (r.quiet) {
      H += row("Windows skipped as silent", r.quiet + " of " + r.total);
    }
    H += '</table>';

    // one bar per window, so a single pass is visible rather than averaged away
    H += '<div class="strip">' + r.windows.map(function (w) {
      return '<div class="' + (w.mosquito >= r.threshold ? "hit" : "") +
             '" style="height:' + Math.max(2, 100 * w.mosquito).toFixed(0) + '%" ' +
             'title="' + w.tSec.toFixed(1) + ' s: ' + (100 * w.mosquito).toFixed(0) +
             '% mosquito, ' + (100 * w.culex).toFixed(0) + '% Culex"></div>';
    }).join("") + '</div>' +
    '<div class="sl"><span>0 s</span><span>confidence per window</span><span>' +
    (r.windows.length ? (r.windows[r.windows.length - 1].tSec + 1).toFixed(0) : "0") +
    ' s</span></div>';

    /* Provenance always; held-out scores only if they are really there.
       A null or a blank must NOT reach toFixed, because null coerces to 0 and
       the screen would then read "ROC AUC 0.000" as though that had been
       measured. An invented number is worse than an absent one, so anything
       that is not a finite number in range is treated as absent. */
    var e = r.report;
    var score = function (sect, key) {
      var v = sect ? sect[key] : null;
      return (typeof v === "number" && isFinite(v) && v > 0 && v <= 1) ? v : null;
    };
    var got = [
      ["Detector, held-out ROC AUC", score(e && e.detector, "roc_auc")],
      ["Detector, held-out PR AUC", score(e && e.detector, "pr_auc")],
      ["Culex head, held-out ROC AUC", score(e && e.culex, "roc_auc")],
      ["Culex head, held-out PR AUC", score(e && e.culex, "pr_auc")],
    ].filter(function (p) { return p[1] !== null; });

    if (got.length) {
      H += '<table class="st" style="margin-top:12px">' +
           got.map(function (p) { return row(p[0], p[1].toFixed(3)); }).join("") +
           '</table>' +
           '<p class="note">Held out by recording, never by window, so these are ' +
           'honest numbers and not the same clip scored twice.</p>';
    } else {
      H += '<p class="caution"><b>The accuracy of these models has not been ' +
           'recorded in this app yet.</b> Until the held-out scores from the ' +
           'training run are published alongside the model files, treat the ' +
           'percentages above as a signal to investigate, not a measurement.</p>';
    }

    H += '<p class="caution"><b>Neither model has ever heard an Indian mosquito.</b> ' +
         'Both were trained on HumBugDB, recorded in Tanzania, Thailand, the UK, Kenya ' +
         'and the USA. Wingbeat frequency follows body size, and body size differs ' +
         'between populations. The detector should travel reasonably well because it ' +
         'is species independent. The <i>Culex</i> answer is a starting point that your ' +
         'own labelled recordings will correct. See the section below for the numbers ' +
         'behind that warning.</p>';

    $("mlBody").innerHTML = H;
  }

  async function runML() {
    if (!state.lastAudio) return;
    $("mlBtn").disabled = true;
    mlSay("Loading the models (227 KB, once only)…");
    try {
      var r = await Classify.run(state.lastAudio, state.sampleRate,
        function (done, total) {
          mlSay("Scoring window " + done + " of " + total + "…");
        });
      showML(r);
      mlSay("Done. " + r.windows.length + " window" +
            (r.windows.length === 1 ? "" : "s") + " scored on this device. " +
            "Nothing was uploaded.");
    } catch (e) {
      mlSay("The model could not run: " + e.message, true);
    }
    $("mlBtn").disabled = false;
  }

  function mlSay(msg, bad) {
    $("mlSay").textContent = msg;
    $("mlSay").className = "say" + (bad ? " bad" : "");
  }

  /* --------------------------------------------------------------- drawing */
  function drawLevel(rms) {
    var pct = Math.min(100, Math.max(0, (20 * Math.log10(rms + 1e-9) + 60) / 60 * 100));
    $("level").style.width = pct + "%";
  }

  function drawHistogram(freqs) {
    var h = DSP.histogram(freqs, 150, 1000, 34);
    var max = Math.max.apply(null, h.counts) || 1;
    var html = '<div class="hwrap">';
    h.counts.forEach(function (c, i) {
      var f = h.lo + i * h.width;
      html += '<div class="hbar" style="height:' + (100 * c / max) + '%" ' +
              'title="' + f.toFixed(0) + '-' + (f + h.width).toFixed(0) + ' Hz: ' + c + '"></div>';
    });
    html += '</div><div class="haxis"><span>150</span><span>400</span><span>600</span><span>800</span><span>1000 Hz</span></div>';
    $("hist").innerHTML = html;
  }

  /* ------------------------------------------------------------ self test */
  function selfTest() {
    /* Four seconds, not one. One second is enough to check the frequency
       measurement, but the trained model scores one-second windows with a
       half-second hop, so a one-second test gives it a single window and the
       per-window display has nothing to show. Four seconds gives seven. */
    var sr = 44100, secs = 4.0, f0 = 450;
    var n = Math.round(sr * secs), x = new Float32Array(n), phase = 0;
    for (var i = 0; i < n; i++) {
      var t = i / sr;
      var f = f0 + 20 * Math.sin(2 * Math.PI * 3 * t);     // wobble like a real insect
      phase += 2 * Math.PI * f / sr;
      var v = 0;
      for (var hh = 1; hh <= 6; hh++) v += Math.pow(0.6, hh - 1) * Math.sin(hh * phase);
      x[i] = 0.3 * v + 0.05 * (Math.random() * 2 - 1);
    }
    state.lastAudio = x;
    state.sampleRate = sr;
    analyseAndShow(x, sr);
    say("Self test: four seconds of a synthetic 450 Hz wingbeat with harmonics and " +
        "noise. If the median reads close to 450 Hz, the measurement chain is working. " +
        "You can also run the trained model on it below.");
  }

  /* ------------------------------------------------------------- exporting */
  function saveSession() {
    var r = state.lastResult;
    if (!r) return;
    var s = {
      id: "rec-" + Date.now(),
      time: new Date().toISOString(),
      species: $("species").value.trim() || "unknown",
      sex: $("sex").value,
      site: $("site").value.trim(),
      tempC: $("temp").value.trim(),
      method: $("method").value,
      notes: $("notes").value.trim(),
      lat: state.lat || "", lon: state.lon || "",
      device: navigator.userAgent,
      sampleRate: Math.round(state.sampleRate),
      medianHz: +r.median.toFixed(1),
      p5Hz: +r.p5.toFixed(1),
      p95Hz: +r.p95.toFixed(1),
      meanHz: +r.mean.toFixed(1),
      sdHz: r.sd ? +r.sd.toFixed(1) : "",
      windowsKept: r.windowsKept,
      windowsTotal: r.windowsTotal,
      freqs: r.freqs.map(function (f) { return +f.toFixed(1); }),
    };

    /* If the model was run, its answer is saved WITH the recording. That is
       what makes the export useful later: once there are real Tamil Nadu
       labels, these stored predictions show where the HumBugDB-trained model
       was right and where it was wrong, which is the measurement nobody has
       made yet. Blank columns where it was not run, rather than zeros. */
    var m = state.lastML;
    if (m && !m.empty) {
      s.mlHitFraction = +m.hitFraction.toFixed(3);
      s.mlMosquitoSeconds = +m.mosquitoSeconds.toFixed(1);
      s.mlPeak = +m.peakMosquito.toFixed(4);
      s.mlMedian = +m.medianMosquito.toFixed(4);
      s.mlCulex = m.culexOnHits === null ? "" : +m.culexOnHits.toFixed(4);
      s.mlWindows = m.windows.map(function (w) {
        return { t: +w.tSec.toFixed(1), m: +w.mosquito.toFixed(4), c: +w.culex.toFixed(4) };
      });
      s.mlTrainedOn = "HumBugDB, no Indian recordings";
    }
    state.sessions.push(s);
    try { localStorage.setItem("wingbeat.sessions", JSON.stringify(state.sessions)); } catch (e) {}
    renderSessions();
    say("Saved. " + state.sessions.length + " recording" + (state.sessions.length === 1 ? "" : "s") + " in this session.");
  }

  function renderSessions() {
    var n = state.sessions.length;
    $("count").textContent = n ? n + " saved" : "none saved yet";
    $("exportCsv").disabled = !n;
    $("exportJson").disabled = !n;
    if (!n) { $("list").innerHTML = ""; return; }
    $("list").innerHTML = state.sessions.slice().reverse().map(function (s) {
      return '<li><b>' + esc(s.species) + '</b> ' + s.medianHz + ' Hz ' +
             '<span>' + esc(s.site || "no site") + ' · ' + s.time.slice(11, 16) + '</span></li>';
    }).join("");
  }

  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; });
  }

  function download(name, text, mime) {
    var b = new Blob([text], { type: mime || "text/plain" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = name;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  function exportCsv() {
    var cols = ["id", "time", "species", "sex", "site", "lat", "lon", "tempC", "method",
                "medianHz", "p5Hz", "p95Hz", "meanHz", "sdHz", "windowsKept", "windowsTotal",
                "mlHitFraction", "mlMosquitoSeconds", "mlPeak", "mlMedian", "mlCulex",
                "sampleRate", "notes"];
    var lines = [cols.join(",")];
    state.sessions.forEach(function (s) {
      lines.push(cols.map(function (c) {
        var v = s[c] === undefined ? "" : String(s[c]);
        return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
      }).join(","));
    });
    download("wingbeat-sessions.csv", lines.join("\n"), "text/csv");
  }

  function exportJson() {
    download("wingbeat-sessions.json", JSON.stringify(state.sessions, null, 2), "application/json");
  }

  /* 16-bit PCM WAV, so the raw audio can be used to train a model later.
     The measurements are useful; the audio is what is irreplaceable. */
  function exportWav() {
    var x = state.lastAudio;
    if (!x) return;
    var sr = Math.round(state.sampleRate);
    var buf = new ArrayBuffer(44 + x.length * 2);
    var v = new DataView(buf);
    function str(o, s) { for (var i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); }
    str(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, x.length * 2, true);
    for (var i = 0; i < x.length; i++) {
      var s = Math.max(-1, Math.min(1, x[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    var name = ($("species").value.trim() || "unknown").replace(/\s+/g, "_") +
               "_" + Date.now() + ".wav";
    download(name, buf, "audio/wav");
  }

  function say(msg, bad) {
    $("say").textContent = msg;
    $("say").className = "say" + (bad ? " bad" : "");
  }

  /* -------------------------------------------------------------- location */
  function getLocation() {
    if (!navigator.geolocation) { say("This device has no location service.", true); return; }
    navigator.geolocation.getCurrentPosition(function (p) {
      state.lat = p.coords.latitude.toFixed(5);
      state.lon = p.coords.longitude.toFixed(5);
      $("loc").textContent = state.lat + ", " + state.lon;
      say("Location recorded. It will be attached to the next saved recording.");
    }, function (e) {
      say("Location refused: " + e.message, true);
    }, { enableHighAccuracy: true, timeout: 10000 });
  }

  /* ------------------------------------------------------------------ wire */
  $("micBtn").addEventListener("click", function () { state.live ? stopLive() : startLive(); });
  $("recBtn").addEventListener("click", function () { if (!state.recording) startRecording(); });
  $("testBtn").addEventListener("click", selfTest);
  $("mlBtn").addEventListener("click", runML);
  $("saveBtn").addEventListener("click", saveSession);
  $("wavBtn").addEventListener("click", exportWav);
  $("exportCsv").addEventListener("click", exportCsv);
  $("exportJson").addEventListener("click", exportJson);
  $("locBtn").addEventListener("click", getLocation);
  $("clearBtn").addEventListener("click", function () {
    if (!state.sessions.length) return;
    state.sessions = [];
    try { localStorage.removeItem("wingbeat.sessions"); } catch (e) {}
    renderSessions();
    say("Cleared.");
  });

  try {
    state.sessions = JSON.parse(localStorage.getItem("wingbeat.sessions") || "[]") || [];
  } catch (e) { state.sessions = []; }
  renderSessions();

  /* A small deliberate opening in the IIFE. Everything above is private, but
     being able to push a known signal through the whole chain and read the
     result back is what makes this instrument checkable rather than merely
     plausible. The automated browser tests use it, and so can anyone
     replaying a stored WAV through the measurement and the model. */
  window.Wingbeat = {
    inject: function (audio, sr) {
      state.lastAudio = audio;
      state.sampleRate = sr;
      analyseAndShow(audio, sr);
      return { median: state.lastResult ? state.lastResult.median : null };
    },
    runModel: runML,
    lastResult: function () { return state.lastResult; },
    lastModel: function () { return state.lastML; },
  };

  if ("serviceWorker" in navigator &&
      (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("sw.js").catch(function () {});
  }
})();
