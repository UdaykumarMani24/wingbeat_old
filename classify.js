/* ============================================================================
   classify.js — runs the two trained models over a whole recording.

   The models take exactly one second of audio at 8 kHz. A recording is eight
   seconds at 44.1 or 48 kHz, so this file does the three things in between:
   resample once, cut one-second windows with a half-second hop, and
   aggregate the fifteen answers into one.

   HOW THE AGGREGATION WORKS, AND WHY
   A mosquito does not fly continuously past a microphone for eight seconds.
   It passes. So the mean probability across windows is the wrong summary: a
   single clear pass in fifteen windows is a detection, and the mean would
   bury it. This reports the fraction of windows over threshold, which is
   also the quantity a surveillance programme actually wants - it is
   mosquito-seconds per recording, and summed over a night it is
   mosquito-minutes, the standard trap statistic.

   WHAT THE MODELS ARE, HONESTLY
   Both were trained on HumBugDB, which holds no Indian recordings at all.
   The detector ("is a mosquito flying near this microphone") is trained on
   18.3 hours of mosquito and 14.6 hours of background, so it has real data
   behind it. The Culex quinquefasciatus head has 124 minutes from Tanzania,
   Thailand, the UK, Kenya and the USA. Body size sets wingbeat frequency and
   body size varies between populations, so the Culex answer is a starting
   point for Tamil Nadu, not a result. The app says this on screen, because
   a number with no provenance is worse than no number.

   Designed by Dr. Udayakumar Mani, SASTRA Deemed to be University.
   ========================================================================== */

(function (root) {
  "use strict";

  var THRESH = 0.5;         // the training threshold; the readout shows raw
                            // probabilities too, so this only shapes counting
  var WIN_S = 1.0, HOP_S = 0.5;

  var loaded = null;        // a promise, so a second press does not refetch

  function loadAll() {
    if (loaded) return loaded;
    loaded = (async function () {
      var base = "models/";
      var fbResp = await fetch(base + "melfb.bin");
      if (!fbResp.ok) throw new Error("melfb.bin missing (" + fbResp.status + ")");
      var fbBuf = await fbResp.arrayBuffer();
      var melfb = new Float32Array(fbBuf);
      if (melfb.length !== root.Features.NMELS * root.Features.NBINS) {
        throw new Error("melfb.bin is the wrong size: " + melfb.length);
      }
      var detector = await root.MosquitoModel.load(base + "detector.json", base + "detector.bin");
      var culex = await root.MosquitoModel.load(base + "culex.json", base + "culex.bin");

      /* models/report.json holds the held-out scores from the training run:
         ROC AUC, PR AUC, the confusion matrix. It is OPTIONAL on purpose. If
         it is absent the app says the scores are not recorded, rather than
         showing a probability as though its reliability were known. An
         unlabelled number on a screen is read as a fact. */
      var report = null;
      try {
        var rr = await fetch(base + "report.json", { cache: "no-cache" });
        if (rr.ok) report = await rr.json();
      } catch (e) { /* offline, or not published yet; both are fine */ }

      return { melfb: melfb, detector: detector, culex: culex, report: report };
    })();
    return loaded;
  }

  /* How many seconds of the recording actually held a mosquito.

     This cannot be hits * hop. The windows overlap by half a second, so two
     adjacent hits cover 1.5 s between them and not 1.0 s, while two hits at
     opposite ends of the clip cover 2.0 s. Counting windows instead of time
     would quietly inflate or deflate mosquito-minutes depending on whether
     the insect made one long pass or several short ones, which is exactly
     the comparison a surveillance programme is trying to make. So this
     merges the overlapping one-second intervals and measures their union. */
  function coveredSeconds(hits) {
    if (!hits.length) return 0;
    var iv = hits.map(function (w) { return [w.tSec, w.tSec + WIN_S]; })
                 .sort(function (a, b) { return a[0] - b[0]; });
    var total = 0, s = iv[0][0], e = iv[0][1];
    for (var i = 1; i < iv.length; i++) {
      if (iv[i][0] <= e) { if (iv[i][1] > e) e = iv[i][1]; }
      else { total += e - s; s = iv[i][0]; e = iv[i][1]; }
    }
    return total + (e - s);
  }

  /* Runs the models over one recording.
     audio: Float32Array at srIn. onProgress(done, total) is optional.
     Returns a promise, and yields to the browser between windows so the page
     does not freeze - fifteen windows is about 90 million multiply-adds.    */
  async function run(audio, srIn, onProgress) {
    var M = await loadAll();
    var F = root.Features;

    var y = F.resampleTo8k(audio, srIn);
    var n = Math.round(F.SR * WIN_S), hop = Math.round(F.SR * HOP_S);
    if (y.length < n) throw new Error("recording is shorter than one second");

    var windows = [];
    for (var s = 0; s + n <= y.length; s += hop) windows.push(s);

    var out = [], quiet = 0;
    for (var i = 0; i < windows.length; i++) {
      var seg = y.subarray(windows[i], windows[i] + n);

      // a silent window teaches the model nothing and it will still answer
      // confidently, so skip it exactly as the training script did
      var peak = 0;
      for (var k = 0; k < seg.length; k++) {
        var a = seg[k] < 0 ? -seg[k] : seg[k];
        if (a > peak) peak = a;
      }
      if (peak < 1e-4) { quiet++; continue; }

      var lm = F.logMel(seg, M.melfb);
      var dm = M.detector.meta.norm, cm = M.culex.meta.norm;
      var pd = M.detector.predict(F.toInput(lm.data, lm.frames, dm.mean, dm.sd));
      var pc = M.culex.predict(F.toInput(lm.data, lm.frames, cm.mean, cm.sd));
      out.push({ tSec: windows[i] / F.SR, mosquito: pd, culex: pc });

      if (onProgress) onProgress(i + 1, windows.length);
      await new Promise(function (r) { setTimeout(r, 0); });   // let the UI breathe
    }

    if (!out.length) {
      return { windows: [], quiet: quiet, total: windows.length, empty: true };
    }

    var hits = out.filter(function (w) { return w.mosquito >= THRESH; });
    var med = function (arr) {
      var v = arr.slice().sort(function (a, b) { return a - b; });
      if (!v.length) return null;
      var m = Math.floor(v.length / 2);
      return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
    };

    return {
      windows: out,
      quiet: quiet,
      total: windows.length,
      // the headline: how much of the recording had a mosquito in it
      hitFraction: hits.length / out.length,
      mosquitoSeconds: coveredSeconds(hits),
      peakMosquito: Math.max.apply(null, out.map(function (w) { return w.mosquito; })),
      medianMosquito: med(out.map(function (w) { return w.mosquito; })),
      // Culex is only meaningful on windows the detector called a mosquito.
      // Asking "is this Culex" about a window of empty room noise is a
      // question with no answer, and the model will still give one.
      culexOnHits: hits.length ? med(hits.map(function (w) { return w.culex; })) : null,
      hitCount: hits.length,
      threshold: THRESH,
      report: M.report,
    };
  }

  root.Classify = { run: run, load: loadAll, THRESH: THRESH, WIN_S: WIN_S, HOP_S: HOP_S };

})(typeof window !== "undefined" ? window : globalThis);
