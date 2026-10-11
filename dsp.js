/* ============================================================================
   dsp.js — wingbeat frequency measurement.

   A flying mosquito produces a tone at its wingbeat frequency, with strong
   harmonics at 2x, 3x and so on. The job is to recover the fundamental from a
   short window of audio, in the presence of harmonics and background noise.

   Method, following Mukundarajan et al. (eLife 2017):
     1. Split the recording into short windows (20 ms, as in the paper).
     2. For each window, take an FFT and find the fundamental.
     3. The recording is then a DISTRIBUTION of frequencies across windows,
        not a single number. That distribution is what identifies a species.

   Finding the fundamental needs care. The loudest peak is often a harmonic,
   not the fundamental, so a naive "argmax of the spectrum" reports 2x or 3x
   the true frequency. The Harmonic Product Spectrum fixes this: downsample the
   spectrum by 2, 3, 4 and multiply. Harmonics of the true fundamental line up
   and reinforce; everything else is suppressed.

   Pure DSP, no DOM, no browser APIs, so it can be tested in Node.
   ========================================================================== */

(function (root) {
  "use strict";

  /* ---------------------------------------------------------------- FFT
     Iterative radix-2 Cooley-Tukey. n must be a power of two. */
  function fft(re, im) {
    var n = re.length;
    // bit-reversal permutation
    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tr = re[i]; re[i] = re[j]; re[j] = tr;
        var ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len;
      var wr = Math.cos(ang), wi = Math.sin(ang);
      for (var s = 0; s < n; s += len) {
        var cr = 1, ci = 0;
        for (var k = 0; k < len / 2; k++) {
          var ur = re[s + k], ui = im[s + k];
          var vr = re[s + k + len / 2] * cr - im[s + k + len / 2] * ci;
          var vi = re[s + k + len / 2] * ci + im[s + k + len / 2] * cr;
          re[s + k] = ur + vr;  im[s + k] = ui + vi;
          re[s + k + len / 2] = ur - vr;  im[s + k + len / 2] = ui - vi;
          var ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = ncr;
        }
      }
    }
  }

  function nextPow2(n) { var p = 1; while (p < n) p <<= 1; return p; }

  /* Hann window reduces spectral leakage, which otherwise smears the peak. */
  function hann(n) {
    var w = new Float64Array(n);
    for (var i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
    return w;
  }

  /* Magnitude spectrum of one window, zero-padded for finer frequency steps. */
  function spectrum(samples, pad) {
    var n = samples.length;
    var N = nextPow2(n) * (pad || 1);
    var re = new Float64Array(N), im = new Float64Array(N);
    var w = hann(n);
    for (var i = 0; i < n; i++) re[i] = samples[i] * w[i];
    fft(re, im);
    var half = N >> 1;
    var mag = new Float64Array(half);
    for (var k = 0; k < half; k++) mag[k] = Math.hypot(re[k], im[k]);
    return { mag: mag, N: N };
  }

  /* Parabolic interpolation around a peak gives sub-bin frequency accuracy,
     which matters because one FFT bin can be 20 Hz or more at these windows. */
  function refine(mag, k) {
    if (k <= 0 || k >= mag.length - 1) return k;
    var a = mag[k - 1], b = mag[k], c = mag[k + 1];
    var d = a - 2 * b + c;
    if (d === 0) return k;
    return k + 0.5 * (a - c) / d;
  }

  /* ------------------------------------------------- fundamental frequency
     Returns { freq, strength, snr } or null when the window is not a tone. */
  function fundamental(samples, sampleRate, opts) {
    opts = opts || {};
    var fmin = opts.fmin || 150;      // below this is room rumble, not a mosquito
    var fmax = opts.fmax || 1000;     // above this is not a wingbeat fundamental
    var harmonics = opts.harmonics || 5;
    var pad = opts.pad || 4;

    var sp = spectrum(samples, pad);
    var mag = sp.mag, N = sp.N;
    var binHz = sampleRate / N;

    // Harmonic product spectrum over the search band
    var kmin = Math.max(1, Math.floor(fmin / binHz));
    var kmax = Math.min(mag.length - 1, Math.ceil(fmax / binHz));
    if (kmax <= kmin) return null;

    /* A plain product fails the moment one harmonic is missing or lands above
       Nyquist: a single zero wipes out the whole score, and the fundamental is
       reported as nothing. Real recordings do this constantly, because higher
       harmonics fade first. So accumulate the SUM OF LOGS with a small floor,
       which is monotonically the same as the product but survives a weak or
       absent harmonic, and average over the harmonics actually available. */
    var floorMag = 1e-9;
    for (var fk = kmin; fk <= kmax; fk++) floorMag += mag[fk];
    floorMag = (floorMag / (kmax - kmin + 1)) * 0.05;     // 5% of the band mean

    var best = -1, bestVal = -Infinity;
    var hps = new Float64Array(kmax + 1);
    for (var k = kmin; k <= kmax; k++) {
      var score = 0, used = 0;
      for (var h = 1; h <= harmonics; h++) {
        var idx = Math.round(k * h);
        if (idx >= mag.length) break;                     // past Nyquist, stop
        score += Math.log(mag[idx] + floorMag);
        used++;
      }
      if (used < 2) continue;                             // need at least f0 and 2f0
      score /= used;
      hps[k] = score;
      if (score > bestVal) { bestVal = score; best = k; }
    }
    if (best < 0 || !isFinite(bestVal)) return null;

    /* Even so, HPS can settle on the second harmonic when the fundamental is
       very weak. If half the chosen frequency scores nearly as well, it is the
       real fundamental and the peak we found is its octave. */
    var halfK = Math.round(best / 2);
    if (halfK >= kmin && hps[halfK] !== 0 &&
        hps[halfK] > bestVal - Math.abs(bestVal) * 0.08) {
      best = halfK;
    }

    var kr = refine(mag, best);
    var freq = kr * binHz;

    /* Signal-to-noise: peak energy against the median of the band. A real
       wingbeat is a clear tone; room noise is broadband and fails this. */
    var band = [];
    for (var b = kmin; b <= kmax; b++) band.push(mag[b]);
    band.sort(function (x, y) { return x - y; });
    var median = band[Math.floor(band.length / 2)] || 1e-12;
    var snr = mag[best] / median;

    return { freq: freq, strength: mag[best], snr: snr };
  }

  /* -------------------------------------------------------- full recording
     Splits into windows and returns the frequency distribution, which is the
     quantity that actually identifies a species. */
  function analyse(samples, sampleRate, opts) {
    opts = opts || {};
    var winMs = opts.windowMs || 20;              // the eLife window length
    var hopMs = opts.hopMs || 10;
    var minSnr = opts.minSnr === undefined ? 6 : opts.minSnr;

    var win = Math.max(64, Math.round(sampleRate * winMs / 1000));
    var hop = Math.max(1, Math.round(sampleRate * hopMs / 1000));

    var freqs = [], all = [], kept = 0, total = 0;
    for (var s = 0; s + win <= samples.length; s += hop) {
      total++;
      var seg = samples.subarray ? samples.subarray(s, s + win) : samples.slice(s, s + win);
      // skip silence outright
      var rms = 0;
      for (var i = 0; i < seg.length; i++) rms += seg[i] * seg[i];
      rms = Math.sqrt(rms / seg.length);
      if (rms < (opts.minRms || 1e-4)) { all.push(null); continue; }

      var f = fundamental(seg, sampleRate, opts);
      all.push(f);
      if (f && f.snr >= minSnr) { freqs.push(f.freq); kept++; }
    }

    freqs.sort(function (a, b) { return a - b; });
    return {
      freqs: freqs,
      frames: all,
      windowsTotal: total,
      windowsKept: kept,
      median: pct(freqs, 50),
      p5: pct(freqs, 5),
      p95: pct(freqs, 95),
      mean: freqs.length ? freqs.reduce(function (a, b) { return a + b; }, 0) / freqs.length : null,
      sd: sd(freqs),
    };
  }

  function pct(sorted, p) {
    if (!sorted.length) return null;
    var i = (p / 100) * (sorted.length - 1);
    var lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }
  function sd(a) {
    if (a.length < 2) return null;
    var m = a.reduce(function (x, y) { return x + y; }, 0) / a.length;
    return Math.sqrt(a.reduce(function (s, x) { return s + (x - m) * (x - m); }, 0) / (a.length - 1));
  }

  /* histogram for display */
  function histogram(freqs, lo, hi, bins) {
    lo = lo || 150; hi = hi || 1000; bins = bins || 34;
    var h = new Array(bins).fill(0), w = (hi - lo) / bins;
    freqs.forEach(function (f) {
      var b = Math.floor((f - lo) / w);
      if (b >= 0 && b < bins) h[b]++;
    });
    return { counts: h, lo: lo, hi: hi, width: w };
  }

  root.DSP = {
    fft: fft, spectrum: spectrum, fundamental: fundamental,
    analyse: analyse, histogram: histogram, percentile: pct,
  };

})(typeof window !== "undefined" ? window : globalThis);
