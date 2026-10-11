/* ============================================================================
   features.js — turns a second of audio into the log-mel spectrogram the
   model was trained on.

   This has to match librosa exactly. A model fed slightly different features
   than it was trained on does not fail loudly; it just returns confident
   nonsense. So three decisions here are deliberate:

     1. The mel filterbank is NOT computed in JavaScript. It is exported from
        librosa itself and shipped as melfb.bin (40 x 257 float32, 40 KB).
        That removes every chance of a mel-scale or normalisation mismatch.
     2. center = true padding is reproduced: librosa pads the signal by
        n_fft/2 on both sides with ZEROS before framing. This is worth
        spelling out because librosa's own default changed: up to 0.9 it
        padded in 'reflect' mode, from 0.10 onward in 'constant' mode. The
        training script did not pass pad_mode, so it got zeros. Reflection
        instead left the interior frames perfect and the first and last
        frames out by up to 12 dB - exactly the kind of error that would
        never show up as a crash.
     3. The Hann window is PERIODIC (librosa's default, sym=False), not
        symmetric. The difference is one sample, and it is enough to shift
        every value.

   Verified against librosa 1.0.0 on a synthetic mosquito tone: mean absolute
   difference 1.3e-6 dB, worst 7.6e-6 dB, which is float32 rounding and
   nothing else. Re-run check_features.js after any change to this file or to
   the training feature code.
   ========================================================================== */

(function (root) {
  "use strict";

  var SR = 8000, NFFT = 512, HOP = 128, NMELS = 40, NBINS = NFFT / 2 + 1;

  /* ------------------------------------------------------------ real FFT */
  function fftRadix2(re, im) {
    var n = re.length;
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
          re[s + k] = ur + vr; im[s + k] = ui + vi;
          re[s + k + len / 2] = ur - vr; im[s + k + len / 2] = ui - vi;
          var ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  /* Periodic Hann, as librosa uses (scipy.signal.get_window default sym=False) */
  var HANN = (function () {
    var w = new Float32Array(NFFT);
    for (var i = 0; i < NFFT; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / NFFT));
    return w;
  })();

  /* librosa pads by n_fft/2 each side when center=true, with zeros
     (pad_mode='constant', the default since librosa 0.10). A Float32Array is
     already zero-filled, so the padding is just the offset copy. */
  function centerPad(x, pad) {
    var out = new Float32Array(x.length + 2 * pad);
    out.set(x, pad);
    return out;
  }

  /* ------------------------------------------------- log-mel spectrogram
     y: Float32Array at 8 kHz. melfb: Float32Array of 40*257, row-major.
     Returns Float32Array of 40*frames in the model's (mel, time) order.     */
  function logMel(y, melfb) {
    var padded = centerPad(y, NFFT / 2);
    var frames = 1 + Math.floor((padded.length - NFFT) / HOP);

    var power = new Float32Array(frames * NBINS);
    var re = new Float64Array(NFFT), im = new Float64Array(NFFT);

    for (var f = 0; f < frames; f++) {
      var off = f * HOP;
      for (var i = 0; i < NFFT; i++) { re[i] = padded[off + i] * HANN[i]; im[i] = 0; }
      fftRadix2(re, im);
      var pb = f * NBINS;
      for (var k = 0; k < NBINS; k++) power[pb + k] = re[k] * re[k] + im[k] * im[k];
    }

    // mel projection: (40 x 257) x (257 x frames)
    var mel = new Float32Array(NMELS * frames);
    for (var m = 0; m < NMELS; m++) {
      var fbRow = m * NBINS;
      for (var t = 0; t < frames; t++) {
        var s = 0, pRow = t * NBINS;
        for (var b = 0; b < NBINS; b++) s += melfb[fbRow + b] * power[pRow + b];
        mel[m * frames + t] = s;
      }
    }

    // librosa.power_to_db(S, ref=np.max): 10*log10(S/max), floored 80 dB down
    var amin = 1e-10, topDb = 80, max = 0;
    for (var i2 = 0; i2 < mel.length; i2++) if (mel[i2] > max) max = mel[i2];
    var refDb = 10 * Math.log10(Math.max(amin, max));
    var peak = -Infinity;
    for (var i3 = 0; i3 < mel.length; i3++) {
      mel[i3] = 10 * Math.log10(Math.max(amin, mel[i3])) - refDb;
      if (mel[i3] > peak) peak = mel[i3];
    }
    var floorDb = peak - topDb;
    for (var i4 = 0; i4 < mel.length; i4++) if (mel[i4] < floorDb) mel[i4] = floorDb;

    return { data: mel, nMels: NMELS, frames: frames };
  }

  /* ------------------------------------------------------------ resample
     The phone records at 44.1 or 48 kHz; the model expects 8 kHz, and the
     training script got there through librosa.load(sr=8000), which uses a
     high quality band-limited resampler.

     A box average and nearest-sample pick is the obvious shortcut and it is
     wrong in a way that matters: it lets everything above 4 kHz fold back
     down into the band the model reads. Measured against librosa on a
     harmonic-rich 450 Hz tone, the box version shifted the log-mel
     spectrogram by 1.17 dB on average and 8.90 dB at worst. So this is a
     windowed-sinc resampler instead: Blackman-windowed, 32 taps per side,
     cutoff at the output Nyquist. That lands at 0.0003 dB mean and 0.046 dB
     worst, which is nothing. check_resample.js measures both.             */
  var SINC_HALF = 32;

  function resampleTo8k(y, srIn) {
    if (srIn === SR) return y;
    var ratio = srIn / SR;                 // input samples per output sample
    var fc = 0.5 / ratio;                  // cutoff, cycles per input sample
    var half = Math.ceil(SINC_HALF * Math.max(1, ratio));
    var outN = Math.floor(y.length / ratio);
    var out = new Float32Array(outN);

    for (var i = 0; i < outN; i++) {
      var c = i * ratio;
      var k0 = Math.ceil(c - half), k1 = Math.floor(c + half);
      var s = 0, wsum = 0;
      for (var k = k0; k <= k1; k++) {
        var t = k - c;
        // band-limited kernel: 2*fc*sinc(2*fc*t), tapered by a Blackman window
        var x = 2 * Math.PI * fc * t;
        var sinc = (x === 0) ? 1 : Math.sin(x) / x;
        var u = (t / half + 1) / 2;                      // 0..1 across the window
        var w = 0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u);
        var h = 2 * fc * sinc * w;
        wsum += h;
        if (k >= 0 && k < y.length) s += y[k] * h;
      }
      // normalise by the taps actually summed, so the edges do not fade out
      out[i] = wsum !== 0 ? s / wsum : 0;
    }
    return out;
  }

  /* -------------------------------------------------------------- window
     Model input is exactly 40 x 63. Take the central 63 frames if longer,
     pad with the floor value if shorter, then apply the training
     normalisation (mean and sd are in the model's meta JSON).              */
  function toInput(logmel, frames, mean, sd) {
    var want = 63, x = new Float32Array(NMELS * want);
    var start = Math.max(0, Math.floor((frames - want) / 2));
    var floorV = -80;
    for (var m = 0; m < NMELS; m++) {
      for (var t = 0; t < want; t++) {
        var src = start + t;
        var v = src < frames ? logmel[m * frames + src] : floorV;
        x[m * want + t] = (v - mean) / sd;
      }
    }
    return x;
  }

  root.Features = {
    SR: SR, NFFT: NFFT, HOP: HOP, NMELS: NMELS, NBINS: NBINS,
    logMel: logMel, resampleTo8k: resampleTo8k, toInput: toInput,
  };

})(typeof window !== "undefined" ? window : globalThis);
