/* ============================================================================
   model.js — runs the trained mosquito CNN in plain JavaScript.

   The network is small (23,553 parameters, 92 KB of float32), so there is no
   reason to pull in TensorFlow.js and a megabyte of CDN. Everything here is
   about sixty lines of arithmetic, which keeps Wingbeat Lab small and working
   with no network at all.

   Architecture, exactly as trained:
     input 40 x 63 x 1   (40 mel bands, 63 time frames, 1 second at 8 kHz)
     Conv2D 3x3 'same' -> 16, relu
     MaxPool 2x2                        -> 20 x 31 x 16
     BatchNormalization (eps 1e-3)
     Conv2D 3x3 'same' -> 32, relu
     MaxPool 2x2                        -> 10 x 15 x 32
     BatchNormalization (eps 1e-3)
     Conv2D 3x3 'same' -> 64, relu      -> 10 x 15 x 64
     GlobalAveragePooling2D             -> 64
     Dense 1, sigmoid

   Verified against Keras: the outputs below agree with the Python model to
   better than 1e-6 on the same inputs. If you retrain, re-run the check in
   tools/check_model.js before shipping.
   ========================================================================== */

(function (root) {
  "use strict";

  /* ------------------------------------------------------------- loading */
  function Tensors(meta, buffer) {
    var f32 = new Float32Array(buffer);
    var by = {};
    var seen = {};
    meta.tensors.forEach(function (t) {
      // names repeat (two conv layers are both "conv.kernel"), so index them
      var n = t.name + "#" + (seen[t.name] = (seen[t.name] || 0) + 1);
      by[n] = { data: f32.subarray(t.offset, t.offset + t.size), shape: t.shape };
    });
    return by;
  }

  async function load(jsonUrl, binUrl) {
    var meta = await (await fetch(jsonUrl)).json();
    var buf = await (await fetch(binUrl)).arrayBuffer();
    return make(meta, buf);
  }

  function make(meta, buffer) {
    var T = Tensors(meta, buffer);
    return {
      meta: meta,
      predict: function (x) { return forward(T, meta, x); },
    };
  }

  /* -------------------------------------------------------------- layers
     Tensors are flat Float32Arrays in channels-last order:
       index = ((row * width) + col) * channels + channel                   */

  function conv2dSameRelu(x, H, W, Cin, kernel, bias, Cout) {
    // kernel is (3, 3, Cin, Cout) flattened the same way Keras stores it
    var out = new Float32Array(H * W * Cout);
    for (var y = 0; y < H; y++) {
      for (var xx = 0; xx < W; xx++) {
        var base = (y * W + xx) * Cout;
        for (var oc = 0; oc < Cout; oc++) out[base + oc] = bias[oc];
        for (var ky = 0; ky < 3; ky++) {
          var sy = y + ky - 1;                 // 'same' padding, 3x3 -> pad 1
          if (sy < 0 || sy >= H) continue;
          for (var kx = 0; kx < 3; kx++) {
            var sx = xx + kx - 1;
            if (sx < 0 || sx >= W) continue;
            var inBase = (sy * W + sx) * Cin;
            var kBase = ((ky * 3 + kx) * Cin) * Cout;
            for (var ic = 0; ic < Cin; ic++) {
              var v = x[inBase + ic];
              if (v === 0) continue;
              var kRow = kBase + ic * Cout;
              for (var oc2 = 0; oc2 < Cout; oc2++) out[base + oc2] += v * kernel[kRow + oc2];
            }
          }
        }
        for (var oc3 = 0; oc3 < Cout; oc3++) {
          if (out[base + oc3] < 0) out[base + oc3] = 0;    // relu
        }
      }
    }
    return out;
  }

  function maxPool2(x, H, W, C) {
    var OH = Math.floor(H / 2), OW = Math.floor(W / 2);   // 'valid'
    var out = new Float32Array(OH * OW * C);
    for (var y = 0; y < OH; y++) {
      for (var xx = 0; xx < OW; xx++) {
        var ob = (y * OW + xx) * C;
        for (var c = 0; c < C; c++) {
          var a = x[((2 * y) * W + 2 * xx) * C + c];
          var b = x[((2 * y) * W + 2 * xx + 1) * C + c];
          var d = x[((2 * y + 1) * W + 2 * xx) * C + c];
          var e = x[((2 * y + 1) * W + 2 * xx + 1) * C + c];
          out[ob + c] = Math.max(Math.max(a, b), Math.max(d, e));
        }
      }
    }
    return { data: out, H: OH, W: OW };
  }

  function batchNorm(x, n, C, g, b, mean, varr, eps) {
    // inference-time BN: y = gamma * (x - moving_mean) / sqrt(moving_var + eps) + beta
    var scale = new Float32Array(C), shift = new Float32Array(C);
    for (var c = 0; c < C; c++) {
      scale[c] = g[c] / Math.sqrt(varr[c] + eps);
      shift[c] = b[c] - mean[c] * scale[c];
    }
    for (var i = 0; i < n; i++) {
      var ch = i % C;
      x[i] = x[i] * scale[ch] + shift[ch];
    }
    return x;
  }

  function globalAvgPool(x, H, W, C) {
    var out = new Float32Array(C), n = H * W;
    for (var i = 0; i < n; i++) {
      var b = i * C;
      for (var c = 0; c < C; c++) out[c] += x[b + c];
    }
    for (var c2 = 0; c2 < C; c2++) out[c2] /= n;
    return out;
  }

  function denseSigmoid(x, C, kernel, bias) {
    var s = bias[0];
    for (var c = 0; c < C; c++) s += x[c] * kernel[c];
    return 1 / (1 + Math.exp(-s));
  }

  /* ------------------------------------------------------------- forward
     x is a Float32Array of 40*63, ALREADY normalised with the mean and sd
     that the model was trained with (they are in the meta JSON).            */
  function forward(T, meta, x) {
    var H = meta.input[0], W = meta.input[1];
    var eps = meta.bn_epsilon;

    var a = conv2dSameRelu(x, H, W, 1, T["conv.kernel#1"].data, T["conv.bias#1"].data, 16);
    var p = maxPool2(a, H, W, 16);
    batchNorm(p.data, p.data.length, 16,
              T["bn.gamma#1"].data, T["bn.beta#1"].data,
              T["bn.mean#1"].data, T["bn.var#1"].data, eps);

    a = conv2dSameRelu(p.data, p.H, p.W, 16, T["conv.kernel#2"].data, T["conv.bias#2"].data, 32);
    var p2 = maxPool2(a, p.H, p.W, 32);
    batchNorm(p2.data, p2.data.length, 32,
              T["bn.gamma#2"].data, T["bn.beta#2"].data,
              T["bn.mean#2"].data, T["bn.var#2"].data, eps);

    a = conv2dSameRelu(p2.data, p2.H, p2.W, 32,
                       T["conv.kernel#3"].data, T["conv.bias#3"].data, 64);
    var g = globalAvgPool(a, p2.H, p2.W, 64);
    return denseSigmoid(g, 64, T["dense.kernel#1"].data, T["dense.bias#1"].data);
  }

  root.MosquitoModel = { load: load, make: make };

})(typeof window !== "undefined" ? window : globalThis);
