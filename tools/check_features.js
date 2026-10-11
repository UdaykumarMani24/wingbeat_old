/* check_features.js — does features.js reproduce librosa's log-mel?
   Compares against ref_logmel.bin, computed by librosa itself on test_signal.bin. */
const fs = require('fs');
const g = {};
new Function('window', fs.readFileSync(__dirname + '/../features.js', 'utf8'))(g);

function f32(path) {
  const b = fs.readFileSync(path);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const meta = JSON.parse(fs.readFileSync(__dirname + '/feat_ref.json', 'utf8'));
const y = f32(__dirname + '/test_signal.bin');
const melfb = f32(__dirname + '/../models/melfb.bin');
const ref = f32(__dirname + '/ref_logmel.bin');

console.log('signal', y.length, 'expected', meta.signal_len);
console.log('melfb ', melfb.length, 'expected', meta.fb_shape[0] * meta.fb_shape[1]);
console.log('ref   ', ref.length, 'expected', meta.logmel_shape[0] * meta.logmel_shape[1]);

const out = g.Features.logMel(y, melfb);
console.log('\njs frames', out.frames, ' librosa frames', meta.logmel_shape[1]);
if (out.frames !== meta.logmel_shape[1]) {
  console.log('FAIL  frame count differs — everything downstream is misaligned');
  process.exit(1);
}

let worst = 0, sum = 0, wm = -1, wt = -1;
for (let m = 0; m < 40; m++) {
  for (let t = 0; t < out.frames; t++) {
    const e = Math.abs(out.data[m * out.frames + t] - ref[m * out.frames + t]);
    sum += e;
    if (e > worst) { worst = e; wm = m; wt = t; }
  }
}
const n = 40 * out.frames;
console.log('mean abs diff :', (sum / n).toExponential(3), 'dB');
console.log('worst abs diff:', worst.toExponential(3), 'dB  at mel', wm, 'frame', wt);

// show a few values side by side so a failure is readable rather than a number
console.log('\n      js        librosa');
[[0, 0], [10, 20], [20, 31], [39, 62]].forEach(([m, t]) => {
  console.log(`m${String(m).padStart(2)} t${String(t).padStart(2)}  ` +
    out.data[m * out.frames + t].toFixed(4).padStart(10) + '  ' +
    ref[m * out.frames + t].toFixed(4).padStart(10));
});

console.log('\n' + (worst < 1e-2 ? 'PASS  features.js matches librosa' : 'FAIL  features differ'));
