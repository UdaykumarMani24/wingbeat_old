/* check_resample.js — 44.1 kHz phone audio down to the 8 kHz the model wants.
   Compares the sinc resampler against librosa's soxr_hq, and against the box
   average it replaced, in the only units that matter: dB of log-mel error,
   because that is what the model actually sees. */
const fs = require('fs');
const g = {};
new Function('window', fs.readFileSync(__dirname + '/../features.js', 'utf8'))(g);

function f32(p) {
  const b = fs.readFileSync(p.indexOf('/') >= 0 ? __dirname + '/' + p : __dirname + '/' + p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const x = f32('rs_in44k.bin');
const ref8 = f32('rs_ref8k.bin');
const refL = f32('rs_ref_logmel.bin');
const melfb = f32('../models/melfb.bin');
const meta = JSON.parse(fs.readFileSync(__dirname + '/rs_ref.json', 'utf8'));

// the old box-average version, kept here only so the comparison is real
function boxResample(y, srIn) {
  const ratio = srIn / 8000, half = Math.max(1, Math.floor(ratio / 2));
  const outN = Math.floor(y.length / ratio), out = new Float32Array(outN);
  for (let i = 0; i < outN; i++) {
    const c0 = Math.round(i * ratio);
    let s = 0, n = 0;
    for (let k = -half; k <= half; k++) {
      const idx = c0 + k;
      if (idx >= 0 && idx < y.length) { s += y[idx]; n++; }
    }
    out[i] = n ? s / n : 0;
  }
  return out;
}

function logmelErr(y8) {
  const out = g.Features.logMel(y8.subarray(0, 8000), melfb);
  let worst = 0, sum = 0;
  const n = 40 * out.frames;
  for (let i = 0; i < n; i++) {
    const e = Math.abs(out.data[i] - refL[i]);
    sum += e;
    if (e > worst) worst = e;
  }
  return { mean: sum / n, worst: worst, frames: out.frames };
}

const sinc = g.Features.resampleTo8k(x, meta.sr_in);
const box = boxResample(x, meta.sr_in);

console.log('librosa output length', ref8.length, ' js', sinc.length);

// waveform agreement, away from the edges where the kernels differ
let wsum = 0, wworst = 0, cnt = 0;
for (let i = 200; i < Math.min(sinc.length, ref8.length) - 200; i++) {
  const e = Math.abs(sinc[i] - ref8[i]); wsum += e; cnt++;
  if (e > wworst) wworst = e;
}
console.log('waveform  mean abs err', (wsum / cnt).toExponential(2),
            ' worst', wworst.toExponential(2),
            ' (signal rms about 0.5)');

const a = logmelErr(sinc), b = logmelErr(box);
console.log('\nlog-mel error against librosa, in dB');
console.log('  windowed sinc :  mean ' + a.mean.toFixed(4) + '   worst ' + a.worst.toFixed(4));
console.log('  box average   :  mean ' + b.mean.toFixed(4) + '   worst ' + b.worst.toFixed(4));

console.log('\n' + (a.mean < 0.5 && a.worst < 4 && a.mean < b.mean
  ? 'PASS  the sinc resampler is close enough to librosa, and better than the box'
  : 'FAIL  resampler not good enough'));
