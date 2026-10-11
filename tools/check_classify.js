/* check_classify.js — the aggregation layer, with fetch stubbed out so the
   whole browser path can run in node. Checks that a mosquito-like signal is
   detected, that silence is not, and that nothing throws. */
const fs = require('fs');
const path = require('path');
const APP = __dirname + '/../';

const g = {};
g.fetch = async function (url) {
  const p = path.join(APP, url);
  const b = fs.readFileSync(p);
  const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  return {
    ok: true, status: 200,
    arrayBuffer: async () => ab,
    json: async () => JSON.parse(b.toString('utf8')),
  };
};
g.setTimeout = setTimeout;
g.Promise = Promise;

for (const f of ['features.js', 'model.js', 'classify.js']) {
  new Function('window', 'fetch', 'setTimeout', fs.readFileSync(APP + f, 'utf8'))
    (g, g.fetch, setTimeout);
}

function synth(sr, secs, f0, amp, noise) {
  const n = Math.round(sr * secs), x = new Float32Array(n);
  let ph = 0;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const f = f0 + 20 * Math.sin(2 * Math.PI * 3 * t);
    ph += 2 * Math.PI * f / sr;
    let v = 0;
    for (let h = 1; h <= 8; h++) v += Math.pow(0.6, h - 1) * Math.sin(h * ph);
    x[i] = amp * v + noise * (Math.random() * 2 - 1);
  }
  return x;
}

(async function () {
  const SR = 44100;

  // 1. a clear 450 Hz mosquito-like tone for 8 seconds
  const tone = synth(SR, 8, 450, 0.3, 0.02);
  let prog = 0;
  const r1 = await g.Classify.run(tone, SR, () => prog++);
  console.log('tone 450 Hz, 8 s');
  console.log('  windows scored   :', r1.windows.length, 'of', r1.total,
              '(quiet skipped', r1.quiet + ')');
  console.log('  progress calls   :', prog);
  console.log('  hit fraction     :', r1.hitFraction.toFixed(3));
  console.log('  mosquito-seconds :', r1.mosquitoSeconds);
  console.log('  peak / median p  :', r1.peakMosquito.toFixed(4), '/',
              r1.medianMosquito.toFixed(4));
  console.log('  culex on hits    :', r1.culexOnHits === null ? 'n/a'
                                      : r1.culexOnHits.toFixed(4));

  // 2. room noise only — the detector should not light up on this
  const noise = new Float32Array(SR * 4);
  for (let i = 0; i < noise.length; i++) noise[i] = 0.01 * (Math.random() * 2 - 1);
  const r2 = await g.Classify.run(noise, SR);
  console.log('\nwhite noise, 4 s');
  console.log('  hit fraction     :', r2.hitFraction.toFixed(3));
  console.log('  median p         :', r2.medianMosquito.toFixed(4));

  // 3. digital silence — every window must be skipped, not scored
  const silence = new Float32Array(SR * 3);
  const r3 = await g.Classify.run(silence, SR);
  console.log('\nsilence, 3 s');
  console.log('  empty flag       :', r3.empty === true);
  console.log('  quiet skipped    :', r3.quiet, 'of', r3.total);

  // 4. too short to score — must fail with a message, not a wrong answer
  let msg = '';
  try { await g.Classify.run(new Float32Array(1000), SR); }
  catch (e) { msg = e.message; }
  console.log('\n0.02 s input    :', msg || 'NO ERROR RAISED');

  // 5. 48 kHz, since many phones record there and the resampler must cope
  const r5 = await g.Classify.run(synth(48000, 4, 520, 0.3, 0.02), 48000);
  console.log('\n520 Hz at 48 kHz:  hit fraction', r5.hitFraction.toFixed(3),
              ' median p', r5.medianMosquito.toFixed(4));

  const pass = r1.hitFraction > 0.8 && r1.windows.length === 15 &&
               r3.empty === true && r3.quiet === r3.total &&
               msg.length > 0 && r5.hitFraction > 0.5;
  console.log('\n' + (pass ? 'PASS  classify.js behaves'
                           : 'FAIL  see the numbers above'));
})();
