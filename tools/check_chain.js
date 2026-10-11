/* check_chain.js — the whole pipeline, audio in and probability out, against
   the same thing computed in Python. This is the test that matters: features
   and model can each be right while the glue between them is wrong. */
const fs = require('fs');
const g = {};
new Function('window', fs.readFileSync(__dirname + '/../features.js', 'utf8'))(g);
new Function('window', fs.readFileSync(__dirname + '/../model.js', 'utf8'))(g);

function f32(p) {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
function model(prefix) {
  const meta = JSON.parse(fs.readFileSync(__dirname + '/../models/' + prefix + '.json', 'utf8'));
  const b = fs.readFileSync(__dirname + '/../models/' + prefix + '.bin');
  return g.MosquitoModel.make(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const y = f32(__dirname + '/test_signal.bin');
const melfb = f32(__dirname + '/../models/melfb.bin');
const ref = JSON.parse(fs.readFileSync(__dirname + '/chain_ref.json', 'utf8'));

const lm = g.Features.logMel(y, melfb);

let worst = 0;
console.log('            python        js          err');
[['detector', 'detector'], ['culex', 'culex']].forEach(([key, prefix]) => {
  const m = model(prefix);
  const x = g.Features.toInput(lm.data, lm.frames, m.meta.norm.mean, m.meta.norm.sd);
  const p = m.predict(x);
  const e = Math.abs(p - ref[key].p);
  worst = Math.max(worst, e);
  console.log(key.padEnd(10) + '  ' + ref[key].p.toFixed(8) + '  ' + p.toFixed(8) +
              '  ' + e.toExponential(1));
  // the meta JSON and the .npy must agree, or the app normalises with the wrong numbers
  if (Math.abs(m.meta.norm.mean - ref[key].mean) > 1e-4 ||
      Math.abs(m.meta.norm.sd - ref[key].sd) > 1e-4) {
    console.log('  FAIL  norm values in ' + prefix + '.json differ from the trained ones');
    worst = Infinity;
  }
});

console.log('\nworst absolute error:', worst.toExponential(2));
console.log(worst < 1e-4 ? 'PASS  the JavaScript pipeline reproduces the Python one'
                         : 'FAIL  pipeline disagrees');
