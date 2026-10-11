const fs=require('fs'); const g={};
new Function('window', fs.readFileSync(__dirname + '/../model.js','utf8'))(g);
const ref=JSON.parse(fs.readFileSync(__dirname+'/ref.json','utf8'));

function loadLocal(prefix){
  const meta=JSON.parse(fs.readFileSync(__dirname+'/../models/'+prefix+'.json','utf8'));
  const buf=fs.readFileSync(__dirname+'/../models/'+prefix+'.bin');
  const ab=buf.buffer.slice(buf.byteOffset, buf.byteOffset+buf.byteLength);
  return g.MosquitoModel.make(meta, ab);
}
const det=loadLocal('detector'), cul=loadLocal('culex');

const N=40*63;
let worst=0, rows=[];
for(let i=0;i<4;i++){
  const x=Float32Array.from(ref.x.slice(i*N,(i+1)*N));
  const a=det.predict(x), b=cul.predict(x);
  const ea=Math.abs(a-ref.det[i]), eb=Math.abs(b-ref.cul[i]);
  worst=Math.max(worst,ea,eb);
  rows.push([i, ref.det[i], a, ea, ref.cul[i], b, eb]);
}
console.log('     keras-det     js-det        err      keras-cul     js-cul        err');
rows.forEach(r=>console.log(
  String(r[0]).padStart(2)+'  '+r[1].toFixed(8)+'  '+r[2].toFixed(8)+'  '+r[3].toExponential(1)
  +'   '+r[4].toFixed(8)+'  '+r[5].toFixed(8)+'  '+r[6].toExponential(1)));
console.log('\nworst absolute error:', worst.toExponential(2));
console.log(worst < 1e-5 ? 'PASS  JavaScript matches Keras' : 'FAIL  implementations disagree');
