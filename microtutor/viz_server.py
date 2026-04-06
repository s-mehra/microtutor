"""3D knowledge graph visualization server using Three.js."""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from microtutor.state_emitter import GraphState

_current_state: dict | None = None
_state_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(GRAPH_HTML.encode())
        elif self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with _state_lock:
                data = _current_state or {}
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence request logs


class ProgressServer:
    """Local HTTP server serving the 3D knowledge graph."""

    def __init__(self) -> None:
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int = 0

    def update_state(self, state: GraphState) -> None:
        global _current_state
        with _state_lock:
            _current_state = _state_to_dict(state)

    def start(self) -> int:
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def port(self) -> int:
        return self._port

    def open_browser(self) -> None:
        webbrowser.open(f"http://127.0.0.1:{self._port}")


def _state_to_dict(s: GraphState) -> dict:
    return {
        "course_title": s.course_title,
        "current_concept": s.current_concept,
        "concepts": [
            {
                "id": c.id,
                "name": c.name,
                "mastery": c.mastery,
                "attempts": c.attempts,
                "status": c.status,
                "depth": c.depth,
                "last_updated_at": c.last_updated_at,
            }
            for c in s.concepts
        ],
        "edges": s.edges,
        "frontier": s.frontier,
        "total_mastered": s.total_mastered,
        "total_concepts": s.total_concepts,
    }


GRAPH_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>microtutor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
#title{position:fixed;top:0;left:0;right:0;text-align:center;padding:20px;z-index:10;pointer-events:none}
#title h1{color:rgba(255,255,255,0.85);font-size:17px;font-weight:500;letter-spacing:1px}
#title .sub{color:rgba(255,255,255,0.3);font-size:11px;margin-top:4px}
#status{position:fixed;bottom:0;left:0;right:0;background:rgba(0,0,0,0.6);
  padding:10px 24px;display:flex;align-items:center;gap:16px;z-index:10;font-size:11px;color:#555}
#status .current{color:#00c8ff}
#status .bar-wrap{width:100px;height:2px;background:#111;border-radius:1px}
#status .bar-fill{height:100%;background:#00b482;border-radius:1px;transition:width 0.5s}
#status .count{margin-left:auto}
#tooltip{position:fixed;pointer-events:none;z-index:100;background:rgba(0,0,0,0.8);
  border:1px solid rgba(255,255,255,0.08);border-radius:5px;padding:5px 10px;
  color:#fff;font-size:11px;display:none;white-space:nowrap;backdrop-filter:blur(4px)}
#tooltip .pct{color:rgba(255,255,255,0.5);font-size:10px}
</style>
</head>
<body>
<div id="title"><h1 id="courseTitle"></h1><div class="sub" id="progressText"></div></div>
<div id="status">
  <span class="current" id="currentText"></span>
  <div class="bar-wrap"><div class="bar-fill" id="progressBar"></div></div>
  <span class="count" id="countText"></span>
</div>
<div id="tooltip"></div>

<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}
</script>
<script type="module">
import*as THREE from'three';
import{OrbitControls}from'three/addons/controls/OrbitControls.js';
import{CSS2DRenderer,CSS2DObject}from'three/addons/renderers/CSS2DRenderer.js';

// ── Texture generators ──
function makeGlow(sz,coreR,cr,cg,cb){
  const c=document.createElement('canvas');c.width=c.height=sz;
  const x=c.getContext('2d'),h=sz/2;
  const g=x.createRadialGradient(h,h,0,h,h,h);
  g.addColorStop(0,`rgba(${cr},${cg},${cb},1)`);
  g.addColorStop(coreR,`rgba(${cr},${cg},${cb},0.8)`);
  g.addColorStop(0.2,`rgba(${cr},${cg},${cb},0.2)`);
  g.addColorStop(0.5,`rgba(${cr},${cg},${cb},0.04)`);
  g.addColorStop(1,'rgba(0,0,0,0)');
  x.fillStyle=g;x.fillRect(0,0,sz,sz);
  const t=new THREE.CanvasTexture(c);t.needsUpdate=true;return t;
}
function makeRing(sz){
  const c=document.createElement('canvas');c.width=c.height=sz;
  const x=c.getContext('2d'),h=sz/2;
  const g=x.createRadialGradient(h,h,h*0.3,h,h,h*0.55);
  g.addColorStop(0,'rgba(0,200,255,0)');
  g.addColorStop(0.3,'rgba(0,200,255,0.4)');
  g.addColorStop(0.5,'rgba(0,200,255,0.7)');
  g.addColorStop(0.7,'rgba(0,200,255,0.4)');
  g.addColorStop(1,'rgba(0,200,255,0)');
  x.fillStyle=g;x.fillRect(0,0,sz,sz);
  const t=new THREE.CanvasTexture(c);t.needsUpdate=true;return t;
}
function makeNebula(sz,r,g,b){
  const c=document.createElement('canvas');c.width=c.height=sz;
  const x=c.getContext('2d'),h=sz/2;
  const gr=x.createRadialGradient(h,h,0,h,h,h);
  gr.addColorStop(0,`rgba(${r},${g},${b},0.1)`);
  gr.addColorStop(0.4,`rgba(${r},${g},${b},0.03)`);
  gr.addColorStop(1,'rgba(0,0,0,0)');
  x.fillStyle=gr;x.fillRect(0,0,sz,sz);
  return new THREE.CanvasTexture(c);
}

const starTex=makeGlow(64,0.06,255,255,255);
const ringTex=makeRing(128);

// ── Scene ──
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.1,500);
camera.position.set(0,4,22);

const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.setClearColor(0x000000);
document.body.appendChild(renderer.domElement);

const labelRenderer=new CSS2DRenderer();
labelRenderer.setSize(innerWidth,innerHeight);
labelRenderer.domElement.style.position='absolute';
labelRenderer.domElement.style.top='0';
labelRenderer.domElement.style.pointerEvents='none';
document.body.appendChild(labelRenderer.domElement);

const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=0.06;
controls.rotateSpeed=0.8;controls.zoomSpeed=1.2;
controls.minDistance=3;controls.maxDistance=65;
controls.enablePan=true;controls.panSpeed=0.5;

scene.add(new THREE.AmbientLight(0xffffff,0.15));
const pl=new THREE.PointLight(0xffffff,0.6,80);pl.position.set(5,12,10);scene.add(pl);

// ── Background stars (shader for twinkle + varying size) ──
const BSC=2500;const bPos=new Float32Array(BSC*3);
const bSz=new Float32Array(BSC);const bPh=new Float32Array(BSC);const bOp=new Float32Array(BSC);
for(let i=0;i<BSC;i++){
  bPos[i*3]=(Math.random()-.5)*400;bPos[i*3+1]=(Math.random()-.5)*400;bPos[i*3+2]=(Math.random()-.5)*400;
  bSz[i]=0.04+Math.pow(Math.random(),3)*0.5;
  bOp[i]=0.15+Math.random()*0.6;bPh[i]=Math.random()*6.28;
}
const bGeo=new THREE.BufferGeometry();
bGeo.setAttribute('position',new THREE.Float32BufferAttribute(bPos,3));
bGeo.setAttribute('aSize',new THREE.Float32BufferAttribute(bSz,1));
bGeo.setAttribute('aOp',new THREE.Float32BufferAttribute(bOp,1));
bGeo.setAttribute('aPh',new THREE.Float32BufferAttribute(bPh,1));
const bMat=new THREE.ShaderMaterial({
  uniforms:{uTime:{value:0}},
  vertexShader:`attribute float aSize;attribute float aOp;attribute float aPh;
    varying float vOp;uniform float uTime;
    void main(){vOp=aOp*(0.6+0.4*sin(uTime*1.5+aPh));
    vec4 mv=modelViewMatrix*vec4(position,1.0);
    gl_PointSize=aSize*(200.0/-mv.z);gl_Position=projectionMatrix*mv;}`,
  fragmentShader:`varying float vOp;void main(){
    float d=length(gl_PointCoord-0.5)*2.0;
    float a=smoothstep(1.0,0.0,d)*vOp;
    gl_FragColor=vec4(0.85,0.88,1.0,a);}`,
  transparent:true,blending:THREE.AdditiveBlending,depthWrite:false
});
scene.add(new THREE.Points(bGeo,bMat));

// ── Nebulae ──
[[-15,5,-30,60,35,15,70],[20,-8,-50,80,15,25,65],[-5,12,-70,50,45,12,35]].forEach(n=>{
  const t=makeNebula(256,n[3],n[4],n[5]);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t,transparent:true,opacity:0.5,blending:THREE.AdditiveBlending,depthWrite:false}));
  s.position.set(n[0],n[1],n[2]);s.scale.setScalar(n[6]||60);s.renderOrder=-10;scene.add(s);
});

// ── State ──
let nodes={};let edgeLines=[];let stablePos={};
const tooltip=document.getElementById('tooltip');
const raycaster=new THREE.Raycaster();
raycaster.params.Sprite={threshold:0.8};
const mouse=new THREE.Vector2();
let hovered=null;

function mColor(m){
  let r,g,b;
  if(m<0.5){const t=m/0.5;r=255;g=60+140*t|0;b=50*(1-t)|0}
  else{const t=(m-.5)/.5;r=255-210*t|0;g=200+30*t|0;b=50*t+30|0}
  return new THREE.Color(r/255,g/255,b/255);
}
function mSize(m){return 0.15+m*0.6}

function buildScene(state){
  Object.values(nodes).forEach(n=>{
    scene.remove(n.mesh);if(n.glow)scene.remove(n.glow);
    if(n.ring)scene.remove(n.ring);if(n.label)scene.remove(n.label);
  });
  edgeLines.forEach(e=>scene.remove(e));
  nodes={};edgeLines=[];
  if(!state||!state.concepts)return;

  const maxD=Math.max(...state.concepts.map(c=>c.depth),1);
  const layers={};
  state.concepts.forEach(c=>{if(!layers[c.depth])layers[c.depth]=[];layers[c.depth].push(c)});

  state.concepts.forEach(c=>{
    if(stablePos[c.id])return;
    const ly=layers[c.depth],idx=ly.indexOf(c),n=ly.length;
    const z=-(c.depth/maxD)*24;
    const x=n===1?0:(idx/(n-1)-.5)*Math.min(n*4,18);
    const y=(Math.random()-.5)*2.5;
    stablePos[c.id]=new THREE.Vector3(x,y,z);
  });

  // Edges — visible constellation lines
  state.edges.forEach(([src,tgt])=>{
    const p1=stablePos[src],p2=stablePos[tgt];if(!p1||!p2)return;
    const act=src===state.current_concept||tgt===state.current_concept;
    const geo=new THREE.BufferGeometry().setFromPoints([p1,p2]);
    const mat=new THREE.LineBasicMaterial({
      color:act?0x5588bb:0x334466,transparent:true,opacity:act?0.7:0.4
    });
    const line=new THREE.Line(geo,mat);
    scene.add(line);edgeLines.push(line);
  });

  // Nodes
  state.concepts.forEach(c=>{
    const pos=stablePos[c.id];if(!pos)return;
    const locked=c.status==='locked';
    const isCur=c.id===state.current_concept;
    const m=c.mastery;
    let mesh,glow=null,ring=null,label=null;

    if(locked){
      // Unexplored: twinkling star sprite — no label
      const mat=new THREE.SpriteMaterial({
        map:starTex,color:0xffffff,transparent:true,
        opacity:0.3+Math.random()*0.4,
        blending:THREE.AdditiveBlending,depthWrite:false
      });
      mesh=new THREE.Sprite(mat);
      const bs=0.3+Math.random()*0.3;
      mesh.scale.setScalar(bs);
      mesh.userData={type:'star',baseOp:mat.opacity,baseScale:bs,
        twinkSpd:1+Math.random()*3,twinkPh:Math.random()*6.28};
    } else {
      // Explored: sphere + bright glow aura
      const sz=mSize(m);const col=mColor(m);
      const geo=new THREE.SphereGeometry(sz,20,20);
      const mat=new THREE.MeshPhongMaterial({
        color:col,emissive:col,emissiveIntensity:0.6+m*0.4,
        transparent:true,opacity:0.95,shininess:80
      });
      mesh=new THREE.Mesh(geo,mat);
      mesh.userData={type:'body',size:sz};

      // Glow aura — bigger and brighter
      const gc=col.clone();
      const gt=makeGlow(128,0.08,gc.r*255|0,gc.g*255|0,gc.b*255|0);
      const gm=new THREE.SpriteMaterial({
        map:gt,transparent:true,opacity:0.5+m*0.4,
        blending:THREE.AdditiveBlending,depthWrite:false
      });
      glow=new THREE.Sprite(gm);
      glow.scale.setScalar(sz*7);
      glow.position.copy(pos);glow.renderOrder=-1;
      scene.add(glow);

      // Always-visible label for explored nodes
      const div=document.createElement('div');
      div.style.cssText='font-size:10px;font-weight:600;color:#fff;text-align:center;'+
        'text-shadow:0 0 6px #000,0 0 12px #000;pointer-events:none;white-space:nowrap';
      if(isCur)div.style.color='#00c8ff';
      div.innerHTML=c.name+'<br><span style="font-size:9px;opacity:0.6">'+Math.round(m*100)+'%</span>';
      label=new CSS2DObject(div);
      label.position.set(pos.x,pos.y-sz-0.6,pos.z);
      scene.add(label);

      // Current concept ring
      if(isCur){
        const rm=new THREE.SpriteMaterial({
          map:ringTex,color:0x00c8ff,transparent:true,
          blending:THREE.AdditiveBlending,depthWrite:false
        });
        ring=new THREE.Sprite(rm);
        ring.scale.setScalar(sz*7);
        ring.position.copy(pos);scene.add(ring);
      }
    }
    mesh.position.copy(pos);scene.add(mesh);
    nodes[c.id]={mesh,glow,ring,label,data:c,baseY:pos.y};
  });
}

function updateNodes(state){
  if(!state)return;
  state.concepts.forEach(c=>{
    const n=nodes[c.id];if(!n)return;
    if(c.status!=='locked'&&n.mesh.userData.type==='body'){
      const col=mColor(c.mastery);const sz=mSize(c.mastery);
      n.mesh.material.color=col;n.mesh.material.emissive=col;
      n.mesh.material.emissiveIntensity=0.6+c.mastery*0.4;
      const s=sz/0.5;n.mesh.scale.setScalar(s>0?s:0.3);
      if(n.glow){
        n.glow.material.opacity=0.5+c.mastery*0.4;
        n.glow.scale.setScalar(sz*7);
      }
      if(n.label){
        const isCur=c.id===state.current_concept;
        n.label.element.style.color=isCur?'#00c8ff':'#fff';
        n.label.element.innerHTML=c.name+'<br><span style="font-size:9px;opacity:0.6">'+Math.round(c.mastery*100)+'%</span>';
      }
    }
    n.data=c;
  });
}

function updateUI(state){
  if(!state)return;
  document.getElementById('courseTitle').textContent=state.course_title||'';
  const pct=state.total_concepts?state.total_mastered/state.total_concepts:0;
  document.getElementById('progressText').textContent=Math.round(pct*100)+'% explored';
  document.getElementById('progressBar').style.width=(pct*100)+'%';
  document.getElementById('countText').textContent=state.total_mastered+'/'+state.total_concepts;
  let cur='';
  if(state.current_concept){
    const cc=state.concepts.find(c=>c.id===state.current_concept);
    if(cc)cur=cc.name;
  }
  document.getElementById('currentText').textContent=cur?'Learning: '+cur:'';
}

// ── Hover ──
renderer.domElement.addEventListener('mousemove',e=>{
  mouse.x=(e.clientX/innerWidth)*2-1;
  mouse.y=-(e.clientY/innerHeight)*2+1;
  raycaster.setFromCamera(mouse,camera);
  const meshes=Object.values(nodes).map(n=>n.mesh);
  const hits=raycaster.intersectObjects(meshes);

  if(hits.length>0){
    const hit=hits[0].object;
    const nd=Object.values(nodes).find(n=>n.mesh===hit);
    if(nd){
      if(hovered&&hovered!==nd)resetHover(hovered);
      hovered=nd;
      const c=nd.data;const locked=c.status==='locked';
      tooltip.innerHTML=locked
        ?`<span style="color:#8af">${c.name}</span>`
        :`<span>${c.name}</span><br><span class="pct">${Math.round(c.mastery*100)}% mastery</span>`;
      tooltip.style.display='block';
      tooltip.style.left=(e.clientX+14)+'px';
      tooltip.style.top=(e.clientY-14)+'px';
      // Highlight
      if(hit.userData.type==='star'){
        hit.material.opacity=1;hit.scale.setScalar(hit.userData.baseScale*2.2);
      }else if(nd.glow){
        nd.glow.material.opacity=Math.min(1,nd.glow.material.opacity*1.8);
      }
    }
  }else{
    if(hovered){resetHover(hovered);hovered=null}
    tooltip.style.display='none';
  }
});

function resetHover(nd){
  const m=nd.mesh;
  if(m.userData.type==='star'){
    m.material.opacity=m.userData.baseOp;m.scale.setScalar(m.userData.baseScale);
  }else if(nd.glow){
    nd.glow.material.opacity=0.5+nd.data.mastery*0.4;
  }
}

// ── Polling ──
let hash='';
async function fetchState(){
  try{
    const r=await fetch('/state');const s=await r.json();
    if(!s||!s.concepts)return;
    const h=s.concepts.map(c=>c.id+c.status).join(',');
    if(h!==hash){buildScene(s);hash=h}else{updateNodes(s)}
    updateUI(s);
  }catch(e){}
}

// ── Animation ──
let t=0;
function animate(){
  requestAnimationFrame(animate);t+=0.012;
  bMat.uniforms.uTime.value=t;

  Object.values(nodes).forEach(n=>{
    // Star twinkle
    if(n.mesh.userData.type==='star'&&n!==hovered){
      const u=n.mesh.userData;
      const f=Math.sin(t*u.twinkSpd+u.twinkPh)*0.15+Math.sin(t*u.twinkSpd*2.3+u.twinkPh)*0.08;
      n.mesh.material.opacity=Math.max(0.1,u.baseOp+f);
    }
    // Ring pulse
    if(n.ring){
      n.ring.material.opacity=0.3+Math.sin(t*3)*0.2;
      const s=1+Math.sin(t*2)*0.08;n.ring.scale.setScalar(mSize(n.data.mastery)*7*s);
    }
    // Gentle float
    if(n.data&&n.data.status!=='locked'){
      const d=Math.sin(t*0.6+n.baseY*3)*0.1;
      n.mesh.position.y=n.baseY+d;
      if(n.glow)n.glow.position.y=n.baseY+d;
      if(n.ring)n.ring.position.y=n.baseY+d;
      if(n.label)n.label.position.y=n.baseY+d-((n.mesh.userData.size||0.5)+0.6);
    }
  });

  bGeo.rotateY(0.00005);
  controls.update();
  renderer.render(scene,camera);
  labelRenderer.render(scene,camera);
}

addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  labelRenderer.setSize(innerWidth,innerHeight);
});

fetchState();setInterval(fetchState,500);animate();
</script>
</body>
</html>
"""
