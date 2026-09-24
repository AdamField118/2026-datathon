import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const AXIS_LEN = 3.5;
const SLOTS = ['X', 'Y', 'Z'];
const SLOT_COLORS = ['#ff5555', '#55dd55', '#5599ff'];   // AxesHelper's red / green / blue

// Data
const [players, features] = await Promise.all([
	fetch('./results/players.json').then(r => r.json()),
	fetch('./results/features.json').then(r => r.json()),
]);
const axes = features.axes;
const axisName = a => a.name_neg ? `${a.name_neg} <--> ${a.name_pos}` : `Axis ${a.id + 1}`;

// Scene, camera, renderers
const stage = document.querySelector('.stage');
const canvas = document.querySelector('canvas.webgl');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b0b0b');

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
camera.position.set(6, 4, 6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
stage.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

function resize() {
	const w = stage.clientWidth, h = stage.clientHeight;
	camera.aspect = w / h;
	camera.updateProjectionMatrix();
	renderer.setSize(w, h, false);
	labelRenderer.setSize(w, h);
}
new ResizeObserver(resize).observe(stage);

// Axes
const axesPos = new THREE.AxesHelper(AXIS_LEN);
const axesNeg = new THREE.AxesHelper(AXIS_LEN);
axesNeg.scale.set(-1, -1, -1);
axesNeg.material.transparent = true;
axesNeg.material.opacity = 0.4;
scene.add(axesPos, axesNeg);

const labels = new THREE.Group();
scene.add(labels);

function label(text, color, pos) {
	const div = document.createElement('div');
	div.className = 'axis-label';
	div.textContent = text;
	div.style.color = color;
	const obj = new CSS2DObject(div);
	obj.position.copy(pos);
	labels.add(obj);
}

// Players
const dummy = new THREE.Object3D();
const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10),
	new THREE.MeshBasicMaterial(), players.length);
scene.add(dots);

function setAxes(sel) {
	const extremes = new Set(sel.flatMap(j => [...axes[j].players_neg, ...axes[j].players_pos]));
	players.forEach((p, i) => {
		const hot = extremes.has(p.pid);
		dummy.position.set(p.z[sel[0]], p.z[sel[1]], p.z[sel[2]]);
		dummy.scale.setScalar(hot ? 0.09 : 0.045);
		dummy.updateMatrix();
		dots.setMatrixAt(i, dummy.matrix);
		dots.setColorAt(i, new THREE.Color(hot ? '#ffcc00' : '#dddddd'));
	});
	dots.instanceMatrix.needsUpdate = true;
	dots.instanceColor.needsUpdate = true;

	labels.clear();
	sel.forEach((j, s) => {
		const a = axes[j];
		const dir = new THREE.Vector3().setComponent(s, 1);
		label(`${SLOTS[s]}+ ${a.name_pos ?? `axis ${j + 1} +`}`, SLOT_COLORS[s], dir.clone().multiplyScalar(AXIS_LEN + 0.3));
		label(`${SLOTS[s]}− ${a.name_neg ?? `axis ${j + 1} −`}`, SLOT_COLORS[s], dir.clone().multiplyScalar(-AXIS_LEN - 0.3));
	});
}

// Axis picker
const list = document.getElementById('axis-list');
let picked = [0, 1, 2];

function renderList() {
	list.innerHTML = '';
	axes.forEach((a, j) => {
		const s = picked.indexOf(j);
		const li = document.createElement('li');
		li.innerHTML = `<label>
			<input type="checkbox" ${s >= 0 ? 'checked' : ''}>
			<span class="slot" style="color:${SLOT_COLORS[s] ?? 'transparent'}">${SLOTS[s] ?? ''}</span>
			<span>${axisName(a)} <small>(${Math.round(100 * a.var_share)}%)</small></span></label>`;
		li.querySelector('input').onchange = e => {
			if (e.target.checked) {
				picked.push(j);
				if (picked.length > 3) picked.shift();
			} else {
				picked = picked.filter(k => k !== j);
			}
			renderList();
			if (picked.length === 3) setAxes(picked);
		};
		list.appendChild(li);
	});
}

renderList();
setAxes(picked);

// Animate
renderer.setAnimationLoop(() => {
	controls.update();
	renderer.render(scene, camera);
	labelRenderer.render(scene, camera);
});