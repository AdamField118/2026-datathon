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

// Scale
const loadings = features.features.map(f => f.loadings);
const axisNorm = axes.map((_, j) => Math.hypot(...loadings.map(w => w[j])));
const STRETCH = axisNorm.map(v => v / axisNorm[0]);

// Colours
const ramp = t => new THREE.Color().setHSL(0.62 - 0.55 * t, 0.85, 0.55);            // blue -> green -> orange
const diverge = t => t < 0.5 ? new THREE.Color('#3b82f6').lerp(new THREE.Color('#555555'), t * 2)
	: new THREE.Color('#555555').lerp(new THREE.Color('#ef4444'), t * 2 - 1);          // blue -> grey -> red
const NO_DATA = new THREE.Color('#333333');
const [AGE_LO, AGE_HI] = [20, 36];   // clipped
const IMPACT_CLIP = 6;

const COLOR_MODES = {
	position: {
		label: 'Position (play-by-play)',
		color: p => ramp((p.pos_num_pbp - 1) / 4),
		scale: ramp, ticks: ['PG', 'SG', 'SF', 'PF', 'C'],
		note: 'Minutes-weighted position from Basketball-Reference play-by-play. The model never saw it.',
	},
	age: {
		label: 'Age',
		color: p => ramp(Math.min(Math.max((p.age - AGE_LO) / (AGE_HI - AGE_LO), 0), 1)),
		scale: ramp, ticks: [`≤${AGE_LO}`, 24, 28, 32, `≥${AGE_HI}`],
		note: 'Age this season, clipped to 20-36. The model never saw it.',
	},
	impact: {
		label: 'Impact beyond style',
		color: p => p.impact.beyond_style == null ? NO_DATA
			: diverge(Math.min(Math.max(p.impact.beyond_style / IMPACT_CLIP, -1), 1) / 2 + 0.5),
		scale: diverge, ticks: [`-${IMPACT_CLIP}`, '0', `+${IMPACT_CLIP}`],
		note: 'On-off net rating (per 100 possessions) minus what the player\'s style predicts, shrunk toward 0 for low minutes. Clipped at ±6.',
	},
	awards: {
		label: 'Awards',
		color: p => p.awards.all_nba && p.awards.all_def ? new THREE.Color('#ff66cc')
			: p.awards.all_nba ? new THREE.Color('#ffcc00')
			: p.awards.all_def ? new THREE.Color('#33ccff')
			: p.awards.all_star ? new THREE.Color('#ffffff') : new THREE.Color('#444444'),
		swatches: [['#ffcc00', 'All-NBA'], ['#33ccff', 'All-Defensive'], ['#ff66cc', 'Both'],
			['#ffffff', 'All-Star (other)'], ['#444444', 'None']],
		note: 'Season awards. The model never saw them.',
	},
};

// Scene, cameras, renderers
const stage = document.querySelector('.stage');
const canvas = document.querySelector('canvas.webgl');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b0b0b');

const HALF = 4;   // 2D view
const cam2d = new THREE.OrthographicCamera(-HALF, HALF, HALF, -HALF, 0.1, 100);
cam2d.position.set(0, 0, 20);
const cam3d = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
cam3d.position.set(6, 4, 6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
stage.appendChild(labelRenderer.domElement);

const controls2d = new OrbitControls(cam2d, canvas);
controls2d.enableRotate = false;   // 2D: pan and zoom only
const controls3d = new OrbitControls(cam3d, canvas);
controls3d.enableDamping = true;

function resize() {
	const w = stage.clientWidth, h = stage.clientHeight, aspect = w / h;
	cam2d.left = -HALF * aspect;
	cam2d.right = HALF * aspect;
	cam2d.updateProjectionMatrix();
	cam3d.aspect = aspect;
	cam3d.updateProjectionMatrix();
	renderer.setSize(w, h, false);
	labelRenderer.setSize(w, h);
}
new ResizeObserver(resize).observe(stage);

// Axes
const axesPos = new THREE.AxesHelper(AXIS_LEN);
const axesNeg = new THREE.AxesHelper(AXIS_LEN);
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
var selected = null;
const dummy = new THREE.Object3D();
const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10),
	new THREE.MeshBasicMaterial(), players.length);
scene.add(dots);

// State
let mode = '2d';
let picked = [0, 1];
let colorBy = 'position';
let stretch = true;
const need = () => mode === '2d' ? 2 : 3;
const factor = j => stretch ? STRETCH[j] : 1;

function draw() {
	const sel = picked;
	const cm = COLOR_MODES[colorBy];
	const f = [0, 1, 2].map(s => sel[s] == null ? 0 : factor(sel[s]));
	players.forEach((p, i) => {
		const c = [0, 1, 2].map(s => sel[s] == null ? 0 : p.z[sel[s]] * f[s]);
		dummy.position.set(c[0], c[1], c[2]);
		const big = colorBy === 'awards' && (p.awards.all_nba || p.awards.all_def);
		dummy.scale.setScalar(i === selected ? 0.12 : big ? 0.09 : 0.05);
		dummy.updateMatrix();
		dots.setMatrixAt(i, dummy.matrix);
		dots.setColorAt(i, i == selected ? new THREE.Color(0xff0000) : cm.color(p));
	});
	dots.instanceMatrix.needsUpdate = true;
	dots.instanceColor.needsUpdate = true;

	const len = f.map(v => Math.max(v, 1e-3));
	axesPos.scale.set(len[0], len[1], len[2]);
	axesNeg.scale.set(-len[0], -len[1], -len[2]);
	labels.clear();
	sel.forEach((j, s) => {
		const a = axes[j];
		const at = Math.max(AXIS_LEN * f[s], 1.2) + 0.3;   // keep labels readable on short axes
		const dir = new THREE.Vector3().setComponent(s, 1);
		label(`${SLOTS[s]}+ ${a.name_pos ?? `axis ${j + 1} +`}`, SLOT_COLORS[s], dir.clone().multiplyScalar(at));
		label(`${SLOTS[s]}- ${a.name_neg ?? `axis ${j + 1} -`}`, SLOT_COLORS[s], dir.clone().multiplyScalar(-at));
	});
	renderLegend();
}

// Controls panel
const list = document.getElementById('axis-list');

function renderList() {
	document.getElementById('pick-hint').textContent = `pick ${need() === 2 ? 'two' : 'three'}`;
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
				if (picked.length > need()) picked.shift();
			} else {
				picked = picked.filter(k => k !== j);
			}
			renderList();
			if (picked.length === need()) draw();
		};
		list.appendChild(li);
	});
}

const playerDisplay = document.getElementById('player-data-card');
const buildDisplay = p => {
	var rows = "";
	p.z.forEach((axis, i) => {
		rows += `<tr><td>${axisName(axes[i])}</td><td>${axis}</td></tr>`;
	})

	return `<h2>${p.name}</h2>
	<h4>${p.team}</h2>
	<table>
		<tr><th>Axis</th><th>Position</th></tr>
		${rows}
	</table>
	<span id="headshot">
	<img src='https://www.basketball-reference.com/req/202106291/images/headshots/${p.pid}.jpg'>
	</span>`;
}

// Raycaster
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

canvas.addEventListener('click', (event) => {
	const rect = renderer.domElement.getBoundingClientRect();
	mouse.x = ( ( event.clientX - rect.left ) / ( rect.right - rect.left ) ) * 2 - 1;
	mouse.y = - ( ( event.clientY - rect.top ) / ( rect.bottom - rect.top) ) * 2 + 1;

	raycaster.setFromCamera(mouse, mode === '2d' ? cam2d : cam3d);

	const intersects = raycaster.intersectObject(dots);



	if (intersects.length > 0) {
		selectPlayer(intersects[0].instanceId);
	}
});

// Selecting a player (from a click on the graph or from the search box)
function selectPlayer(i) {
	selected = i;
	playerDisplay.hidden = false;
	playerDisplay.innerHTML = buildDisplay(players[i]);
	search.value = players[i].name;
	results.hidden = true;
	draw();   // draw() colours the selected dot red and enlarges it
}

// Player search: fuzzy find on names (accents ignored, like "doncic" finds Dončić)
const search = document.getElementById('player-search');
const results = document.getElementById('search-results');
const fold = s => s.normalize('NFD').replace(/[^a-zA-Z0-9\s.'-]/g, '').toLowerCase();
const folded = players.map(p => Array.from(fold(p.name)).join(''));
let matches = [];
let active = 0;

// every query letter must appear in order; consecutive letters and word starts score higher
function fuzzy(query, text) {
	let score = 0, from = 0, prev = -2;
	const hits = [];
	for (const ch of query.replace(/\s+/g, '')) {
		const k = text.indexOf(ch, from);
		if (k < 0) return null;
		score += k === prev + 1 ? 3 : 1;
		if (k === 0 || /[\s.'-]/.test(text[k - 1])) score += 2;
		hits.push(k);
		prev = k;
		from = k + 1;
	}
	return { score, hits };
}

function renderResults() {
	results.hidden = matches.length === 0;
	results.innerHTML = matches.map(({ i, hits }, n) => {
		const name = Array.from(players[i].name)
			.map((ch, k) => hits.includes(k) ? `<mark>${ch}</mark>` : ch).join('');
		return `<li data-n="${n}" class="${n === active ? 'active' : ''}">
			<span>${name}</span><span class="meta">${players[i].team} - ${players[i].pos_listed}</span></li>`;
	}).join('');
}

search.addEventListener('input', () => {
	const q = fold(search.value.trim());
	matches = q ? players.map((p, i) => ({ i, ...fuzzy(q, folded[i]) }))
		.filter(m => m.hits)
		.sort((a, b) => b.score - a.score || players[b.i].mp - players[a.i].mp)
		.slice(0, 8) : [];
	active = 0;
	renderResults();
});

search.addEventListener('keydown', e => {
	if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
		e.preventDefault();
		if (!matches.length) return;
		active = (active + (e.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length;
		renderResults();
	} else if (e.key === 'Enter' && matches.length) {
		selectPlayer(matches[active].i);
	} else if (e.key === 'Escape') {
		results.hidden = true;
	}
});

// mousedown (not click) so the choice lands before the input loses focus
results.addEventListener('mousedown', e => {
	const li = e.target.closest('li');
	if (li) {
		e.preventDefault();
		selectPlayer(matches[+li.dataset.n].i);
	}
});
search.addEventListener('blur', () => { results.hidden = true; });
search.addEventListener('focus', () => { if (matches.length) results.hidden = false; });

function renderLegend() {
	const cm = COLOR_MODES[colorBy];
	const box = document.getElementById('color-legend');
	if (cm.swatches) {
		box.innerHTML = cm.swatches.map(([c, t]) =>
			`<div class="swatch"><span style="background:${c}"></span>${t}</div>`).join('');
	} else {
		const stops = [0, 0.25, 0.5, 0.75, 1].map(t => cm.scale(t).getStyle()).join(', ');
		box.innerHTML = `<div class="gradient" style="background: linear-gradient(to right, ${stops})"></div>
			<div class="ticks">${cm.ticks.map(t => `<span>${t}</span>`).join('')}</div>`;
	}
	box.innerHTML += `<p class="note">${cm.note}</p>`;

	document.getElementById('scale-note').innerHTML = stretch
		? `Each axis is stretched by how much it changes the player's stats (the length of its loading vector), relative to axis 1. Longer axis = more important.
		   <br>${picked.map((j, s) => `<b style="color:${SLOT_COLORS[s]}">${SLOTS[s]}</b> ×${STRETCH[j].toFixed(2)}`).join(' &nbsp; ')}`
		: 'Not stretched: every axis is drawn with the same spread, so small axes look as big as large ones.';
}

const select = document.getElementById('color-by');
select.innerHTML = Object.entries(COLOR_MODES).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join('');
select.onchange = () => { colorBy = select.value; draw(); };

document.getElementById('stretch').onchange = e => { stretch = e.target.checked; draw(); };

const btn = document.getElementById('view-toggle');
btn.onclick = () => {
	mode = mode === '2d' ? '3d' : '2d';
	if (mode === '2d') picked = picked.slice(0, 2);
	else while (picked.length < 3) picked.push([0, 1, 2, 3].find(j => !picked.includes(j)));
	btn.textContent = mode === '2d' ? 'Switch to 3D' : 'Switch to 2D';
	controls2d.enabled = mode === '2d';
	controls3d.enabled = mode === '3d';
	renderList();
	draw();
};
controls3d.enabled = false;

renderList();
draw();

// Animate
renderer.setAnimationLoop(() => {
	const cam = mode === '2d' ? cam2d : cam3d;
	(mode === '2d' ? controls2d : controls3d).update();
	renderer.render(scene, cam);
	labelRenderer.render(scene, cam);
});
