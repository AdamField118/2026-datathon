import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function newPoint(scene, x, y, z, color, radius) {
		if (radius == null) {
				radius = 0.1
		}
		const geo = (new THREE.SphereGeometry(radius, 10, 10)).translate(x,y,z);
		const material = new THREE.MeshBasicMaterial({ color: color });
		const point = new THREE.Mesh(geo, material);
		scene.add(point);
}

function setAxes(scene, x, y, z) {
		while(scene.children.length > 0){ 
			scene.remove(scene.children[0]); 
		}

		fetch('./results/players.json').then( res => {
				if (!res.ok) {
						throw new Error("failed to get player data");
				} else {
						console.log(res);
						return res.json()
				}
		}).then( playerdata => {
				fetch('./results/features.json').then( res => {
						if (!res.ok) {
								throw new Error("failed to get features");
						} else {
								console.log(res);
								return res.json()
						}
				}).then( features => {
						console.log(features)
						const players1 = features.axes[x].players_neg + features.axes[x].players_pos;
						const players2 = features.axes[y].players_neg + features.axes[y].players_pos;
						const players3 = features.axes[z].players_neg + features.axes[z].players_pos;

						console.log(playerdata);
						for(let p in playerdata) {
								const player = playerdata[p];
								var color;
								var radius;
								if (players1.includes(player.pid) || players2.includes(player.pid) || players3.includes(player.pid)) {
										color = new THREE.Color(1,0,0);
										radius = 0.1;
								} else {
										color = new THREE.Color(1,1,1);
										radius = 0.05;
								}
								newPoint(scene, player.z[x], player.z[y], player.z[z], color, radius);
						}
				}
		)})
}


// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

//const square = 5;
//for (let x = 0; x < square; x++) {
//		for (let y = 0; y < square; y++) {
//				for (let z = 0; z < square; z++) {
//						newPoint(scene, x,y,z, new THREE.Color(x/5, y/5, z/5));
//}}}

// Lights

const pointLight = new THREE.PointLight(0xffffff, 0.1)
pointLight.position.x = 2
pointLight.position.y = 3
pointLight.position.z = 4
scene.add(pointLight)

/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

window.addEventListener('resize', () =>
{
    // Update sizes
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    // Update camera
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(sizes.width, sizes.height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100)
camera.position.x = -5
camera.position.y = 2
camera.position.z = -5
camera.lookAt(0,0,0);
scene.add(camera)


// Controls
const controls = new OrbitControls(camera, canvas)
// controls.enableDamping = true

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
    canvas: canvas
})
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

/**
 * Animate
 */
document.getElementById('update').onclick = () => {
		const axis1 = document.querySelector('input[name="axis1"]:checked').value;
		const axis2 = document.querySelector('input[name="axis2"]:checked').value;
		const axis3 = document.querySelector('input[name="axis3"]:checked').value;
		setAxes(scene, axis1, axis2, axis3);
}

const clock = new THREE.Clock()

const tick = () =>
{

    const elapsedTime = clock.getElapsedTime()

    // Update Orbital Controls
    controls.update()

    // Render
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}

tick()
