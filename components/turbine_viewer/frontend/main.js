import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { RoomEnvironment } from './vendor/RoomEnvironment.js';
import { createTurbineScene } from './turbine.js';
import * as bridge from './streamlit_bridge.js';

// Configuration constants - framed heroically, centered slightly to right to balance left chat panel
const DEFAULT_CAM_POS = new THREE.Vector3(7.0, 2.7, 7.8);
const DEFAULT_CAM_TARGET = new THREE.Vector3(0.4, 0.1, 0);
const DEFAULT_HEIGHT = Math.max(window.innerHeight || 850, 750);

// State management
let scene, camera, renderer, controls, turbineData;
let isRotating = true;
let userInteracting = false;
let idleTimer = null;
let lastTime = 0;
let firstFrameRendered = false;

// Camera tween state
let tweenActive = false;
let tweenStart = 0;
const tweenDuration = 700; // ms
const startPos = new THREE.Vector3();
const endPos = new THREE.Vector3();
const startTarget = new THREE.Vector3();
const endTarget = new THREE.Vector3();

// Cubic ease in-out
function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function checkWebGLSupport() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl2') || canvas.getContext('webgl')));
  } catch (e) {
    return false;
  }
}

function showFallback() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.classList.add('hidden');
  const fallback = document.getElementById('fallback-container');
  if (fallback) fallback.style.display = 'flex';
  const canvas = document.getElementById('turbine-canvas');
  if (canvas) canvas.style.display = 'none';
}

function init() {
  if (!checkWebGLSupport()) {
    console.warn("WebGL is not supported in this browser. Displaying SVG schematic fallback.");
    showFallback();
    bridge.ready();
    bridge.setHeight(window.innerHeight || DEFAULT_HEIGHT);
    return;
  }

  const container = document.getElementById('viewer-container');
  const canvas = document.getElementById('turbine-canvas');

  try {
    // 1. Scene Setup
    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x080b10, 16, 36);

    // 2. Camera Setup
    const width = window.innerWidth || container.clientWidth || 1200;
    const height = window.innerHeight || container.clientHeight || DEFAULT_HEIGHT;
    camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
    camera.position.copy(DEFAULT_CAM_POS);

    // 3. Renderer Setup
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setSize(width, height);
    renderer.setClearColor(0x080b10, 1.0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Handle WebGL context loss
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      console.warn("WebGL context lost. Falling back to schematic.");
      showFallback();
    }, false);

    // 4. Studio Environment Lighting (RoomEnvironment)
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();
    const roomEnvironment = new RoomEnvironment(renderer);
    scene.environment = pmremGenerator.fromScene(roomEnvironment, 0.04).texture;
    if ('environmentIntensity' in scene) {
      scene.environmentIntensity = 0.65;
    }

    // 5. Cinematic Industrial Lighting
    // Ambient fill
    const hemiLight = new THREE.HemisphereLight(0x9bc2ff, 0x080b10, 0.45);
    scene.add(hemiLight);

    // Key Light (Main white highlight from upper-front-right)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
    keyLight.position.set(6, 8, 6);
    scene.add(keyLight);

    // Atmospheric Electric Blue Rim Light (Back-left giving blade and casing silhouette edge glow)
    const rimLight = new THREE.DirectionalLight(0x3aa0ff, 1.6);
    rimLight.position.set(-8, 3, -6);
    scene.add(rimLight);

    // Deep Teal/Blue Underbody Fill
    const fillLight = new THREE.DirectionalLight(0x204875, 0.85);
    fillLight.position.set(2, -4, 5);
    scene.add(fillLight);

    // 6. Build Procedural Gas Turbine Assembly
    turbineData = createTurbineScene();
    scene.add(turbineData.root);

    // 7. OrbitControls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 4.2;
    controls.maxDistance = 20.0;
    controls.maxPolarAngle = 0.48 * Math.PI; // Don't go below base floor
    controls.target.copy(DEFAULT_CAM_TARGET);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.45;

    // Auto-orbit idle resume handler
    controls.addEventListener('start', () => {
      userInteracting = true;
      if (idleTimer) clearTimeout(idleTimer);
    });

    controls.addEventListener('end', () => {
      userInteracting = false;
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (!userInteracting && isRotating) {
          controls.autoRotate = true;
        }
      }, 6000);
    });

    // 8. UI Button Event Listeners
    setupUIControls();

    // 9. Resize Handling via window resize & ResizeObserver
    const handleResize = () => {
      const w = window.innerWidth || container.clientWidth;
      const h = window.innerHeight || container.clientHeight;
      if (w > 0 && h > 0) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
        bridge.setHeight(h);
      }
    };
    window.addEventListener('resize', handleResize);

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });
    resizeObserver.observe(container);

    // 10. Start Animation Loop
    lastTime = performance.now();
    requestAnimationFrame(animate);

    // 11. Streamlit Component Handshake
    bridge.ready();
    bridge.setHeight(window.innerHeight || DEFAULT_HEIGHT);

    // Handle render updates from Streamlit parent
    bridge.onRender((args) => {
      if (args && args.height) {
        bridge.setHeight(args.height);
      }
      if (args && typeof args.auto_rotate === 'boolean') {
        isRotating = args.auto_rotate;
        controls.autoRotate = args.auto_rotate;
      }
    });

  } catch (err) {
    console.error("Failed to initialize Three.js 3D turbine viewer:", err);
    showFallback();
    bridge.ready();
    bridge.setHeight(window.innerHeight || DEFAULT_HEIGHT);
  }
}

function setupUIControls() {
  const resetBtn = document.getElementById('btn-reset');
  const rotateBtn = document.getElementById('btn-rotate');

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      // Tween camera back to default angle
      startPos.copy(camera.position);
      endPos.copy(DEFAULT_CAM_POS);
      startTarget.copy(controls.target);
      endTarget.copy(DEFAULT_CAM_TARGET);
      tweenStart = performance.now();
      tweenActive = true;

      // Resume auto-orbit and rotor rotation
      isRotating = true;
      controls.autoRotate = true;
      if (rotateBtn) {
        rotateBtn.innerHTML = '<span>⏸</span> Rotation';
      }
    });
  }

  if (rotateBtn) {
    rotateBtn.addEventListener('click', () => {
      isRotating = !isRotating;
      controls.autoRotate = isRotating;
      rotateBtn.innerHTML = isRotating ? '<span>⏸</span> Rotation' : '<span>▶</span> Rotation';
    });
  }
}

function animate(now) {
  // Pause render loop if document is not visible
  if (document.hidden) {
    lastTime = now;
    requestAnimationFrame(animate);
    return;
  }

  const delta = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  // 1. Rotor spin & effects
  if (turbineData && turbineData.rotorGroup && isRotating) {
    turbineData.rotorGroup.rotation.x += 0.65 * delta;
  }

  // Combustor warm flame glow pulsation
  if (turbineData && turbineData.combustorGlow) {
    turbineData.combustorGlow.intensity = 0.90 + Math.sin(now * 0.0035) * 0.22;
  }

  // Subtle airflow animation
  if (turbineData && turbineData.flowGroup) {
    const offset = (now * 0.0006) % 0.6 - 0.3;
    turbineData.flowGroup.position.x = offset;
  }

  // 2. Camera tween animation
  if (tweenActive) {
    const elapsed = now - tweenStart;
    const progress = Math.min(elapsed / tweenDuration, 1.0);
    const eased = easeInOutCubic(progress);

    camera.position.lerpVectors(startPos, endPos, eased);
    controls.target.lerpVectors(startTarget, endTarget, eased);

    if (progress >= 1.0) {
      tweenActive = false;
    }
  }

  // 3. OrbitControls & Render
  controls.update();
  renderer.render(scene, camera);

  // 4. Remove loading overlay upon first successful frame
  if (!firstFrameRendered) {
    firstFrameRendered = true;
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
      overlay.classList.add('hidden');
    }
  }

  requestAnimationFrame(animate);
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
