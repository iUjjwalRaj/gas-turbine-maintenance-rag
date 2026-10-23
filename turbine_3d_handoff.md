# 3D Gas Turbine Viewer — Design Handoff

Target: the existing Streamlit app in `/Users/ujjwalraj/Desktop/GEN AI EXAM`.
Implementer: a coding model working one milestone at a time.

---

## 0. Repository findings (verified)

| Item | Finding |
| :--- | :--- |
| Frontend | **Streamlit only** (`app.py`, Streamlit 1.66.1). No React, no `package.json`, no Node build. |
| Launch | `streamlit run app.py` → `http://localhost:8501` |
| Layout today | Header → `st.columns([7, 5])`: left = Q&A form + answer + citation + debug expander; right = 5 sample-question buttons. Sidebar = document/index status + engine mode. |
| Q&A entry point | `answer_question(query, vectorstore)` in `rag_pipeline.py`, returns `RAGResponse`. The form's text input reads `st.session_state.query_input`. |
| Component APIs available | `streamlit.components.v1`: `html`, `declare_component`, `iframe`. (`components.v2` also imports, but this design does not rely on it.) |

What can be exposed to the 3D view: only **navigation data** (which component was clicked, which question to prefill). The 3D view receives nothing from the RAG pipeline except an optional "highlight this component" argument.

---

## 1. Architecture decision

| | A. Custom Streamlit component (no build) | B. React + Vite frontend + Python API |
| :--- | :--- | :--- |
| New tooling | None (vendored Three.js files) | Node, npm, Vite, FastAPI/Flask, CORS |
| Backend changes | None | New HTTP API around `answer_question` |
| Breaks existing UI? | No — added as one block | Replaces Streamlit UI or runs two servers |
| Two-way events | Yes, via `declare_component` | Yes |
| Exam risk | **Low** | Medium–high |

**Recommendation: Option A.** A folder with `index.html` + vendored `three.module.js` + `OrbitControls.js`, registered with `components.declare_component(path=...)`. There's no npm and no second server, and the RAG backend is untouched.

Milestone 1 already uses `declare_component`, because it serves the folder over HTTP so the vendored ES modules load. It uses only `componentReady` + `setFrameHeight`, which proves the bridge early. Milestone 3 adds `setComponentValue` for click → Python events.

---

## 2. Visual specification

### Layout
Insert a full-width **"Turbine Explorer"** panel between the header and the existing `st.columns([7, 5])`. Height is **460 px** on desktop, and it is set from Python. Inside the iframe:

```
┌──────────────────────────────────────────────────────────────┐
│ [canvas: turbine]                          ┌───────────────┐ │
│                                            │ INFO PANEL    │ │
│   hover label ▸ "COMBUSTOR"                │ Combustor     │ │
│                                            │ short desc    │ │
│                                            │ [Ask about →] │ │
│ ◉ ◉ ◉ hotspots (amber)                     └───────────────┘ │
│ [⟲ Reset view]  [⏸ Rotation]     AIR ──▶ ──▶ ──▶ EXHAUST    │
└──────────────────────────────────────────────────────────────┘
```
- Info panel: absolute, top-right, width 260 px, collapses into a bottom sheet when the viewport is under 700 px wide.
- Buttons: bottom-left. Flow legend: bottom-right.

### Palette (CSS variables, which the shaders also reuse)
| Token | Hex | Use |
| :--- | :--- | :--- |
| `--bg-0` | `#0B0D10` | canvas clear color, panel background |
| `--bg-1` | `#14181D` | info panel, buttons |
| `--line` | `#262C34` | borders, grid |
| `--text` | `#E6EAF0` | primary text |
| `--muted` | `#8A94A3` | secondary text |
| `--blue` | `#3AA0FF` | rim light, selection outline, flow arrows |
| `--amber` | `#FFB020` | hotspots, warnings |

Font: `Inter, system-ui, sans-serif`. Labels are 11 px, uppercase, letter-spacing 0.08em. Panel title is 15 px/600.

### Materials (`MeshStandardMaterial`, no textures)
| Name | color | metalness | roughness | Use |
| :--- | :--- | :--- | :--- | :--- |
| `casing` | `#5B636E` | 0.85 | 0.38 | outer casings (shown cut-away, see §3) |
| `steel` | `#9AA3AE` | 0.95 | 0.25 | shaft, compressor blades |
| `hot` | `#8C6A52` | 0.8 | 0.45 | turbine blades/vanes (heat-tinted) |
| `combustor` | `#7A7F87` | 0.7 | 0.5 | combustor cans |
| `exhaust` | `#3F454D` | 0.6 | 0.55 | diffuser/exhaust |
| `base` | `#1A1E24` | 0.2 | 0.9 | skid/base plate |

Reflections: generate a **PMREM environment from `RoomEnvironment`** (`scene.environment`, intensity 0.6). This makes metals read as metal without loading an HDR file.

### Lighting
- `HemisphereLight(#bcd4ff, #0b0d10, 0.35)`
- Key `DirectionalLight(#ffffff, 1.6)` at `(6, 8, 6)`, shadows **off**
- Rim `DirectionalLight(#3AA0FF, 1.2)` at `(-8, 3, -6)` (gives the blue edge)
- Combustor glow: `PointLight(#FF8A3D, 0.8, 4)` inside the combustor ring, slowly pulsing ±15%
- Renderer: `ACESFilmicToneMapping`, `toneMappingExposure = 1.0`, `outputColorSpace = SRGBColorSpace`
- Fog: `scene.fog = new THREE.Fog(0x0B0D10, 14, 30)`

### Camera & animation
- `PerspectiveCamera(fov 35)`. Default position `(7.5, 3.2, 8.5)`, target `(0, 0.2, 0)` (a three-quarter view from the intake side).
- OrbitControls: `enableDamping` (0.08), `enablePan=false`, `minDistance 5`, `maxDistance 18`, `maxPolarAngle = 0.48π` (stays above the floor).
- Rotor spin: blades + shaft rotate about X at **0.6 rad/s**. Casings don't spin.
- Idle auto-orbit: `controls.autoRotate = true`, speed 0.4. Pause on user interaction and resume after 6 s idle. The ⏸ button toggles both spin and auto-orbit.
- Camera tween on select / reset: 700 ms ease-in-out (cubic) lerp of position + target. Write it by hand; don't pull in a tween library.

---

## 3. 3D construction plan (procedural)

**Axis convention:** the shaft lies on the **X axis**, air flows **−X → +X** (intake left, exhaust right). Units are arbitrary. The whole assembly is about 9 units long.

Everything sits in a `turbineGroup`. Each section is its own `THREE.Group` with `userData.componentId`, which is what picking and highlighting use.

| componentId | X range | Geometry (reusable builders) | Notes |
| :--- | :--- | :--- | :--- |
| `inlet` | −4.6 → −3.6 | `LatheGeometry` bell-mouth (profile: r 1.6 → 1.05) + 8 thin radial struts (`BoxGeometry`) | open front, so the compressor is visible |
| `compressor` | −3.6 → −0.6 | **10 stages**: each stage is a disc `CylinderGeometry` + `InstancedMesh` of blades (`BoxGeometry(0.03, h, 0.12)` rotated 25°). Blade height tapers 0.95 → 0.55; blades per stage 28 → 40 | rotor part spins |
| `combustor` | −0.6 → 0.6 | ring of **12 cans** (`CylinderGeometry` r 0.18, len 1.0, tilted 12° toward the shaft) on radius 1.15 + 12 small fuel-nozzle caps at the front | contains the glow light |
| `turbine` | 0.6 → 2.0 | **3 stages**: nozzle ring (static vanes, `hot`) + rotor disc + bucket blades `InstancedMesh`, blade height 0.6 → 0.85 (expanding) | rotor part spins |
| `exhaust` | 2.0 → 4.4 | `LatheGeometry` diffuser cone (r 1.0 → 1.5) + rectangular exhaust collar `BoxGeometry` | |
| `rotor` | −4.0 → 3.0 | shaft `CylinderGeometry` r 0.12 along X + 2 bearing housings (`TorusGeometry`) at x=−4.0 and x=2.6 | spins |
| `casing` | −3.6 → 2.0 | **cut-away** shell: `CylinderGeometry(..., openEnded, thetaStart=0, thetaLength=1.25π)` so the upper-front quadrant is open; `side: DoubleSide` | not clickable |
| — base | — | `BoxGeometry(10, 0.25, 3)` at y=−1.85 + 4 support legs | not clickable |

Helper builders to write once and reuse:
- `makeBladeRing({count, radius, height, chord, twist, material})` → InstancedMesh
- `makeStage({x, discRadius, bladeHeight, count, material})` → Group
- `makeLathe(profilePoints, material)` → Mesh

**Flow cue:** 3 thin `TubeGeometry` arrows above the casing (blue, opacity 0.5) with an animated dashed texture offset to indicate airflow, or simply small cones that slide along X and loop. Pick the cones; they're simpler.

**Polygon budget:** < 150k triangles. Use `InstancedMesh` for all blades, which keeps draw calls under ~60.

### Hotspots
These are amber sprites/discs (`CircleGeometry` r 0.09 + outer `RingGeometry` that pulses in scale 1 → 1.6 → fade), always facing the camera (`lookAt(camera)` each frame). Set `depthTest: false` on the ring only so it stays visible through the casing gap.

| componentId | Hotspot position | Label |
| :--- | :--- | :--- |
| `inlet` | (−4.1, 1.4, 0.6) | INLET |
| `compressor` | (−2.1, 1.25, 0.8) | COMPRESSOR |
| `combustor` | (0.0, 1.55, 0.7) | COMBUSTOR |
| `turbine` | (1.3, 1.3, 0.8) | TURBINE (HGP) |
| `exhaust` | (3.3, 1.7, 0.6) | EXHAUST |
| `rotor` | (−3.0, 0.15, 0.9) | ROTOR / BEARINGS |

### Interaction
- **Picking:** one `Raycaster`. Test against hotspot meshes first, then section groups (recursive). Walk up `.parent` until `userData.componentId` is found.
- **Hover:** show an HTML label (an absolutely positioned div; project the hotspot position to screen) and set `emissive = #3AA0FF`, `emissiveIntensity 0.25` on that section's materials. Cursor becomes `pointer`.
- **Select (click):**
  1. The selected section keeps full opacity plus a blue emissive of 0.35.
  2. All other sections dim to **opacity 0.35** (`transparent: true`). The model stays readable, not hidden.
  3. The camera tweens to that section's preset view (`target = section center`, position offset `(2.5, 1.6, 4.0)` from center).
  4. The info panel opens.
  5. An event is sent to Python (milestone 3).
- Click empty space or press **Esc** to deselect: restore opacity, close the panel, keep the camera where it is.
- **Reset view:** tween back to the default camera, deselect, and resume auto-orbit.
- Materials must be **cloned per section** so that dimming one section doesn't affect others.

---

## 4. Component contract

### Files
```
components/
  turbine_viewer/
    __init__.py            # Python wrapper: turbine_viewer(...)
    frontend/
      index.html           # canvas + overlay DOM + importmap
      style.css
      main.js              # scene bootstrap, render loop, events
      turbine.js           # procedural geometry builders (pure, no DOM)
      components.js        # COMPONENTS metadata table (see below)
      streamlit_bridge.js  # postMessage protocol (milestone 3)
      vendor/
        three.module.js            # pinned three@0.160.0
        OrbitControls.js           # from three@0.160.0/examples/jsm/controls/
        RoomEnvironment.js         # from three@0.160.0/examples/jsm/environments/
```
**No new Python dependencies. No npm.** Vendor the three files once with `curl` from `https://unpkg.com/three@0.160.0/...` so that runtime works offline. `index.html` contains:
```html
<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script>
```
In `OrbitControls.js` and `RoomEnvironment.js`, the `import ... from 'three'` line resolves through the importmap with no edits needed.

### `components.js` — metadata (presentation only)
```js
export const COMPONENTS = {
  combustor: {
    id: "combustor",
    label: "Combustor",
    description: "Ring of combustion cans where fuel is mixed with compressed air and burned.",
    suggestedQuestion: "What does a combustion inspection include?",
  },
  // inlet, compressor, turbine, exhaust, rotor ...
};
```
Rules:
- Each `description` is **one neutral, generic sentence** about what the part is. It contains no intervals, numbers, prices, or recommendations.
- The info panel shows a footnote: *"Visual aid only. Ask the assistant for document-backed answers."*
- `suggestedQuestion` is just text that prefills the input. It goes through the normal RAG pipeline and may legitimately return `Not found`.

### Python wrapper (`components/turbine_viewer/__init__.py`)
```python
def turbine_viewer(
    selected_id: str | None = None,   # highlight driven from Python (optional)
    height: int = 460,
    auto_rotate: bool = True,
    key: str = "turbine_viewer",
) -> dict | None
```
Returns the **last event** or `None`:
```json
{ "type": "select",   "componentId": "combustor", "label": "Combustor",
  "suggestedQuestion": "What does a combustion inspection include?", "ts": 1729680000000 }
{ "type": "ask",      "componentId": "combustor", "question": "...", "ts": ... }
{ "type": "deselect", "ts": ... }
```
`ts` (`Date.now()`) makes each event unique, so Python can ignore repeats: store `last_ts` in `st.session_state`.

### Integration in `app.py` (the only app change, ~15 lines)
```python
from components.turbine_viewer import turbine_viewer

event = turbine_viewer(selected_id=st.session_state.get("turbine_selected"))
if event and event.get("ts") != st.session_state.get("turbine_last_ts"):
    st.session_state.turbine_last_ts = event["ts"]
    if event["type"] == "select":
        st.session_state.turbine_selected = event["componentId"]
    elif event["type"] == "ask":
        st.session_state.query_input = event["question"]   # prefill only
        st.rerun()
```
**Boundary:** the viewer never calls `answer_question`, and never displays answers or citations. "Ask about this" only prefills the existing text input. The user still presses **Ask Assistant**, so evidence validation and `Not found` behave exactly as now.

### Bridge protocol (`streamlit_bridge.js`)
This is the Streamlit custom-component message protocol, implemented without `streamlit-component-lib`:
```js
const send = (type, data={}) =>
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
export function ready()            { send("streamlit:componentReady", { apiVersion: 1 }); }
export function setHeight(h)       { send("streamlit:setFrameHeight", { height: h }); }
export function setValue(v)        { send("streamlit:setComponentValue", { value: v, dataType: "json" }); }
export function onRender(cb) {
  window.addEventListener("message", e => {
    if (e.data?.type === "streamlit:render") cb(e.data.args || {});
  });
}
```
> ⚠️ This protocol is not a formally documented public API. Milestone 3 must **verify** it round-trips (click → Python prints the event) before anything is built on top of it. If it fails, fall back to `components.html` (display only) plus a Streamlit `st.radio`/`st.pills` row underneath for component selection. The demo still works.

### Responsive behavior
- `ResizeObserver` on the container calls `renderer.setSize(w, h)` and `camera.aspect = w/h`.
- `renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75))`.
- If width < 700: info panel becomes a bottom sheet, labels hide (hotspots only), and the camera distance multiplies by 1.25.

### Loading / errors / fallback
- Before the first frame, show a centered spinner and "Initializing 3D view…" on `--bg-0`.
- WebGL check: create a canvas and call `getContext('webgl2') || getContext('webgl')`. If that is null, render a static SVG side-view schematic (5 labeled boxes along the flow arrow) whose labels are clickable and emit the same events.
- `try/catch` around scene init. On error, show the same SVG fallback plus a small "3D unavailable" note. Log to `console.error` only.
- `webglcontextlost` → show the fallback; never leave a blank iframe.

### Performance (ordinary laptop)
- One render loop with `requestAnimationFrame`. **Pause when `document.hidden`** or when the iframe is off-screen (`IntersectionObserver`).
- No shadows, no post-processing, `antialias: true`, and the pixel ratio cap above.
- Blades use instancing; geometry is built once; there are no per-frame allocations (reuse `Vector3`s).
- Target: ≥ 50 fps on integrated graphics, < 150k triangles, < 60 draw calls.
- **Streamlit rerun caveat:** the iframe persists across reruns as long as `key` is stable, so don't change `key` or `height` per rerun, or the scene will rebuild.

---

## 5. Implementation sequence (one milestone per prompt)

| # | Milestone | Done when |
| :-: | :--- | :--- |
| **1** | Vendor Three.js; build the `frontend/` scene: base, shaft, compressor stages, combustor cans, turbine stages, exhaust, cut-away casing, lighting, OrbitControls, rotation, reset button. Register with `declare_component(path=...)` using only `componentReady` + `setFrameHeight` (no return value yet) | Turbine renders in the app; drag, zoom, and reset work; existing Q&A still works |
| 2 | Hotspots, hover labels, raycast selection, dimming of other sections, camera tween, info panel from `components.js`, Esc/deselect, pause toggle | Every hotspot selects its section; panel shows label + neutral description |
| 3 | `declare_component` wrapper + bridge; verify round-trip; wire `select`/`ask` events into `app.py` to prefill `query_input` | Clicking "Ask about this" fills the input; pressing Ask runs the unchanged RAG pipeline |
| 4 | Responsive layout, WebGL/SVG fallback, loading state, visibility-based pause | Fallback shows with WebGL disabled; resize keeps aspect correct |
| 5 | Polish: flow arrows, combustor glow pulse, fog, label styling; README section + screenshots | Acceptance checklist passes |

---

## 6. Acceptance checklist

- [ ] `streamlit run app.py` starts with no new Python dependencies; `pytest tests/ -v` still passes unchanged.
- [ ] Turbine is recognizable as an axial gas turbine: intake left, compressor (many short stages), combustor ring, 3 turbine stages, exhaust diffuser right.
- [ ] Renders offline (no CDN requests in DevTools Network tab).
- [ ] Drag orbits, scroll zooms within limits, camera can't go under the floor, reset returns to the default view.
- [ ] Rotor spins; casings don't; pause button stops spin and auto-orbit.
- [ ] Each of the 6 hotspots highlights the correct section on hover and selects it on click; other sections dim but stay visible.
- [ ] Info panel text contains no numbers, intervals, prices, or maintenance recommendations.
- [ ] "Ask about this" only prefills the question box and doesn't auto-submit. Answers still come only from `answer_question`, and unsupported questions still return `Not found`.
- [ ] No full-scene rebuild on unrelated Streamlit reruns (rotation doesn't visibly restart when using the Q&A form).
- [ ] At 1366×768 the turbine silhouette and info panel are both fully visible; under 700 px wide the panel becomes a bottom sheet.
- [ ] With WebGL disabled, the SVG fallback appears and remains clickable.
- [ ] ≥ 50 fps idle on a typical laptop (check with DevTools Performance or `renderer.info`).

---

## 7. Technical risks

| Risk | Mitigation |
| :--- | :--- |
| Custom-component postMessage protocol differs in this Streamlit version | Verify in milestone 3 with a minimal button before wiring the real UI; fallback = `components.html` + `st.pills` selector |
| Relative imports / importmap fail inside the component iframe | Always serve via `declare_component(path=...)`, which serves the folder over HTTP. Don't use `components.html` for this: it can't serve the vendored module files |
| Scene rebuilds on every Streamlit rerun | Stable `key`, constant `height`; don't pass changing args except `selected_id` |
| Transparent dimming causes sorting artifacts | Dimmed materials: `depthWrite = false`; keep the selected section opaque; set `renderOrder` so the casing renders last |
| Coding model writes facts into descriptions | Explicit rule in §4 + acceptance item; review `components.js` manually |
| Cut-away casing looks like a broken mesh | `DoubleSide`, slightly darker inner color, thin `TorusGeometry` lip on cut edges |
| Over-engineering (shaders, GLTF, postprocessing) | Out of scope; standard materials + RoomEnvironment are enough |
| Blank iframe on error | Every init path ends in either a frame or the SVG fallback |
