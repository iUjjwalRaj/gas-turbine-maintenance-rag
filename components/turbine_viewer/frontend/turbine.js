import * as THREE from 'three';

/**
 * Procedural Industrial Gas Turbine Model Builder
 * Follows GE Heavy-Duty Frame Gas Turbine architecture:
 * Shaft along X axis; Flow moves from -X (inlet) to +X (exhaust).
 */

export function createTurbineScene() {
  const root = new THREE.Group();
  root.name = 'turbineAssembly';

  // 1. Materials according to specification
  const materials = {
    casing: new THREE.MeshStandardMaterial({
      color: 0x5b636e,
      metalness: 0.85,
      roughness: 0.38,
      side: THREE.DoubleSide
    }),
    casingInner: new THREE.MeshStandardMaterial({
      color: 0x3d434b,
      metalness: 0.80,
      roughness: 0.45,
      side: THREE.DoubleSide
    }),
    cutLip: new THREE.MeshStandardMaterial({
      color: 0xd4a359, // Machined brass/bronze cutaway lip accent
      metalness: 0.90,
      roughness: 0.25
    }),
    steel: new THREE.MeshStandardMaterial({
      color: 0x9aa3ae,
      metalness: 0.95,
      roughness: 0.25
    }),
    hot: new THREE.MeshStandardMaterial({
      color: 0x8c6a52, // Heat-tinted nickel superalloy
      metalness: 0.80,
      roughness: 0.42
    }),
    combustor: new THREE.MeshStandardMaterial({
      color: 0x7a7f87,
      metalness: 0.70,
      roughness: 0.48
    }),
    exhaust: new THREE.MeshStandardMaterial({
      color: 0x3f454d,
      metalness: 0.60,
      roughness: 0.55
    }),
    base: new THREE.MeshStandardMaterial({
      color: 0x1a1e24,
      metalness: 0.25,
      roughness: 0.85
    }),
    glow: new THREE.MeshBasicMaterial({
      color: 0xff8a3d
    }),
    arrow: new THREE.MeshBasicMaterial({
      color: 0x3aa0ff,
      transparent: true,
      opacity: 0.75
    })
  };

  // 2. Base / Structural Skid
  const baseGroup = new THREE.Group();
  baseGroup.name = 'base';
  baseGroup.userData.componentId = 'base';

  // Heavy steel skid frame
  const skidGeom = new THREE.BoxGeometry(10.6, 0.24, 3.2);
  const skidMesh = new THREE.Mesh(skidGeom, materials.base);
  skidMesh.position.set(0, -1.85, 0);
  baseGroup.add(skidMesh);

  // Longitudinal I-beams / skid runners
  const runnerGeom = new THREE.BoxGeometry(10.6, 0.12, 0.2);
  const runner1 = new THREE.Mesh(runnerGeom, materials.base);
  runner1.position.set(0, -1.70, 1.45);
  const runner2 = new THREE.Mesh(runnerGeom, materials.base);
  runner2.position.set(0, -1.70, -1.45);
  baseGroup.add(runner1, runner2);

  // 6 Support pedestals anchoring bearing frames to base
  const legPositions = [
    [-3.8, -1.15, 0.9],
    [-3.8, -1.15, -0.9],
    [-0.5, -1.15, 1.1],
    [-0.5, -1.15, -1.1],
    [2.8, -1.15, 0.9],
    [2.8, -1.15, -0.9]
  ];
  const legGeom = new THREE.BoxGeometry(0.35, 1.2, 0.35);
  legPositions.forEach(([x, y, z]) => {
    const leg = new THREE.Mesh(legGeom, materials.base);
    leg.position.set(x, y, z);
    baseGroup.add(leg);
  });
  root.add(baseGroup);

  // 3. Rotating Assembly (Rotor Group)
  // This group will spin around the X-axis in the animation loop
  const rotorGroup = new THREE.Group();
  rotorGroup.name = 'rotorGroup';
  rotorGroup.userData.componentId = 'rotor';

  // Central shaft (cylinder along X)
  const shaftGeom = new THREE.CylinderGeometry(0.13, 0.13, 7.6, 32);
  shaftGeom.rotateZ(Math.PI / 2);
  const shaftMesh = new THREE.Mesh(shaftGeom, materials.steel);
  shaftMesh.position.set(-0.5, 0, 0);
  rotorGroup.add(shaftMesh);

  // Front & rear bearing journals on rotor
  [-3.8, 2.7].forEach((bx) => {
    const journalGeom = new THREE.CylinderGeometry(0.24, 0.24, 0.35, 32);
    journalGeom.rotateZ(Math.PI / 2);
    const journal = new THREE.Mesh(journalGeom, materials.steel);
    journal.position.set(bx, 0, 0);
    rotorGroup.add(journal);
  });

  // Nose aerodynamic bullet spinner attached to rotor intake
  const spinnerGeom = new THREE.ConeGeometry(0.42, 0.9, 32);
  spinnerGeom.rotateZ(Math.PI / 2); // points toward -X
  const spinner = new THREE.Mesh(spinnerGeom, materials.steel);
  spinner.position.set(-4.3, 0, 0);
  rotorGroup.add(spinner);

  // 4. 10 Axial Compressor Rotor Stages (X from -3.6 to -0.7)
  const compressorStagesCount = 10;
  const compStartX = -3.55;
  const compEndX = -0.75;
  const compSpacing = (compEndX - compStartX) / (compressorStagesCount - 1);

  const dummy = new THREE.Object3D();

  for (let s = 0; s < compressorStagesCount; s++) {
    const t = s / (compressorStagesCount - 1);
    const stageX = compStartX + s * compSpacing;

    // Disc radius increases as blade height tapers down (constant or slightly tapering outer radius)
    const bladeHeight = THREE.MathUtils.lerp(0.92, 0.52, t);
    const discRadius = THREE.MathUtils.lerp(0.42, 0.72, t);
    const bladeCount = Math.round(THREE.MathUtils.lerp(28, 42, t));

    // Rotor disc
    const discGeom = new THREE.CylinderGeometry(discRadius, discRadius, 0.08, 32);
    discGeom.rotateZ(Math.PI / 2);
    const discMesh = new THREE.Mesh(discGeom, materials.steel);
    discMesh.position.set(stageX, 0, 0);
    rotorGroup.add(discMesh);

    // Instanced airfoil blades around the disc rim
    const bladeGeom = new THREE.BoxGeometry(0.045, bladeHeight, 0.11);
    const bladeMesh = new THREE.InstancedMesh(bladeGeom, materials.steel, bladeCount);

    for (let b = 0; b < bladeCount; b++) {
      const angle = (b / bladeCount) * Math.PI * 2;
      const radialDist = discRadius + bladeHeight / 2;
      const y = Math.cos(angle) * radialDist;
      const z = Math.sin(angle) * radialDist;

      dummy.position.set(stageX, y, z);
      // Rotation: align blade outward along radial vector, with aerodynamic stagger angle
      dummy.rotation.set(0, 0, 0);
      dummy.rotation.x = -angle; // align with radius
      dummy.rotation.y = 0.42;   // stagger angle (~24 deg)
      dummy.updateMatrix();

      bladeMesh.setMatrixAt(b, dummy.matrix);
    }
    bladeMesh.instanceMatrix.needsUpdate = true;
    rotorGroup.add(bladeMesh);
  }

  // 5. 3 Turbine Rotor Stages (Hot Gas Path, X from 0.85 to 1.85)
  const turbineStagesCount = 3;
  const turbXs = [0.90, 1.35, 1.80];
  const turbBladeHeights = [0.65, 0.76, 0.88]; // expanding hot gas stages
  const turbDiscRadii = [0.68, 0.62, 0.56];
  const turbBladeCounts = [36, 40, 46];

  for (let s = 0; s < turbineStagesCount; s++) {
    const stageX = turbXs[s];
    const bladeHeight = turbBladeHeights[s];
    const discRadius = turbDiscRadii[s];
    const bladeCount = turbBladeCounts[s];

    // Heavy forged turbine disc
    const discGeom = new THREE.CylinderGeometry(discRadius, discRadius, 0.11, 32);
    discGeom.rotateZ(Math.PI / 2);
    const discMesh = new THREE.Mesh(discGeom, materials.hot);
    discMesh.position.set(stageX, 0, 0);
    rotorGroup.add(discMesh);

    // High temperature turbine bucket blades
    const bladeGeom = new THREE.BoxGeometry(0.06, bladeHeight, 0.13);
    const bladeMesh = new THREE.InstancedMesh(bladeGeom, materials.hot, bladeCount);

    for (let b = 0; b < bladeCount; b++) {
      const angle = (b / bladeCount) * Math.PI * 2;
      const radialDist = discRadius + bladeHeight / 2;
      const y = Math.cos(angle) * radialDist;
      const z = Math.sin(angle) * radialDist;

      dummy.position.set(stageX, y, z);
      dummy.rotation.set(0, 0, 0);
      dummy.rotation.x = -angle;
      dummy.rotation.y = 0.55; // aggressive turbine extraction angle (~32 deg)
      dummy.updateMatrix();

      bladeMesh.setMatrixAt(b, dummy.matrix);
    }
    bladeMesh.instanceMatrix.needsUpdate = true;
    rotorGroup.add(bladeMesh);
  }

  root.add(rotorGroup);

  // 6. Stationary Components

  // 6A. Inlet Section (userData.componentId = 'inlet')
  const inletGroup = new THREE.Group();
  inletGroup.name = 'inlet';
  inletGroup.userData.componentId = 'inlet';

  // Bell-mouth intake duct (open cone)
  const bellGeom = new THREE.CylinderGeometry(1.42, 1.62, 0.9, 32, 1, true);
  bellGeom.rotateZ(Math.PI / 2);
  const bellMesh = new THREE.Mesh(bellGeom, materials.casing);
  bellMesh.position.set(-4.1, 0, 0);
  inletGroup.add(bellMesh);

  // Outer intake flange ring
  const intakeFlangeGeom = new THREE.TorusGeometry(1.63, 0.05, 16, 48);
  intakeFlangeGeom.rotateY(Math.PI / 2);
  const intakeFlange = new THREE.Mesh(intakeFlangeGeom, materials.casing);
  intakeFlange.position.set(-4.55, 0, 0);
  inletGroup.add(intakeFlange);

  // 8 Radial inlet struts supporting front bearing
  const strutGeom = new THREE.BoxGeometry(0.05, 1.15, 0.08);
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2;
    const strut = new THREE.Mesh(strutGeom, materials.steel);
    strut.position.set(-4.0, (Math.cos(angle) * 0.85), (Math.sin(angle) * 0.85));
    strut.rotation.x = -angle;
    inletGroup.add(strut);
  }

  // Front bearing stationary housing
  const fBearingGeom = new THREE.CylinderGeometry(0.35, 0.35, 0.6, 32);
  fBearingGeom.rotateZ(Math.PI / 2);
  const fBearing = new THREE.Mesh(fBearingGeom, materials.casing);
  fBearing.position.set(-3.95, 0, 0);
  inletGroup.add(fBearing);

  root.add(inletGroup);

  // 6B. Combustor Section (userData.componentId = 'combustor')
  const combustorGroup = new THREE.Group();
  combustorGroup.name = 'combustor';
  combustorGroup.userData.componentId = 'combustor';

  const combustorCansCount = 12;
  const canRadius = 0.17;
  const canLength = 1.05;
  const canRingRadius = 1.18;

  for (let c = 0; c < combustorCansCount; c++) {
    const angle = (c / combustorCansCount) * Math.PI * 2;
    const cy = Math.cos(angle) * canRingRadius;
    const cz = Math.sin(angle) * canRingRadius;

    // Combustion can body
    const canGeom = new THREE.CylinderGeometry(canRadius, canRadius * 0.9, canLength, 18);
    canGeom.rotateZ(Math.PI / 2);
    const canMesh = new THREE.Mesh(canGeom, materials.combustor);
    canMesh.position.set(0.05, cy, cz);

    // Tilt can slightly inward toward turbine inlet
    canMesh.rotation.z = Math.sin(angle) * 0.12;
    canMesh.rotation.y = -Math.cos(angle) * 0.12;

    // Fuel nozzle end cap
    const nozzleGeom = new THREE.CylinderGeometry(canRadius * 0.7, canRadius * 0.7, 0.14, 16);
    nozzleGeom.rotateZ(Math.PI / 2);
    const nozzle = new THREE.Mesh(nozzleGeom, materials.steel);
    nozzle.position.set(-0.52, cy, cz);

    // Liner clamp / cooling ring
    const clampGeom = new THREE.TorusGeometry(canRadius * 1.05, 0.02, 12, 24);
    clampGeom.rotateY(Math.PI / 2);
    const clamp = new THREE.Mesh(clampGeom, materials.steel);
    clamp.position.set(0.15, cy, cz);

    combustorGroup.add(canMesh, nozzle, clamp);
  }

  // Inner transition ducts ring
  const transGeom = new THREE.TorusGeometry(canRingRadius * 0.95, 0.14, 16, 36);
  transGeom.rotateY(Math.PI / 2);
  const transRing = new THREE.Mesh(transGeom, materials.combustor);
  transRing.position.set(0.55, 0, 0);
  combustorGroup.add(transRing);

  // Combustor internal warm glow point light
  const combustorGlow = new THREE.PointLight(0xff8a3d, 0.85, 4.5);
  combustorGlow.position.set(0.05, 0, 0);
  combustorGroup.add(combustorGlow);

  root.add(combustorGroup);

  // 6C. Turbine Stator Vane Nozzle Rings (userData.componentId = 'turbine')
  const turbineStatorGroup = new THREE.Group();
  turbineStatorGroup.name = 'turbine';
  turbineStatorGroup.userData.componentId = 'turbine';

  const statorXs = [0.70, 1.12, 1.58];
  statorXs.forEach((sx, idx) => {
    // Stationary nozzle diaphragm outer shroud
    const shroudGeom = new THREE.CylinderGeometry(1.48, 1.48, 0.07, 32, 1, true, 0, Math.PI * 1.35);
    shroudGeom.rotateZ(Math.PI / 2);
    const shroud = new THREE.Mesh(shroudGeom, materials.hot);
    shroud.position.set(sx, 0, 0);
    turbineStatorGroup.add(shroud);

    // Stator vanes (stationary)
    const vaneCount = 24;
    const vaneGeom = new THREE.BoxGeometry(0.04, 0.58 + idx * 0.08, 0.10);
    for (let v = 0; v < vaneCount; v++) {
      const angle = (v / vaneCount) * Math.PI * 2;
      // In cutaway area (top-front, ~0 to ~135 deg), leave open for clear interior viewing
      if (angle > 0.1 && angle < Math.PI * 0.72) continue;

      const r = 0.95 + idx * 0.05;
      const vane = new THREE.Mesh(vaneGeom, materials.hot);
      vane.position.set(sx, Math.cos(angle) * r, Math.sin(angle) * r);
      vane.rotation.x = -angle;
      vane.rotation.y = -0.45; // counter-angle for flow direction into rotor
      turbineStatorGroup.add(vane);
    }
  });

  root.add(turbineStatorGroup);

  // 6D. Exhaust Diffuser & Collector (userData.componentId = 'exhaust')
  const exhaustGroup = new THREE.Group();
  exhaustGroup.name = 'exhaust';
  exhaustGroup.userData.componentId = 'exhaust';

  // Aerodynamic expanding diffuser outer shell
  const diffuserGeom = new THREE.CylinderGeometry(1.58, 1.28, 1.8, 32, 1, true);
  diffuserGeom.rotateZ(Math.PI / 2); // points toward +X
  const diffuserMesh = new THREE.Mesh(diffuserGeom, materials.exhaust);
  diffuserMesh.position.set(2.8, 0, 0);
  exhaustGroup.add(diffuserMesh);

  // Inner center exhaust cone (bullet pointing to +X)
  const exhaustConeGeom = new THREE.ConeGeometry(0.52, 1.5, 32);
  exhaustConeGeom.rotateZ(-Math.PI / 2); // points +X
  const exhaustCone = new THREE.Mesh(exhaustConeGeom, materials.exhaust);
  exhaustCone.position.set(2.65, 0, 0);
  exhaustGroup.add(exhaustCone);

  // 4 Diffuser guide struts
  for (let i = 0; i < 4; i++) {
    const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const strutGeom = new THREE.BoxGeometry(0.08, 0.85, 0.18);
    const strut = new THREE.Mesh(strutGeom, materials.exhaust);
    strut.position.set(2.5, Math.cos(angle) * 0.8, Math.sin(angle) * 0.8);
    strut.rotation.x = -angle;
    exhaustGroup.add(strut);
  }

  // Rectangular exhaust plenum collar
  const collarGeom = new THREE.BoxGeometry(0.3, 3.1, 2.7);
  const collar = new THREE.Mesh(collarGeom, materials.exhaust);
  collar.position.set(3.75, 0, 0);
  exhaustGroup.add(collar);

  // Rear bearing casing support
  const rBearingGeom = new THREE.CylinderGeometry(0.36, 0.36, 0.5, 32);
  rBearingGeom.rotateZ(Math.PI / 2);
  const rBearing = new THREE.Mesh(rBearingGeom, materials.casing);
  rBearing.position.set(2.7, 0, 0);
  exhaustGroup.add(rBearing);

  root.add(exhaustGroup);

  // 6E. Cut-Away Outer Casing (userData.componentId = 'casing')
  // Upper-front quadrant is open (thetaStart = Math.PI * 0.2, thetaLength = Math.PI * 1.35)
  // so viewer can inspect compressor, combustor cans, and turbine
  const casingGroup = new THREE.Group();
  casingGroup.name = 'casing';
  casingGroup.userData.componentId = 'casing';

  const cutStart = Math.PI * 0.22;
  const cutLength = Math.PI * 1.35;

  // Compressor section casing (tapered)
  const compCasingGeom = new THREE.CylinderGeometry(1.32, 1.48, 2.9, 32, 1, true, cutStart, cutLength);
  compCasingGeom.rotateZ(Math.PI / 2);
  const compCasing = new THREE.Mesh(compCasingGeom, materials.casing);
  compCasing.position.set(-2.15, 0, 0);
  casingGroup.add(compCasing);

  // Combustor wrapper casing
  const combCasingGeom = new THREE.CylinderGeometry(1.58, 1.58, 1.3, 32, 1, true, cutStart, cutLength);
  combCasingGeom.rotateZ(Math.PI / 2);
  const combCasing = new THREE.Mesh(combCasingGeom, materials.casing);
  combCasing.position.set(0.05, 0, 0);
  casingGroup.add(combCasing);

  // Turbine section casing
  const turbCasingGeom = new THREE.CylinderGeometry(1.52, 1.38, 1.3, 32, 1, true, cutStart, cutLength);
  turbCasingGeom.rotateZ(Math.PI / 2);
  const turbCasing = new THREE.Mesh(turbCasingGeom, materials.casing);
  turbCasing.position.set(1.35, 0, 0);
  casingGroup.add(turbCasing);

  // Bolted casing flange rings along joints
  const flangePositions = [-3.6, -2.2, -0.65, 0.70, 2.0];
  const flangeRadii = [1.50, 1.44, 1.62, 1.62, 1.58];

  flangePositions.forEach((fx, idx) => {
    const fr = flangeRadii[idx];
    const flangeGeom = new THREE.TorusGeometry(fr, 0.045, 12, 36, cutLength);
    flangeGeom.rotateY(Math.PI / 2);
    flangeGeom.rotateX(cutStart);
    const flange = new THREE.Mesh(flangeGeom, materials.casing);
    flange.position.set(fx, 0, 0);
    casingGroup.add(flange);

    // Accent cutaway edge lip marker
    const edgeLip1 = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.08), materials.cutLip);
    edgeLip1.position.set(fx, Math.cos(cutStart) * fr, Math.sin(cutStart) * fr);
    const edgeLip2 = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.08), materials.cutLip);
    const endAngle = cutStart + cutLength;
    edgeLip2.position.set(fx, Math.cos(endAngle) * fr, Math.sin(endAngle) * fr);
    casingGroup.add(edgeLip1, edgeLip2);
  });

  root.add(casingGroup);

  // 7. Dynamic Airflow Arrows above turbine (-X to +X)
  const flowGroup = new THREE.Group();
  flowGroup.name = 'flowIndicators';
  const arrowPositions = [-2.8, -0.2, 2.4];
  arrowPositions.forEach((ax) => {
    const arrowGeom = new THREE.ConeGeometry(0.12, 0.35, 16);
    arrowGeom.rotateZ(-Math.PI / 2); // points +X
    const arrow = new THREE.Mesh(arrowGeom, materials.arrow);
    arrow.position.set(ax, 2.05, 0);
    flowGroup.add(arrow);

    const stemGeom = new THREE.CylinderGeometry(0.03, 0.03, 0.45, 12);
    stemGeom.rotateZ(Math.PI / 2);
    const stem = new THREE.Mesh(stemGeom, materials.arrow);
    stem.position.set(ax - 0.28, 2.05, 0);
    flowGroup.add(stem);
  });
  root.add(flowGroup);

  return {
    root,
    rotorGroup,
    combustorGlow,
    flowGroup,
    materials
  };
}
