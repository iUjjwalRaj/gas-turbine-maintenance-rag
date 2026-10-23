/**
 * Turbine Component Metadata Table
 * Visual presentation metadata for Gas Turbine Explorer.
 * Strictly neutral descriptive labels; no intervals, numbers, or prices.
 */

export const COMPONENTS = {
  inlet: {
    id: "inlet",
    label: "Air Inlet System",
    description: "Aerodynamic bell-mouth intake casing and radial struts guiding ambient air into the axial compressor.",
    suggestedQuestion: "What components are inspected during an inlet inspection?"
  },
  compressor: {
    id: "compressor",
    label: "Axial Compressor",
    description: "Multi-stage bladed rotor and stator assembly that compresses incoming air to high pressure prior to combustion.",
    suggestedQuestion: "What causes compressor blade degradation?"
  },
  combustor: {
    id: "combustor",
    label: "Combustion System",
    description: "Can-annular combustion chamber array where fuel is atomized, mixed with compressed air, and continuously burned.",
    suggestedQuestion: "What does a combustion inspection include?"
  },
  turbine: {
    id: "turbine",
    label: "Turbine (Hot Gas Path)",
    description: "High-temperature nozzle guide vanes and rotating bucket wheels that expand hot combustion gases to produce shaft work.",
    suggestedQuestion: "What components are examined during a hot gas path inspection?"
  },
  exhaust: {
    id: "exhaust",
    label: "Exhaust Diffuser",
    description: "Annular diffuser duct and collector frame that decelerates high-velocity exhaust gas and recovers static pressure.",
    suggestedQuestion: "What is the purpose of the exhaust diffuser?"
  },
  rotor: {
    id: "rotor",
    label: "Rotor & Bearings",
    description: "Central rotating shaft assembly supported by hydrodynamic journal and thrust bearings transmitting mechanical power.",
    suggestedQuestion: "What are the key considerations for rotor maintenance?"
  }
};
