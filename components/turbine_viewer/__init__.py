"""Turbine Viewer Streamlit Custom Component.

Provides a custom Streamlit component embedding the standalone Three.js
industrial gas turbine viewer with OrbitControls, smooth rotation, and camera reset.
"""

from __future__ import annotations

import os
from typing import Any, Optional
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Component path points directly to the frontend static directory
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component_func = None


def _get_component_func():
    global _component_func
    if _component_func is None:
        _component_func = components.declare_component(
            "turbine_viewer",
            path=_FRONTEND_DIR,
        )
    else:
        # If declared prior to ScriptRunContext being ready, register it with current runtime
        try:
            from streamlit.runtime import get_instance
            inst = get_instance()
            if inst and hasattr(inst, "component_registry"):
                inst.component_registry.register_component(_component_func)
        except Exception:
            pass
    return _component_func


def turbine_viewer(
    selected_id: Optional[str] = None,
    height: int = 950,
    auto_rotate: bool = True,
    key: Optional[str] = "turbine_viewer",
) -> Optional[dict[str, Any]]:
    """Renders the 3D Gas Turbine Viewer Streamlit component.

    Args:
        selected_id: Optional component ID to focus/highlight.
        height: Height of the component frame in pixels (default 950).
        auto_rotate: Whether idle camera auto-rotation is initially active.
        key: Streamlit component key for state persistence across reruns.

    Returns:
        Event dictionary or None.
    """
    comp = _get_component_func()
    # Also ensure registration if ctx is available
    ctx = get_script_run_ctx()
    if ctx is not None:
        try:
            from streamlit.runtime import get_instance
            inst = get_instance()
            if inst and hasattr(inst, "component_registry"):
                inst.component_registry.register_component(comp)
        except Exception:
            pass

    return comp(
        selected_id=selected_id,
        height=height,
        auto_rotate=auto_rotate,
        key=key,
        default=None,
    )
