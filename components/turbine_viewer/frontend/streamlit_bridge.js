/**
 * Streamlit Custom Component Bridge
 * Handles postMessage communication between the embedded iframe and the parent Streamlit app.
 */

const send = (type, data = {}) => {
  try {
    if (window.parent && window.parent !== window) {
      window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
    }
  } catch (err) {
    console.warn("Streamlit postMessage notification error:", err);
  }
};

export function ready() {
  send("streamlit:componentReady", { apiVersion: 1 });
}

export function setHeight(h) {
  send("streamlit:setFrameHeight", { height: h });
}

export function setValue(v) {
  send("streamlit:setComponentValue", { value: v, dataType: "json" });
}

export function onRender(cb) {
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "streamlit:render") {
      cb(e.data.args || {}, e.data.disabled, e.data.theme);
    }
  });
}
