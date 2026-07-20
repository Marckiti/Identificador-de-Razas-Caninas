/**
 * principal.jsx
 * -------------
 * Punto de entrada del frontend. Solo hace una cosa: montar el componente
 * Aplicacion dentro del <div id="raiz"> del index.html.
 */
import React from "react";
import ReactDOM from "react-dom/client";
import Aplicacion from "./aplicacion.jsx";
import "./estilos.css";

ReactDOM.createRoot(document.getElementById("raiz")).render(
  <React.StrictMode>
    <Aplicacion />
  </React.StrictMode>
);
