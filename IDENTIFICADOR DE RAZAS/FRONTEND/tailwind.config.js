/** @type {import('tailwindcss').Config} */

// Sistema de diseño del proyecto.
// Todo color, fuente y animación se declara AQUÍ y Tailwind genera las clases:
//   pino  ->  bg-pino, text-pino, border-pino
// En el JSX no debe aparecer ningún #hex suelto.
export default {
  // content le dice a Tailwind dónde buscar clases. Si un archivo no está en
  // esta lista, sus estilos NO se generan y el diseño sale roto en producción.
  content: ["./index.html", "./fuente/**/*.{js,jsx}"],

  theme: {
    extend: {
      colors: {
        tinta: "#14201f",    // texto principal, casi negro con matiz verde
        grafito: "#5b6b67",  // texto secundario
        papel: "#eef1ef",    // fondo de la página
        lienzo: "#ffffff",   // fondo de las tarjetas
        niebla: "#dbe2df",   // bordes y separadores
        pino: "#1b5e4b",     // color de marca: botones y acentos
        ambar: "#c97b2a",    // exclusivo del dato de confianza
      },

      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        dato: ["JetBrains Mono", "monospace"],
      },

      keyframes: {
        // Entrada suave de las tarjetas de resultado.
        aparecer: {
          from: { opacity: "0", transform: "translateY(12px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        // Elemento distintivo: la línea que recorre la foto mientras el
        // modelo la analiza.
        barrido: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(1000%)" },
        },
      },

      animation: {
        aparecer: "aparecer 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) both",
        barrido: "barrido 1.6s ease-in-out infinite",
      },
    },
  },

  plugins: [],
};
