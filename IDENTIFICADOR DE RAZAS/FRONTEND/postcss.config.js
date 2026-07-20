// Tailwind funciona como un plugin de PostCSS: lee las clases que usaste
// en el JSX y genera solo el CSS necesario. Autoprefixer agrega los prefijos
// de navegador (-webkit-, -moz-) automáticamente.
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
