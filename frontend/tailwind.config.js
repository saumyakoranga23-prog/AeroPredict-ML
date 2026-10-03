/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        panel: '0 0 0 1px rgba(148, 163, 184, 0.1), 0 12px 32px rgba(2, 6, 23, 0.6)',
      },
      colors: {
        slate: {
          950: '#020b16',
          925: '#071827',
          900: '#0a1628',
        },
      },
    },
  },
  plugins: [],
};
