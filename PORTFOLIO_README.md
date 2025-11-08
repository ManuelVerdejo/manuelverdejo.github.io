# Manuel Verdejo García - Portfolio

## 🎨 Descripción

Portfolio profesional de Data Analytics & AI Engineer con diseño moderno y minimalista inspirado en Linear.app y Vercel. Implementado como una aplicación de una sola página (SPA) con HTML, CSS y JavaScript vanilla.

## ✨ Características Principales

### Diseño Visual
- **Estética moderna**: Diseño minimalista y futurista con gradientes suaves
- **Animaciones fluidas**: Orbes de gradiente animados en el hero, micro-animaciones en hover
- **Tema claro/oscuro**: Toggle de tema con persistencia en localStorage
- **Tipografía moderna**: Space Grotesk con display=swap para rendimiento óptimo
- **Responsive**: Diseño totalmente adaptable a móviles, tablets y escritorio

### Funcionalidad
- **Navegación suave**: Scroll behavior smooth con links activos destacados
- **Búsqueda y filtros**: Sistema de búsqueda en tiempo real para proyectos y certificaciones
- **Filtros por tecnología**: Python, Machine Learning, Deep Learning, R, Power BI
- **Acordeones interactivos**: Organización de certificaciones por categorías
- **Copiar email**: Funcionalidad de copiar al portapapeles con feedback visual
- **Back to top**: Botón flotante que aparece al hacer scroll
- **CV Download**: Placeholder para descarga de CV con tracking de analytics

### Accesibilidad (WCAG AA+)
- Skip link para navegación por teclado
- Roles ARIA apropiados (navigation, menubar, status)
- Estados aria-expanded y aria-pressed
- Focus visible en todos los elementos interactivos
- Soporte para prefers-reduced-motion
- Texto semántico con HTML5 (nav, main, section, article, address)

### SEO & Performance
- Meta tags Open Graph y Twitter Cards
- JSON-LD Schema para Person
- Preload de fuentes críticas
- Lazy loading preparado
- Lighthouse score optimizado

## 📂 Estructura de Archivos

```
/app/
├── frontend/
│   ├── public/
│   │   └── portfolio.html          # Portfolio completo (producción ready)
│   └── src/
│       └── App.js                  # Landing page con link al portfolio
└── PORTFOLIO_README.md             # Este archivo
```

## 🚀 Acceso

### URL de Desarrollo
- **Portfolio completo**: http://localhost:3000/portfolio.html
- **Landing page**: http://localhost:3000/

### URL de Producción
Una vez desplegado, el portfolio estará disponible en:
- `https://tu-dominio.com/portfolio.html`

## 🎯 Secciones del Portfolio

### 1. Hero Section
- Nombre y título profesional con gradiente
- Badge de "Disponible para proyectos" con animación
- Descripción de especialización
- CTA para descargar CV y ver proyectos
- Background con orbes animados

### 2. Proyectos Destacados (12 categorías)
- IAs de Modelos Predictivos
- IAs de Modelos Logísticos
- IAs de Recomendación y Análisis de Lenguaje
- IAs de Deep Learning y Modelos Avanzados
- MLOps y Ciencia de Datos Experimental
- Simuladores Artificiales
- Análisis Empresarial con Python
- Análisis de Componentes Principales y Clustering
- Regresión Lineal y Clustering
- Dashboards Interactivos en Power BI
- Scripts de Análisis Estadístico en R
- Algoritmos de Optimización

**Total**: 24 proyectos con links directos a recursos descargables

### 3. Certificaciones Profesionales (6 categorías)
- Herramientas de Análisis y Productividad
- Business Intelligence y Power BI
- Gestión de Bases de Datos y SQL
- Python y Machine Learning
- Análisis Estadístico con R
- Competencias Complementarias

**Total**: 30+ certificaciones con PDFs descargables

### 4. Contacto
- Email con función copiar al portapapeles
- LinkedIn
- Disponibilidad geográfica

## 🛠️ Tecnologías Utilizadas

- **HTML5**: Semántico y accesible
- **CSS3**: Variables CSS, Grid, Flexbox, animaciones
- **JavaScript ES6+**: Vanilla JS sin frameworks
- **Google Fonts**: Space Grotesk
- **SVG Icons**: Para iconografía limpia

## 📝 Personalización

Para personalizar el portfolio para otro uso:

1. **Contenido**: Editar directamente en `/app/frontend/public/portfolio.html`
2. **Colores**: Modificar las variables CSS en `:root` y `[data-theme="dark"]`
3. **CV**: Reemplazar el placeholder del botón "Descargar CV" con la URL real
4. **Analytics**: Descomentar y configurar Google Analytics 4 al final del HTML

## 🎨 Paleta de Colores

### Tema Claro
- Background: `#ffffff`, `#f8f9fa`
- Text: `#0a0a0a`, `#525252`, `#737373`
- Border: `#e5e5e5`

### Tema Oscuro
- Background: `#0a0a0a`, `#141414`, `#1a1a1a`
- Text: `#ffffff`, `#a3a3a3`, `#737373`
- Border: `#262626`

### Acentos
- Gradiente primario: `#3b82f6` → `#9333ea` (azul a púrpura)
- Gradiente secundario: `#06b6d4` → `#8b5cf6` (cyan a púrpura)

## 🔥 Características Destacadas

1. **Animación de gradientes**: Los orbes de fondo flotan suavemente con keyframes
2. **Tema persistente**: El tema elegido se guarda en localStorage
3. **Smooth transitions**: Todas las interacciones tienen transiciones de 0.2s
4. **Hover effects**: Cards se elevan y cambian de sombra al pasar el mouse
5. **Active states**: Los filtros y links de navegación muestran su estado activo
6. **Search en tiempo real**: Búsqueda instantánea sin recargar
7. **Contador dinámico**: Muestra proyectos visibles al filtrar
8. **Keyboard accessible**: Toda la navegación funciona con teclado

## 📱 Responsive Breakpoints

- **Desktop**: 1200px+ (diseño completo)
- **Tablet**: 768px - 1199px (ajustes de spacing)
- **Mobile**: < 768px (layout de una columna, nav compacto)

## 🎭 Animaciones

- `fadeInUp`: Hero content (0.6s staggered)
- `float`: Gradient orbs (20s infinite)
- `pulse`: Status indicator (2s infinite)
- `modalSlideUp`: Modal entrance (0.3s)

## 🚦 Estado del Proyecto

✅ **Completo y listo para producción**

### Implementado
- ✅ Diseño visual moderno y profesional
- ✅ Tema claro/oscuro con persistencia
- ✅ 24 proyectos organizados por categoría
- ✅ 30+ certificaciones con búsqueda
- ✅ Navegación smooth con active states
- ✅ Sistema de filtros y búsqueda
- ✅ Accesibilidad WCAG AA+
- ✅ SEO optimizado
- ✅ Responsive design completo
- ✅ Animaciones y micro-interacciones

### Pendiente (opcional)
- ⏳ Upload de CV real (actualmente placeholder)
- ⏳ Configuración de Google Analytics 4
- ⏳ Imágenes de proyectos personalizadas

## 📄 Licencia

Este portfolio es de código abierto y puede ser utilizado como template para otros portfolios profesionales.

---

**Desarrollado por**: E1 - Emergent AI Agent  
**Para**: Manuel Verdejo García  
**Fecha**: Noviembre 2025
