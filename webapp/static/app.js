/* ═══════════════════════════════════════════════════════════════
   GeoRock AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initUpload();
    initTabs();
    initFilters();
    loadStatus();
    loadGallery();
});

// ─── Navbar ─────────────────────────────────────────────────────
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const burger = document.getElementById('navBurger');
    const links = document.querySelector('.nav-links');
    const navLinks = document.querySelectorAll('.nav-link');

    // Scroll effect
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 40);
    });

    // Burger menu
    if (burger) {
        burger.addEventListener('click', () => {
            links.classList.toggle('open');
        });
    }

    // Active link tracking
    const sections = document.querySelectorAll('section, .hero');
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.dataset.section === id);
                });
            }
        });
    }, { threshold: 0.3 });

    sections.forEach(s => observer.observe(s));

    // Close mobile menu on link click
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            links.classList.remove('open');
        });
    });
}

// ─── Upload & Classification ────────────────────────────────────
function initUpload() {
    const zone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const content = document.getElementById('uploadContent');
    const preview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImg');
    const clearBtn = document.getElementById('clearBtn');
    const classifyBtn = document.getElementById('classifyBtn');

    let selectedFile = null;

    // Click to upload
    zone.addEventListener('click', (e) => {
        if (e.target === clearBtn || clearBtn.contains(e.target)) return;
        fileInput.click();
    });

    // Drag & Drop
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            content.style.display = 'none';
            preview.style.display = 'flex';
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        content.style.display = 'flex';
        preview.style.display = 'none';
        classifyBtn.disabled = true;

        // Reset results
        document.getElementById('resultsContent').style.display = 'none';
        document.getElementById('resultsLoading').style.display = 'none';
        document.getElementById('resultsPlaceholder').style.display = 'flex';
    }

    // Classify button
    classifyBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        classifyImage(selectedFile);
    });
}

async function classifyImage(file) {
    const placeholder = document.getElementById('resultsPlaceholder');
    const loading = document.getElementById('resultsLoading');
    const content = document.getElementById('resultsContent');

    placeholder.style.display = 'none';
    content.style.display = 'none';
    loading.style.display = 'flex';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            alert('Erreur: ' + data.error);
            loading.style.display = 'none';
            placeholder.style.display = 'flex';
            return;
        }

        displayResults(data);
    } catch (err) {
        alert('Erreur de connexion au serveur.');
        console.error(err);
        loading.style.display = 'none';
        placeholder.style.display = 'flex';
    }
}

function displayResults(data) {
    const loading = document.getElementById('resultsLoading');
    const content = document.getElementById('resultsContent');

    // Prediction Class
    document.getElementById('predClass').textContent = data.prediction;

    // Subtypes
    const subtypesEl = document.getElementById('predSubtypes');
    subtypesEl.innerHTML = '';
    (data.subtypes || []).forEach(st => {
        const tag = document.createElement('span');
        tag.className = 'subtype-tag';
        tag.textContent = st;
        subtypesEl.appendChild(tag);
    });

    // Confidence Ring
    const conf = data.confidence;
    const circumference = 2 * Math.PI * 42; // r = 42
    const offset = circumference - (conf / 100) * circumference;

    // We need an SVG gradient for the ring
    const ringFill = document.getElementById('ringFill');
    const confRing = document.getElementById('confRing');

    // Add gradient definition if not exists
    const svg = confRing.querySelector('svg');
    if (!svg.querySelector('defs')) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        grad.setAttribute('id', 'ringGradient');
        grad.innerHTML = `
            <stop offset="0%" stop-color="#6c5ce7"/>
            <stop offset="50%" stop-color="#a855f7"/>
            <stop offset="100%" stop-color="#ec4899"/>
        `;
        defs.appendChild(grad);
        svg.insertBefore(defs, svg.firstChild);
    }

    ringFill.style.stroke = 'url(#ringGradient)';
    ringFill.style.strokeDasharray = circumference;

    // Animate after a small delay
    setTimeout(() => {
        ringFill.style.strokeDashoffset = offset;
    }, 100);

    document.getElementById('confValue').textContent = conf.toFixed(1) + '%';

    // Probability Bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';

    const classColors = {
        'Igneous': 'igneous',
        'Metamorphic': 'metamorphic',
        'Sedimentary': 'sedimentary'
    };

    Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cls, prob]) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <span class="prob-label">${cls}</span>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill ${classColors[cls]}" style="width: 0%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            `;
            probBars.appendChild(row);

            // Animate bar width
            setTimeout(() => {
                row.querySelector('.prob-bar-fill').style.width = prob + '%';
            }, 200);
        });

    // Grad-CAM image
    document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam;

    // Show results
    loading.style.display = 'none';
    content.style.display = 'block';
    content.style.animation = 'fadeInUp 0.5s var(--ease-out)';
}

// ─── Tabs ───────────────────────────────────────────────────────
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update panels
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.getElementById('panel-' + tabId).classList.add('active');
        });
    });
}

// ─── Gallery Filters ────────────────────────────────────────────
function initFilters() {
    const filterBtns = document.querySelectorAll('.filter-btn');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.dataset.filter;
            const cards = document.querySelectorAll('.gallery-card');

            cards.forEach((card, i) => {
                const show = filter === 'all' || card.dataset.class === filter;
                card.style.display = show ? 'block' : 'none';
                if (show) {
                    card.style.animation = `fadeInUp 0.4s var(--ease-out) ${i * 0.05}s both`;
                }
            });
        });
    });
}

// ─── Load Status & Training Results ─────────────────────────────
async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Training images
        const imageMap = {
            'training_curves.png': 'img-training',
            'confusion_matrix.png': 'img-confusion',
            'classification_report.png': 'img-report',
            'per_class_accuracy.png': 'img-accuracy'
        };

        const nodataMap = {
            'training_curves.png': 'nodata-training',
            'confusion_matrix.png': 'nodata-confusion',
            'classification_report.png': 'nodata-report',
            'per_class_accuracy.png': 'nodata-accuracy'
        };

        for (const [filename, imgId] of Object.entries(imageMap)) {
            const imgEl = document.getElementById(imgId);
            const nodataEl = document.getElementById(nodataMap[filename]);

            if (data.training_images.includes(filename)) {
                imgEl.src = '/api/training-image/' + filename;
                imgEl.style.display = 'block';
                if (nodataEl) nodataEl.style.display = 'none';
            } else {
                imgEl.style.display = 'none';
                if (nodataEl) nodataEl.style.display = 'block';
            }
        }

        // Classification report text
        if (data.classification_report) {
            const reportWrap = document.getElementById('reportTextWrap');
            const reportText = document.getElementById('reportText');
            reportText.textContent = data.classification_report;
            reportWrap.style.display = 'block';

            // Try to extract accuracy for hero stat
            const accMatch = data.classification_report.match(/accuracy\s+(\d+\.\d+)/);
            if (accMatch) {
                document.getElementById('accuracyStat').textContent = (parseFloat(accMatch[1]) * 100).toFixed(0) + '%';
            }
        }

        // EDA images
        const edaGallery = document.getElementById('edaGallery');
        const nodataEda = document.getElementById('nodata-eda');

        if (data.eda_images.length > 0) {
            edaGallery.innerHTML = '';
            data.eda_images.forEach(filename => {
                const wrap = document.createElement('div');
                wrap.className = 'result-image-wrap';
                wrap.innerHTML = `<img src="/api/eda-image/${filename}" alt="${filename}" class="result-img">`;
                edaGallery.appendChild(wrap);
            });
            nodataEda.style.display = 'none';
        } else {
            edaGallery.innerHTML = '';
            nodataEda.style.display = 'block';
        }

    } catch (err) {
        console.error('Failed to load status:', err);
    }
}

// ─── Load Gallery ───────────────────────────────────────────────
async function loadGallery() {
    const grid = document.getElementById('galleryGrid');

    try {
        const res = await fetch('/api/sample-images');
        const samples = await res.json();

        grid.innerHTML = '';

        if (samples.length === 0) {
            grid.innerHTML = '<div class="gallery-loading"><p>Aucune image trouvée dans le dataset.</p></div>';
            return;
        }

        samples.forEach((sample, i) => {
            const card = document.createElement('div');
            card.className = 'gallery-card';
            card.dataset.class = sample.class;
            card.style.animationDelay = `${i * 0.06}s`;

            const colorClass = sample.class.toLowerCase();

            card.innerHTML = `
                <img class="gallery-card-img" src="data:image/png;base64,${sample.image}" alt="${sample.subtype}">
                <div class="gallery-card-info">
                    <span class="gallery-class-badge ${colorClass}">${sample.class}</span>
                    <div class="gallery-subtype">${sample.subtype}</div>
                </div>
            `;

            // Click to classify
            card.addEventListener('click', () => {
                // Convert base64 to file and trigger classification
                fetch(`data:image/png;base64,${sample.image}`)
                    .then(r => r.blob())
                    .then(blob => {
                        const file = new File([blob], sample.filename, { type: 'image/png' });

                        // Set preview
                        const previewImg = document.getElementById('previewImg');
                        const content = document.getElementById('uploadContent');
                        const preview = document.getElementById('uploadPreview');
                        const classifyBtn = document.getElementById('classifyBtn');

                        previewImg.src = `data:image/png;base64,${sample.image}`;
                        content.style.display = 'none';
                        preview.style.display = 'flex';
                        classifyBtn.disabled = false;

                        // Scroll to classify section
                        document.getElementById('classify').scrollIntoView({ behavior: 'smooth' });

                        // Auto classify
                        setTimeout(() => classifyImage(file), 600);
                    });
            });

            grid.appendChild(card);
        });

    } catch (err) {
        console.error('Failed to load gallery:', err);
        grid.innerHTML = '<div class="gallery-loading"><p>Erreur de chargement de la galerie.</p></div>';
    }
}
