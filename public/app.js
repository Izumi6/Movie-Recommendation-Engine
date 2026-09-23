// CineMatch - Client-Side Engine for Vercel Deployment
// Zero emojis, 100% responsive, high-performance hybrid recommendation logic

let movies = [];
let moviesById = {};
let contentSimilar = {};
let svdModel = {};
let selectedSeeds = [];

// Initialize application
async function initApp() {
  try {
    const [moviesRes, simRes, svdRes] = await Promise.all([
      fetch('data/movies.json').then(r => r.json()),
      fetch('data/content_similar.json').then(r => r.json()),
      fetch('data/svd_model.json').then(r => r.json())
    ]);

    movies = moviesRes;
    contentSimilar = simRes;
    svdModel = svdRes;

    movies.forEach(m => {
      moviesById[m.id] = m;
    });

    populateGenreFilter();
    setupNavigation();
    setupSearchAndFilters();
    setupSimilarPage();
    setupPersonalizedPage();
    renderHomeMovies(movies.slice(0, 32));
    initAnalytics();

    // Default seed movies for Personalized Picks
    addSeedMovie(1, 5.0); // Toy Story
    addSeedMovie(2571, 4.5); // Matrix, The
  } catch (err) {
    console.error('Failed to load application data:', err);
  }
}

// Navigation Tabs
function setupNavigation() {
  const buttons = document.querySelectorAll('.nav-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const target = btn.dataset.tab;
      document.querySelectorAll('.tab-section').forEach(sec => {
        sec.classList.remove('active');
      });
      const activeSec = document.getElementById(target);
      if (activeSec) {
        activeSec.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });
}

// Genre Colors helper
function getGenreColor(genre) {
  const colorMap = {
    'Action': '#ef4444',
    'Adventure': '#f97316',
    'Animation': '#eab308',
    'Comedy': '#22c55e',
    'Crime': '#06b6d4',
    'Documentary': '#64748b',
    'Drama': '#3b82f6',
    'Fantasy': '#a855f7',
    'Horror': '#ec4899',
    'Romance': '#f43f5e',
    'Sci-Fi': '#8b5cf6',
    'Thriller': '#14b8a6',
    'Mystery': '#6366f1'
  };
  return colorMap[genre] || '#94a3b8';
}

// Populate Genre filter dropdown
function populateGenreFilter() {
  const select = document.getElementById('genre-filter');
  if (!select) return;
  const genres = new Set();
  movies.forEach(m => m.genres.forEach(g => genres.add(g)));
  Array.from(genres).sort().forEach(g => {
    const opt = document.createElement('option');
    opt.value = g;
    opt.textContent = g;
    select.appendChild(opt);
  });
}

// Setup Search & Filters on Home
function setupSearchAndFilters() {
  const searchInput = document.getElementById('home-search');
  const genreSelect = document.getElementById('genre-filter');

  function filterMovies() {
    const query = (searchInput?.value || '').toLowerCase().trim();
    const genre = genreSelect?.value || 'ALL';

    const filtered = movies.filter(m => {
      const matchQuery = !query || m.clean_title.toLowerCase().includes(query) || m.tags.toLowerCase().includes(query);
      const matchGenre = genre === 'ALL' || m.genres.includes(genre);
      return matchQuery && matchGenre;
    });

    renderHomeMovies(filtered.slice(0, 36));
  }

  searchInput?.addEventListener('input', filterMovies);
  genreSelect?.addEventListener('change', filterMovies);
}

// Render Movies Grid
function renderHomeMovies(movieList) {
  const grid = document.getElementById('home-movies-grid');
  if (!grid) return;
  grid.innerHTML = '';

  if (movieList.length === 0) {
    grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 3rem;">No movies matched your search criteria.</div>';
    return;
  }

  movieList.forEach(m => {
    const card = document.createElement('div');
    card.className = 'movie-card';

    const badgesHtml = m.genres.slice(0, 3).map(g => {
      const color = getGenreColor(g);
      return `<span class="badge" style="border-left: 2px solid ${color};">${g}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="movie-card-header">
        <div class="movie-title">${escapeHtml(m.clean_title)}</div>
        <div class="movie-year">${m.year ? m.year : 'Classic'}</div>
      </div>
      <div class="movie-badges">${badgesHtml}</div>
      <div class="movie-footer">
        <span class="rating-badge">&#9733; ${m.rating > 0 ? m.rating.toFixed(1) : 'N/A'}</span>
        <span class="votes-count">${m.votes} ratings</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

// Find Similar Movies Setup
function setupSimilarPage() {
  const input = document.getElementById('similar-search');
  const dropdown = document.getElementById('similar-dropdown');
  if (!input || !dropdown) return;

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    if (q.length < 2) {
      dropdown.style.display = 'none';
      return;
    }
    const matches = movies.filter(m => m.clean_title.toLowerCase().includes(q)).slice(0, 8);
    dropdown.innerHTML = '';
    if (matches.length > 0) {
      dropdown.style.display = 'block';
      matches.forEach(m => {
        const item = document.createElement('div');
        item.style.padding = '0.75rem 1rem';
        item.style.cursor = 'pointer';
        item.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
        item.textContent = `${m.clean_title} (${m.year || 'N/A'})`;
        item.addEventListener('mouseenter', () => item.style.background = 'rgba(99,102,241,0.2)');
        item.addEventListener('mouseleave', () => item.style.background = 'transparent');
        item.addEventListener('click', () => {
          input.value = `${m.clean_title} (${m.year || 'N/A'})`;
          dropdown.style.display = 'none';
          showSimilarResults(m.id);
        });
        dropdown.appendChild(item);
      });
    } else {
      dropdown.style.display = 'none';
    }
  });

  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.style.display = 'none';
    }
  });

  // Default to The Matrix
  showSimilarResults(2571);
}

function showSimilarResults(movieId) {
  const target = moviesById[movieId];
  const targetContainer = document.getElementById('similar-target-movie');
  const resultsGrid = document.getElementById('similar-results-grid');
  if (!target || !targetContainer || !resultsGrid) return;

  targetContainer.innerHTML = `
    <div style="background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
      <h3 style="font-size: 1.3rem; margin-bottom: 0.4rem; color: #fff;">${escapeHtml(target.clean_title)} (${target.year || 'N/A'})</h3>
      <div style="color: var(--text-secondary); margin-bottom: 0.8rem; font-size: 0.9rem;">
        Genres: ${target.genres.join(', ') || 'General'} | Rating: &#9733; ${target.rating.toFixed(1)} (${target.votes} votes)
      </div>
      <div style="font-size: 0.85rem; color: var(--text-muted);">
        Tags: ${target.tags || 'General themes, cinema classic'}
      </div>
    </div>
  `;

  const pairs = contentSimilar[movieId] || [];
  resultsGrid.innerHTML = '';

  if (pairs.length === 0) {
    resultsGrid.innerHTML = '<div style="grid-column: 1/-1; color: var(--text-muted); text-align: center; padding: 2rem;">No close content similarity matches found.</div>';
    return;
  }

  pairs.forEach(([candId, simScore]) => {
    const cand = moviesById[candId];
    if (!cand) return;

    const card = document.createElement('div');
    card.className = 'movie-card';

    const badgesHtml = cand.genres.slice(0, 3).map(g => {
      const color = getGenreColor(g);
      return `<span class="badge" style="border-left: 2px solid ${color};">${g}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="movie-card-header">
        <div class="movie-title">${escapeHtml(cand.clean_title)}</div>
        <div class="movie-year">${cand.year ? cand.year : 'Classic'}</div>
      </div>
      <div class="movie-badges">${badgesHtml}</div>
      <div class="movie-footer">
        <span class="rating-badge">&#9733; ${cand.rating > 0 ? cand.rating.toFixed(1) : 'N/A'}</span>
        <span class="match-score">${Math.round(simScore * 100)}% Match</span>
      </div>
    `;
    resultsGrid.appendChild(card);
  });
}

// Personalized Picks Setup (Hybrid Recommender)
function setupPersonalizedPage() {
  const input = document.getElementById('personalized-search');
  const dropdown = document.getElementById('personalized-dropdown');
  const alphaSlider = document.getElementById('alpha-slider');
  const alphaVal = document.getElementById('alpha-val');
  const getRecBtn = document.getElementById('get-recommendations-btn');

  if (alphaSlider && alphaVal) {
    alphaSlider.addEventListener('input', () => {
      alphaVal.textContent = parseFloat(alphaSlider.value).toFixed(2);
    });
  }

  if (input && dropdown) {
    input.addEventListener('input', () => {
      const q = input.value.toLowerCase().trim();
      if (q.length < 2) {
        dropdown.style.display = 'none';
        return;
      }
      const matches = movies.filter(m => m.clean_title.toLowerCase().includes(q) && !selectedSeeds.some(s => s.id === m.id)).slice(0, 6);
      dropdown.innerHTML = '';
      if (matches.length > 0) {
        dropdown.style.display = 'block';
        matches.forEach(m => {
          const item = document.createElement('div');
          item.style.padding = '0.75rem 1rem';
          item.style.cursor = 'pointer';
          item.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
          item.textContent = `${m.clean_title} (${m.year || 'N/A'})`;
          item.addEventListener('mouseenter', () => item.style.background = 'rgba(99,102,241,0.2)');
          item.addEventListener('mouseleave', () => item.style.background = 'transparent');
          item.addEventListener('click', () => {
            addSeedMovie(m.id, 4.5);
            input.value = '';
            dropdown.style.display = 'none';
          });
          dropdown.appendChild(item);
        });
      } else {
        dropdown.style.display = 'none';
      }
    });

    document.addEventListener('click', (e) => {
      if (!input.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.style.display = 'none';
      }
    });
  }

  getRecBtn?.addEventListener('click', computeHybridRecommendations);
}

function addSeedMovie(movieId, initialRating = 4.0) {
  const m = moviesById[movieId];
  if (!m || selectedSeeds.some(s => s.id === movieId)) return;
  selectedSeeds.push({ id: movieId, rating: initialRating });
  renderSeedList();
}

function removeSeedMovie(movieId) {
  selectedSeeds = selectedSeeds.filter(s => s.id !== movieId);
  renderSeedList();
}

function renderSeedList() {
  const container = document.getElementById('seed-list-container');
  if (!container) return;
  container.innerHTML = '';

  if (selectedSeeds.length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No seed movies selected. Search above to add titles you enjoy.</div>';
    return;
  }

  selectedSeeds.forEach(seed => {
    const m = moviesById[seed.id];
    if (!m) return;

    const row = document.createElement('div');
    row.className = 'seed-item';
    row.innerHTML = `
      <div class="seed-title">${escapeHtml(m.clean_title)} (${m.year || 'N/A'})</div>
      <div class="seed-controls">
        <label style="font-size: 0.85rem; color: var(--text-secondary);">Your Rating: </label>
        <select class="genre-select" style="padding: 0.35rem 0.6rem; font-size: 0.85rem;" onchange="updateSeedRating(${seed.id}, this.value)">
          <option value="5.0" ${seed.rating === 5.0 ? 'selected' : ''}>5.0 Stars</option>
          <option value="4.5" ${seed.rating === 4.5 ? 'selected' : ''}>4.5 Stars</option>
          <option value="4.0" ${seed.rating === 4.0 ? 'selected' : ''}>4.0 Stars</option>
          <option value="3.5" ${seed.rating === 3.5 ? 'selected' : ''}>3.5 Stars</option>
          <option value="3.0" ${seed.rating === 3.0 ? 'selected' : ''}>3.0 Stars</option>
        </select>
        <button class="seed-remove" onclick="removeSeedMovie(${seed.id})">&#10005;</button>
      </div>
    `;
    container.appendChild(row);
  });
}

window.updateSeedRating = function(movieId, ratingVal) {
  const seed = selectedSeeds.find(s => s.id === movieId);
  if (seed) seed.rating = parseFloat(ratingVal);
};

window.removeSeedMovie = removeSeedMovie;

// Compute Hybrid Recommendations using SVD Item Factors + Content Similarity
function computeHybridRecommendations() {
  const resultsGrid = document.getElementById('personalized-results-grid');
  if (!resultsGrid) return;

  if (selectedSeeds.length === 0) {
    resultsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">Please add at least one seed movie to generate recommendations.</div>';
    return;
  }

  const alpha = parseFloat(document.getElementById('alpha-slider')?.value || '0.5');
  const seedIds = new Set(selectedSeeds.map(s => s.id));

  // 1. Content Scores
  const contentScores = {};
  selectedSeeds.forEach(seed => {
    const pairs = contentSimilar[seed.id] || [];
    pairs.forEach(([candId, sim]) => {
      if (!seedIds.has(candId)) {
        contentScores[candId] = (contentScores[candId] || 0) + sim * (seed.rating / 5.0);
      }
    });
  });

  // 2. Collaborative Scores via SVD item factors
  const collabScores = {};
  if (svdModel && svdModel.item_factors) {
    const k = svdModel.k;
    const movieIdToSvdIdx = {};
    svdModel.movie_ids.forEach((id, idx) => {
      movieIdToSvdIdx[id] = idx;
    });

    // Approximate user latent profile: p_u = sum(r_i * v_i)
    const userLatent = new Float32Array(k);
    let totalWeight = 0;
    selectedSeeds.forEach(seed => {
      const idx = movieIdToSvdIdx[seed.id];
      if (idx !== undefined) {
        const factors = svdModel.item_factors[idx];
        const weight = seed.rating;
        totalWeight += weight;
        for (let j = 0; j < k; j++) {
          userLatent[j] += factors[j] * weight;
        }
      }
    });

    if (totalWeight > 0) {
      for (let j = 0; j < k; j++) {
        userLatent[j] /= totalWeight;
      }

      // Dot product with all movie factors
      svdModel.movie_ids.forEach((mId, idx) => {
        if (!seedIds.has(mId)) {
          const factors = svdModel.item_factors[idx];
          let dot = 0;
          for (let j = 0; j < k; j++) {
            dot += userLatent[j] * factors[j];
          }
          collabScores[mId] = dot;
        }
      });
    }
  }

  // 3. Normalization and Hybrid Blending
  const candidateIds = new Set([...Object.keys(contentScores).map(Number), ...Object.keys(collabScores).map(Number)]);
  
  // Find max values for normalization
  let maxContent = 0.0001;
  Object.values(contentScores).forEach(s => { if (s > maxContent) maxContent = s; });

  let minCollab = 999, maxCollab = -999;
  Object.values(collabScores).forEach(s => {
    if (s < minCollab) minCollab = s;
    if (s > maxCollab) maxCollab = s;
  });
  const collabRange = (maxCollab - minCollab) || 1.0;

  const hybridList = [];
  candidateIds.forEach(id => {
    const cand = moviesById[id];
    if (!cand) return;

    const normContent = (contentScores[id] || 0) / maxContent;
    const normCollab = collabScores[id] !== undefined ? (collabScores[id] - minCollab) / collabRange : 0.5;

    const finalScore = alpha * normContent + (1.0 - alpha) * normCollab;
    hybridList.push({
      movie: cand,
      score: finalScore,
      contentScore: normContent,
      collabScore: normCollab
    });
  });

  hybridList.sort((a, b) => b.score - a.score);
  const topRecommendations = hybridList.slice(0, 16);

  resultsGrid.innerHTML = '';
  topRecommendations.forEach((item, idx) => {
    const m = item.movie;
    const card = document.createElement('div');
    card.className = 'movie-card';

    const badgesHtml = m.genres.slice(0, 3).map(g => {
      const color = getGenreColor(g);
      return `<span class="badge" style="border-left: 2px solid ${color};">${g}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="movie-card-header">
        <div style="font-size: 0.75rem; color: var(--accent-primary); font-weight: 700; margin-bottom: 0.2rem;">RANK #${idx + 1}</div>
        <div class="movie-title">${escapeHtml(m.clean_title)}</div>
        <div class="movie-year">${m.year ? m.year : 'Classic'}</div>
      </div>
      <div class="movie-badges">${badgesHtml}</div>
      <div class="movie-footer">
        <span class="rating-badge">&#9733; ${m.rating > 0 ? m.rating.toFixed(1) : 'N/A'}</span>
        <span class="match-score">${Math.round(item.score * 100)}% Match</span>
      </div>
    `;
    resultsGrid.appendChild(card);
  });
}

// Analytics Setup
function initAnalytics() {
  const chartEl = document.getElementById('ratings-dist-chart');
  if (!chartEl || !window.Plotly) return;

  const genresCount = {};
  movies.forEach(m => {
    m.genres.forEach(g => {
      genresCount[g] = (genresCount[g] || 0) + 1;
    });
  });

  const sortedGenres = Object.entries(genresCount).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const gNames = sortedGenres.map(x => x[0]).reverse();
  const gVals = sortedGenres.map(x => x[1]).reverse();

  const data = [{
    type: 'bar',
    x: gVals,
    y: gNames,
    orientation: 'h',
    marker: {
      color: '#6366f1'
    }
  }];

  const layout = {
    title: { text: 'Top 10 Genres in Dataset', font: { color: '#f8fafc', size: 14 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 90, r: 20, t: 40, b: 40 },
    xaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.06)' },
    yaxis: { color: '#f8fafc' },
    font: { family: 'sans-serif' }
  };

  Plotly.newPlot(chartEl, data, layout, { responsive: true, displayModeBar: false });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Run on load
window.addEventListener('DOMContentLoaded', initApp);
