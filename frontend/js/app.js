(() => {
  'use strict';

  // ---- element refs -------------------------------------------------------
  const dropzone       = document.getElementById('dropzone');
  const dropzoneEmpty  = document.getElementById('dropzoneEmpty');
  const fileInput      = document.getElementById('fileInput');
  const previewImg     = document.getElementById('previewImg');
  const clearImageBtn  = document.getElementById('clearImageBtn');
  const fileError      = document.getElementById('fileError');

  const metaMasterCategory = document.getElementById('metaMasterCategory');
  const metaSubCategory    = document.getElementById('metaSubCategory');
  const metaBaseColour     = document.getElementById('metaBaseColour');
  const metaYear            = document.getElementById('metaYear');

  const classifyBtn    = document.getElementById('classifyBtn');
  const searchBtn      = document.getElementById('searchBtn');
  const statusMessage  = document.getElementById('statusMessage');

  const resultsEmpty   = document.getElementById('resultsEmpty');
  const classifySection = document.getElementById('classifySection');
  const classifyList    = document.getElementById('classifyList');
  const similarSection  = document.getElementById('similarSection');
  const similarGrid     = document.getElementById('similarGrid');
  const similarCount    = document.getElementById('similarCount');

  const ACCEPTED_TYPES = ['image/png', 'image/jpeg'];
  const LABELS = { articleType: 'articleType', season: 'season', gender: 'gender', usage: 'usage' };

  let currentFile = null;

  // ---- upload: click / drag+drop / keyboard --------------------------------
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('is-dragover');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('is-dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) handleFile(file);
  });

  clearImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleFile(file) {
    fileError.hidden = true;

    if (!ACCEPTED_TYPES.includes(file.type)) {
      fileError.textContent = 'That file type isn\u2019t supported \u2014 use a JPG or PNG.';
      fileError.hidden = false;
      return;
    }

    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.alt = file.name;
    previewImg.hidden = false;
    dropzoneEmpty.hidden = true;
    clearImageBtn.hidden = false;

    classifyBtn.disabled = false;
    searchBtn.disabled = false;
    clearResults();
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImg.hidden = true;
    previewImg.removeAttribute('src');
    dropzoneEmpty.hidden = false;
    clearImageBtn.hidden = true;
    classifyBtn.disabled = true;
    searchBtn.disabled = true;
    clearResults();
  }

  // ---- results rendering ---------------------------------------------------
  function clearResults() {
    classifySection.hidden = true;
    similarSection.hidden = true;
    classifyList.innerHTML = '';
    similarGrid.innerHTML = '';
    resultsEmpty.hidden = false;
    statusMessage.textContent = '';
  }

  function renderClassification(data) {
    resultsEmpty.hidden = true;
    classifySection.hidden = false;
    classifyList.innerHTML = '';

    Object.keys(LABELS).forEach((key) => {
      const entry = data[key];
      if (!entry) return;
      const pct = Math.round((entry.confidence || 0) * 100);

      const row = document.createElement('div');
      row.className = 'spec-row';
      row.innerHTML = `
        <dt>${LABELS[key]}</dt>
        <dd>
          <span class="spec-row__value">${escapeHtml(entry.label)}</span>
          <span class="spec-row__bar"><span class="spec-row__bar-fill" style="width:${pct}%"></span></span>
        </dd>
        <span class="spec-row__confidence">${pct}%</span>
      `;
      classifyList.appendChild(row);
    });
  }

  function renderSimilar(results) {
    resultsEmpty.hidden = true;
    similarSection.hidden = false;
    similarGrid.innerHTML = '';
    similarCount.textContent = `(${results.length})`;

    results.slice(0, 10).forEach((item) => {
      const li = document.createElement('li');
      li.className = 'similar-card';
      const pct = Math.round((item.similarity || 0) * 100);
      li.innerHTML = `
        <img class="similar-card__img" src="${escapeAttr(item.imageUrl)}" alt="${escapeAttr(item.articleType || 'Similar item')}" loading="lazy">
        <span class="similar-card__meta">
          <span class="similar-card__type">${escapeHtml(item.articleType || '\u2014')}</span>
          <span class="similar-card__score">${pct}% match</span>
        </span>
      `;
      similarGrid.appendChild(li);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str == null ? '' : String(str);
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return escapeHtml(str).replace(/"/g, '&quot;');
  }

  function setLoading(button, isLoading) {
    button.classList.toggle('is-loading', isLoading);
    button.disabled = isLoading || !currentFile;
  }

  function showStatusError(message) {
    statusMessage.textContent = message;
  }

  // ---- API calls -------------------------------------------------------
  async function classify() {
    if (!currentFile) return;
    statusMessage.textContent = '';
    setLoading(classifyBtn, true);
    searchBtn.disabled = true;

    try {
      const form = new FormData();
      form.append('image', currentFile);
      appendMetaIfPresent(form, 'masterCategory', metaMasterCategory.value);
      appendMetaIfPresent(form, 'subCategory', metaSubCategory.value);
      appendMetaIfPresent(form, 'baseColour', metaBaseColour.value);
      appendMetaIfPresent(form, 'year', metaYear.value);

      const res = await fetch(API_CONFIG.classifyUrl, { method: 'POST', body: form });
      if (!res.ok) throw new Error(await readErrorMessage(res));
      const data = await res.json();
      renderClassification(data);
    } catch (err) {
      showStatusError(`Couldn\u2019t classify that photo \u2014 ${err.message}`);
    } finally {
      setLoading(classifyBtn, false);
      searchBtn.disabled = !currentFile;
    }
  }

  async function searchSimilar() {
    if (!currentFile) return;
    statusMessage.textContent = '';
    setLoading(searchBtn, true);
    classifyBtn.disabled = true;

    try {
      const form = new FormData();
      form.append('image', currentFile);

      const res = await fetch(API_CONFIG.searchSimilarUrl, { method: 'POST', body: form });
      if (!res.ok) throw new Error(await readErrorMessage(res));
      const data = await res.json();
      renderSimilar(data.results || []);
    } catch (err) {
      showStatusError(`Couldn\u2019t search for similar items \u2014 ${err.message}`);
    } finally {
      setLoading(searchBtn, false);
      classifyBtn.disabled = !currentFile;
    }
  }

  function appendMetaIfPresent(form, key, value) {
    if (value !== undefined && value !== null && String(value).trim() !== '') {
      form.append(key, String(value).trim());
    }
  }

  async function readErrorMessage(res) {
    try {
      const data = await res.json();
      return data.error || `server responded with ${res.status}`;
    } catch {
      return `server responded with ${res.status}`;
    }
  }

  classifyBtn.addEventListener('click', classify);
  searchBtn.addEventListener('click', searchSimilar);
})();
