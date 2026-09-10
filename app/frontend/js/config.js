// Single place to point the frontend at a backend.
//
// The bundled mock_server/ (see README.md) implements this exact contract
// with fake data, so you can develop and demo the UI before the real
// classification/similarity-search pipeline is wired up. Swap these two
// URLs to point at the real service — nothing else in the frontend needs
// to change as long as the response shapes documented in README.md match.
// const API_CONFIG = {
//     classifyUrl: 'https://cosc2753-mock-server.onrender.com/api/classify',
//     searchSimilarUrl: 'https://cosc2753-mock-server.onrender.com/api/search-similar',
// };

const IS_LOCAL =
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1';

const API_BASE_URL = IS_LOCAL
    ? 'http://localhost:5000'
    : 'https://cosc2753-assignment-2-backend.onrender.com';

const API_CONFIG = {
    classifyUrl: `${API_BASE_URL}/api/classify`,
    searchSimilarUrl: `${API_BASE_URL}/api/search-similar`,
};