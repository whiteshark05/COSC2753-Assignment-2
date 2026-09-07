// Single place to point the frontend at a backend.
//
// The bundled mock_server/ (see README.md) implements this exact contract
// with fake data, so you can develop and demo the UI before the real
// classification/similarity-search pipeline is wired up. Swap these two
// URLs to point at the real service — nothing else in the frontend needs
// to change as long as the response shapes documented in README.md match.
const API_CONFIG = {
    classifyUrl: 'https://cosc2753-mock-server.onrender.com/api/classify',
    searchSimilarUrl: 'https://cosc2753-mock-server.onrender.com/api/search-similar',
};