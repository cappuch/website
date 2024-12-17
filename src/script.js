function loadPost(filePath) {
    const contentContainer = document.getElementById('post-content');
    if (filePath === 'about') {
      filePath = 'posts/about.md';
    }

    fetch(filePath)
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load post');
        }
        return response.text();
      })
      .then(markdown => {
        contentContainer.innerHTML = marked.parse(markdown);
      })
      .catch(error => {
        contentContainer.innerHTML = `<p>Error loading post: ${error.message}</p>`;
      });
  }

  document.addEventListener('DOMContentLoaded', () => {
    loadPost('about');
  });
