function loadPost(filePath) {
  const contentContainer = document.getElementById('post-content');
  const sidebar = document.getElementsByClassName('sidebar')[0];
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

function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  sidebar.classList.toggle('hidden');
}
  

document.addEventListener('DOMContentLoaded', () => {
  loadPost('about');
  const toggleButton = document.getElementById('sidebar-toggle');
  toggleButton.addEventListener('click', toggleSidebar);

  const sidebarLinks = document.querySelectorAll('.sidebar li');
  sidebarLinks.forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 768) {
        toggleSidebar();
      }
    });
  });
});