function wrapCodeBlocks() {
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach((code, index) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-wrapper collapsed';
    const pre = code.parentElement;
    pre.parentNode.insertBefore(wrapper, pre);

    const header = document.createElement('div');
    header.className = 'code-header';

    const collapseButton = document.createElement('button');
    collapseButton.className = 'code-button collapse-button';
    collapseButton.textContent = 'Expand';
    collapseButton.addEventListener('click', () => {
      wrapper.classList.toggle('collapsed');
      collapseButton.textContent = wrapper.classList.contains('collapsed') ? 'Expand' : 'Collapse';
    });

    const copyButton = document.createElement('button');
    copyButton.className = 'code-button';
    copyButton.textContent = 'Copy';
    copyButton.addEventListener('click', () => {
      navigator.clipboard.writeText(code.textContent)
        .then(() => {
          copyButton.textContent = 'Copied!';
          setTimeout(() => {
            copyButton.textContent = 'Copy';
          }, 2000);
        })
        .catch(err => {
          console.error('Failed to copy:', err);
          copyButton.textContent = 'Failed';
          setTimeout(() => {
            copyButton.textContent = 'Copy';
          }, 2000);
        });
    });
    
    const language = code.className.match(/language-(\w+)/)?.[1];
    if (language) {
      const languageLabel = document.createElement('span');
      languageLabel.style.marginRight = 'auto';
      languageLabel.style.color = 'var(--text-secondary)';
      languageLabel.textContent = language;
      header.appendChild(languageLabel);
    }

    header.appendChild(collapseButton);
    header.appendChild(copyButton);
    wrapper.appendChild(header);
    wrapper.appendChild(pre);
  });
}

function setupImageModal() {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const closeBtn = document.getElementsByClassName('modal-close')[0];

  document.querySelectorAll('#post-content img').forEach(img => {
    img.onclick = function() {
      modal.style.display = 'block';
      modalImg.src = this.src;
    }
  });

  closeBtn.onclick = function() {
    modal.style.display = 'none';
  }

  modal.onclick = function(event) {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  }

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && modal.style.display === 'block') {
      modal.style.display = 'none';
    }
  });
}

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
      wrapCodeBlocks();
      setupImageModal();
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