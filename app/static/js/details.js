document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('show-columns-btn');
    const details = document.getElementById('column-details');
  
    if (btn && details) {
      btn.addEventListener('click', function () {
        details.style.display = (details.style.display === 'none') ? 'block' : 'none';
      });
    }
  });

  
