document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('show-stats-btn');
    var statsDiv = document.getElementById('stats-details');
    if (btn && statsDiv) {
        btn.onclick = function() {
            if (statsDiv.style.display === 'none' || statsDiv.style.display === '') {
                statsDiv.style.display = 'block';
            } else {
                statsDiv.style.display = 'none';
            }
        };
    }
});