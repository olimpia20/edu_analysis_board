document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('show-columns-btn');
  const details = document.getElementById('column-details');

  if (btn && details) {
    btn.addEventListener('click', function () {
      details.style.display = (details.style.display === 'none') ? 'block' : 'none';
    });
  }
});

document.getElementById('show-stats-btn').onclick = function() {
  var statsDiv = document.getElementById('stats-details');
  if (statsDiv.style.display === 'none') {
    statsDiv.style.display = 'block';
  } else {
    statsDiv.style.display = 'none';
  }
};

$(document).ready(function() {
  $.get('/data_table_ajax', function(json) {
    // 1. Build columns array dynamically
    var columns = json.columns.map(function(col) {
      return { data: col, title: col };
    });

    // 2. Fill in the <th> elements
    var thead = $('#dataTable thead tr');
    thead.empty();
    columns.forEach(function(col) {
      thead.append('<th>' + col.title + '</th>');
    });

    // 3. Initialize DataTables with data option
    $('#dataTable').DataTable({
      data: json.data,
      columns: columns,
      scrollX: true
    });
  });
});



