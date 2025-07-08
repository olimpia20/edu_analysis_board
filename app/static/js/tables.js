$(document).ready(function() {
  var firstN = null;
  var lastN = null;
  var columns = []; // Will be set after first AJAX call

  // Initial AJAX to get columns
  $.get('/data_table_ajax', function(json) {
    columns = json.columns.map(function(col) {
      return { data: col, title: col };
    });

    // Fill in the <th> elements
    var thead = $('#dataTable thead tr');
    thead.empty();
    columns.forEach(function(col) {
      thead.append('<th>' + col.title + '</th>');
    });

    // Initialize DataTable
    var table = $('#dataTable').DataTable({
      serverSide: true,
      processing: true,
      ajax: {
        url: '/data_table_ajax',
        type: 'GET',
        data: function(d) {
          if (firstN) d.num_head = firstN;
          if (lastN) d.num_tail = lastN;
        }
      },
      columns: columns,
      scrollX: true
    });

    // First N form
    $('#firstNForm').on('submit', function(e) {
      e.preventDefault();
      firstN = $('#firstNInput').val();
      lastN = null;
      table.ajax.reload();
    });

    // Last N form
    $('#lastNForm').on('submit', function(e) {
      e.preventDefault();
      lastN = $('#lastNInput').val();
      firstN = null;
      table.ajax.reload();
    });

    // Reset form
    $('#resetTable').on('click', function() {
      firstN = null;
      lastN = null;
      $('#firstNInput').val('');
      $('#lastNInput').val('');
      table.ajax.reload();
    });

    $('#dropColumnsModal').on('hidden.bs.modal', function () {
      // Move focus to the Drop Columns sidebar button after modal closes
      var dropBtn = document.getElementById('btn-drop-columns');
      if (dropBtn) dropBtn.focus();
    });
  });
});



