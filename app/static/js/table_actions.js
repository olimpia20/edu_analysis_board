console.warn = function() {};

document.addEventListener('DOMContentLoaded', function () {
    // Preprocessing main link - redirect to tables page
    var preprocessingMainLink = document.getElementById('preprocessing-main-link');
    if (preprocessingMainLink) {
        preprocessingMainLink.addEventListener('click', function(e) {
            // Check if the click was on the main text/icon (not the dropdown arrow)
            var target = e.target;
            var isMainText = target.closest('span') || target.closest('i') || target === this;
            
            if (isMainText) {
                // Check if we're already on the tables page
                var currentPath = window.location.pathname;
                if (currentPath === '/tables') {
                    // Already on tables page, just toggle the dropdown without reloading
                    e.preventDefault();
                    e.stopPropagation();
                    var collapseElement = document.getElementById('collapsePreprocessing');
                    if (collapseElement) {
                        $(collapseElement).collapse('toggle');
                    }
                } else {
                    // On other page, navigate to tables page
                    e.preventDefault();
                    e.stopPropagation();
                    window.location.href = '/tables';
                }
            }
        });
    }

    // Clean Data (Remove Empty Rows) AJAX
    var cleanBtn = document.getElementById('btn-clean-data');
    var cleanColumnsModal = document.getElementById('cleanColumnsModal');
    var cleanColumnsCheckboxes = document.getElementById('clean-columns-checkboxes');
    var confirmCleanColumnsBtn = document.getElementById('confirmCleanColumns');
    var selectedCleanColumns = [];

    function populateCleanColumnsModal() {
        // Get columns from DataTable
        if (window.$ && $.fn.dataTable && $('#dataTable').length) {
            var dt = $('#dataTable').DataTable();
            var allColumns = dt.settings().init().columns.map(function(col) { return col.title; });
            cleanColumnsCheckboxes.innerHTML = '';
            allColumns.forEach(function(col) {
                var id = 'cleancol_' + col.replace(/\W/g, '_');
                cleanColumnsCheckboxes.innerHTML +=
                    '<div class="form-check">' +
                    '<input class="form-check-input" type="checkbox" value="' + col + '" id="' + id + '">' +
                    '<label class="form-check-label" for="' + id + '">' + col + '</label>' +
                    '</div>';
            });
        }
    }

    if (cleanBtn) {
        cleanBtn.onclick = function(e) {
            e.preventDefault();
            populateCleanColumnsModal();
            $(cleanColumnsModal).modal('show');
        };
    }

    if (confirmCleanColumnsBtn) {
        confirmCleanColumnsBtn.onclick = function() {
            var checked = Array.from(cleanColumnsCheckboxes.querySelectorAll('input[type="checkbox"]:checked'));
            if (checked.length === 0) {
                alert('Please select at least one column to check for missing values.');
                return;
            }
            selectedCleanColumns = checked.map(function(cb) { return cb.value; });
            // Send AJAX request to clean data with selected columns
            fetch('/ajax_clean_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': window.csrf_token
                },
                credentials: 'include',
                body: JSON.stringify({ columns: selectedCleanColumns })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    window.location.href = '/tables';
                } else {
                    alert('Failed to clean data: ' + (data.error || 'Unknown error'));
                }
            });
            $(cleanColumnsModal).modal('hide');
        };
    }

    // Drop Columns Modal logic
    var dropColumnsBtn = document.getElementById('btn-drop-columns');
    var dropColumnsModal = document.getElementById('dropColumnsModal');
    var dropColumnsCheckboxes = document.getElementById('drop-columns-checkboxes');
    var restoreDropColumnsBtn = document.getElementById('restoreDropColumns');
    var confirmDropColumnsBtn = document.getElementById('confirmDropColumns');
    var dropColumnsForm = document.getElementById('dropColumnsForm');
    var allColumns = [];

    function populateDropColumnsModal() {
        // Get columns from DataTable
        if (window.$ && $.fn.dataTable && $('#dataTable').length) {
            var dt = $('#dataTable').DataTable();
            allColumns = dt.settings().init().columns.map(function(col) { return col.title; });
            var dtypes = window.table_dtypes || {};
            dropColumnsCheckboxes.innerHTML = '';
            allColumns.forEach(function(col) {
                var id = 'dropcol_' + col.replace(/\W/g, '_');
                var type = dtypes[col] ? ' <span class="text-muted">(' + dtypes[col] + ')</span>' : '';
                dropColumnsCheckboxes.innerHTML +=
                    '<div class="form-check">' +
                    '<input class="form-check-input" type="checkbox" value="' + col + '" id="' + id + '" checked>' +
                    '<label class="form-check-label" for="' + id + '">' + col + type + '</label>' +
                    '</div>';
            });
        }
    }

    if (dropColumnsBtn && dropColumnsModal) {
        dropColumnsBtn.onclick = function(e) {
            e.preventDefault();
            populateDropColumnsModal();
            // Show modal (Bootstrap 4)
            $(dropColumnsModal).modal('show');
        };
    }

    // Restore button: uncheck all checkboxes
    if (restoreDropColumnsBtn) {
        restoreDropColumnsBtn.onclick = function() {
            var checkboxes = dropColumnsCheckboxes.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(function(cb) { cb.checked = false; });
        };
    }

    // Drop button: send AJAX to drop columns
    if (confirmDropColumnsBtn) {
        confirmDropColumnsBtn.onclick = function() {
            var checked = Array.from(dropColumnsCheckboxes.querySelectorAll('input[type="checkbox"]:checked'));
            if (checked.length === 0) {
                alert('Please select at least one column to drop.');
                return;
            }
            var columnsToDrop = checked.map(function(cb) { return cb.value; });
            fetch('/ajax_drop_columns', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': window.csrf_token
                },
                credentials: 'include',
                body: JSON.stringify({ columns: columnsToDrop })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Redirect to tables page after successful column dropping
                    window.location.href = '/tables';
                } else {
                    alert('Failed to drop columns: ' + (data.error || 'Unknown error'));
                }
            });
        };
    }

    // (Legacy) Dropna button logic
    var btn = document.getElementById('dropna-btn');
    if (btn) {
        btn.onclick = function(e) {
            e.preventDefault();
            if (confirm("Are you sure you want to create a new dataset without empty rows?")) {
                // Get CSRF token from meta tag
                const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
                if (!csrfMetaTag) {
                    alert('CSRF token not found. Please refresh the page.');
                    return;
                }
                const csrfToken = csrfMetaTag.getAttribute('content');
                
                fetch('/dropna', { 
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': csrfToken
                    }
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.new_file) {
                            // Redirect to the tables page with the new CSV selected
                            window.location.href = '/tables?csv=' + encodeURIComponent(data.new_file);
                        } else {
                            alert('Failed to create cleaned dataset.');
                        }
                    });
            }
        };
    }

    $('#dropColumnsModal').on('hidden.bs.modal', function () {
        setTimeout(function() {
            var dropBtn = document.getElementById('btn-drop-columns');
            if (dropBtn) dropBtn.focus();
        }, 100); // 100ms delay
    });

    // Normalize Data Modal logic
    var normalizeDataBtn = document.getElementById('btn-normalize-data');
    var normalizeDataModal = document.getElementById('normalizeDataModal');
    var normalizeColumnsTable = document.getElementById('normalize-columns-table');
    var applyNormalizeDataBtn = document.getElementById('applyNormalizeData');

    function populateNormalizeDataModal() {
        // Get columns and types from DataTable (if available)
        if (window.$ && $.fn.dataTable && $('#dataTable').length) {
            var dt = $('#dataTable').DataTable();
            var columns = dt.settings().init().columns.map(function(col) { return col.title; });
            // Try to get types from the global variable if available
            var dtypes = window.table_dtypes || {};
            normalizeColumnsTable.innerHTML = '';
            columns.forEach(function(col) {
                var type = dtypes[col] || '';
                var selectId = 'normtype_' + col.replace(/\W/g, '_');
                normalizeColumnsTable.innerHTML +=
                    '<tr>' +
                    '<td>' + col + '</td>' +
                    '<td>' + type + '</td>' +
                    '<td>' +
                    '<select class="form-control form-control-sm" id="' + selectId + '">' +
                    '<option value="none">None</option>' +
                    '<option value="onehot">One-Hot</option>' +
                    '<option value="ordinal">Ordinal</option>' +
                    '<option value="frequency">Frequency</option>' +
                    '</select>' +
                    '</td>' +
                    '</tr>';
            });
            // Fetch and pre-select recommendations
            fetch('/get_normalization_recommendations', {
                method: 'GET',
                headers: { 'X-Requested-With': 'XMLHttpRequest' },
                credentials: 'include'
            })
            .then(response => response.json())
            .then(recommendations => {
                var rows = normalizeColumnsTable.querySelectorAll('tr');
                rows.forEach(function(row) {
                    var col = row.children[0].textContent;
                    var select = row.querySelector('select');
                    if (select && recommendations[col]) {
                        select.value = recommendations[col];
                    }
                });
            });
        }
    }

    if (normalizeDataBtn && normalizeDataModal) {
        normalizeDataBtn.onclick = function(e) {
            e.preventDefault();
            populateNormalizeDataModal();
            // Show modal (Bootstrap 4)
            $(normalizeDataModal).modal('show');
        };
    }

    // Apply button: send AJAX to normalize/encode columns
    if (applyNormalizeDataBtn) {
        applyNormalizeDataBtn.onclick = function() {
            var transforms = {};
            var rows = normalizeColumnsTable.querySelectorAll('tr');
            rows.forEach(function(row) {
                var col = row.children[0].textContent;
                var select = row.querySelector('select');
                if (select && select.value !== 'none') {
                    transforms[col] = select.value;
                }
            });
            if (Object.keys(transforms).length === 0) {
                alert('Please select at least one transformation.');
                return;
            }
            // Get CSRF token from meta tag
            const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
            if (!csrfMetaTag) {
                alert('CSRF token not found. Please refresh the page.');
                return;
            }
            const csrfToken = csrfMetaTag.getAttribute('content');
            
            fetch('/ajax_normalize_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrfToken
                },
                credentials: 'include',
                body: JSON.stringify({ transforms: transforms })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Redirect to tables page after successful normalization
                    window.location.href = '/tables';
                } else {
                    alert('Failed to normalize/encode data: ' + (data.error || 'Unknown error'));
                }
            });
        };
    }
});

$('#dataTable').on('xhr.dt', function (e, settings, json, xhr) {
    // json.shape should be sent from backend, or use json.recordsTotal and columns.length
    if (json && json.recordsTotal !== undefined && json.columns) {
        document.getElementById('rowxcol-info').textContent = json.recordsTotal + ' x ' + json.columns.length;
    }
});

function reloadDataTableWithNewColumns() {
    $.get('/data_table_ajax', function(json) {
        var columns = json.columns.map(function(col) {
            return { data: col, title: col };
        });

        // Destroy the old DataTable
        if ($.fn.DataTable.isDataTable('#dataTable')) {
            $('#dataTable').DataTable().destroy();
        }

        // Clear the table head and body
        var thead = $('#dataTable thead tr');
        thead.empty();
        var tbody = $('#dataTable tbody');
        tbody.empty();

        columns.forEach(function(col) {
            thead.append('<th>' + col.title + '</th>');
        });

        // Re-initialize DataTable
        var table = $('#dataTable').DataTable({
            serverSide: true,
            processing: true,
            ajax: {
                url: '/data_table_ajax',
                type: 'GET',
                data: function(d) {
                    if (window.firstN) d.num_head = window.firstN;
                    if (window.lastN) d.num_tail = window.lastN;
                }
            },
            columns: columns,
            scrollX: true
        });

        // Re-attach filter event handlers
        $('#firstNForm').off('submit').on('submit', function(e) {
            e.preventDefault();
            window.firstN = $('#firstNInput').val();
            window.lastN = null;
            table.ajax.reload();
        });
        $('#lastNForm').off('submit').on('submit', function(e) {
            e.preventDefault();
            window.lastN = $('#lastNInput').val();
            window.firstN = null;
            table.ajax.reload();
        });
        $('#resetTable').off('click').on('click', function() {
            window.firstN = null;
            window.lastN = null;
            $('#firstNInput').val('');
            $('#lastNInput').val('');
            table.ajax.reload();
        });
    });
}
