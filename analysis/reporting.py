from __future__ import annotations

from html import escape

import pandas as pd
from IPython.display import HTML


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def interactive_table(df: pd.DataFrame, table_id: str, title: str | None = None) -> HTML:
    """Return a dependency-free searchable/sortable HTML table for Quarto pages."""
    safe_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in table_id)
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")

    title_html = f"<h4>{escape(title)}</h4>" if title else ""
    html = f"""
{title_html}
<div class="interactive-table-wrap" id="{safe_id}_wrap">
  <input class="interactive-table-search" id="{safe_id}_search" type="search" placeholder="Search table" aria-label="Search table">
  <div class="interactive-table-scroll">
    <table class="interactive-table" id="{safe_id}">
      <thead><tr>{headers}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
<script>
(function() {{
  const table = document.getElementById("{safe_id}");
  const search = document.getElementById("{safe_id}_search");
  if (!table || !search) return;

  search.addEventListener("input", function() {{
    const query = search.value.toLowerCase();
    Array.from(table.tBodies[0].rows).forEach(function(row) {{
      row.style.display = row.innerText.toLowerCase().includes(query) ? "" : "none";
    }});
  }});

  Array.from(table.tHead.rows[0].cells).forEach(function(header, index) {{
    header.style.cursor = "pointer";
    header.title = "Click to sort";
    header.addEventListener("click", function() {{
      const tbody = table.tBodies[0];
      const currentDirection = header.getAttribute("data-sort") === "asc" ? "desc" : "asc";
      Array.from(table.tHead.rows[0].cells).forEach(function(cell) {{ cell.removeAttribute("data-sort"); }});
      header.setAttribute("data-sort", currentDirection);
      const rows = Array.from(tbody.rows);
      rows.sort(function(a, b) {{
        const av = a.cells[index].innerText.trim();
        const bv = b.cells[index].innerText.trim();
        const an = Number(av.replace(/[%,$]/g, ""));
        const bn = Number(bv.replace(/[%,$]/g, ""));
        const result = (!Number.isNaN(an) && !Number.isNaN(bn))
          ? an - bn
          : av.localeCompare(bv);
        return currentDirection === "asc" ? result : -result;
      }});
      rows.forEach(function(row) {{ tbody.appendChild(row); }});
    }});
  }});
}})();
</script>
"""
    return HTML(html)
