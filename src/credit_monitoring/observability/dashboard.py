"""Self-contained HTML dashboard for pipeline health and model degradation."""

from __future__ import annotations

from html import escape

from credit_monitoring import config


def _level(batch: dict) -> str:
    flags = batch["flags"]
    if flags["silent_degradation"]:
        return "critical"
    if flags["drift_alert"] or flags["performance_drop"] or flags["contract_blocked"]:
        return "warning"
    if flags["calibration_alert"]:
        return "warning"
    return "ok"


def _fmt_pct(value, digits=2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_num(value, digits=4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _psi_color(psi: float) -> str:
    if psi >= config.PSI_SIGNIFICANT:
        return "#b42318"
    if psi >= config.PSI_MODERATE:
        return "#b54708"
    return "#027a48"


def render_dashboard(health: dict) -> str:
    batches = health.get("batches") or []
    steps = health.get("pipeline", {}).get("steps") or []
    alerts = health.get("active_alerts") or []
    metrics = health.get("metrics_documented") or []

    kpis = [
        ("Alertas ativos", str(health.get("alert_count", 0)), "critical" if alerts else "ok"),
        (
            "Degradação silenciosa",
            str(sum(1 for b in batches if b["flags"]["silent_degradation"])),
            "critical"
            if any(b["flags"]["silent_degradation"] for b in batches)
            else "ok",
        ),
        (
            "PSI máximo",
            _fmt_num(max((b["psi_max"] for b in batches), default=0), 4),
            "warning"
            if any(b["flags"]["drift_alert"] for b in batches)
            else "ok",
        ),
        (
            "Tarefas com falha",
            str(health.get("pipeline", {}).get("failed_steps", 0)),
            "critical" if health.get("pipeline", {}).get("failed_steps") else "ok",
        ),
    ]

    kpi_html = "".join(
        f'<article class="kpi {_level}"><p class="label">{escape(label)}</p>'
        f'<p class="value">{escape(value)}</p></article>'
        for label, value, _level in kpis
    )

    alert_html = (
        "<ul class='alerts'>"
        + "".join(f"<li>{escape(a)}</li>" for a in alerts)
        + "</ul>"
        if alerts
        else "<p class='muted'>Nenhum alerta ativo nos lotes avaliados.</p>"
    )

    batch_rows = []
    for batch in batches:
        level = _level(batch)
        batch_rows.append(
            "<tr class='{level}'>"
            "<td><strong>{name}</strong></td>"
            "<td>{rows}</td>"
            "<td>{blocked}</td>"
            "<td>{psi}</td>"
            "<td>{gap}</td>"
            "<td>{auc}</td>"
            "<td>{approved_bad}</td>"
            "<td>{flags}</td>"
            "</tr>".format(
                level=level,
                name=escape(batch["batch"]),
                rows=batch["rows"],
                blocked=batch["blocked_rows"],
                psi=_fmt_num(batch["psi_max"], 4),
                gap=_fmt_pct(batch["calibration_gap"]),
                auc=_fmt_num(batch.get("roc_auc"), 4),
                approved_bad=_fmt_pct(batch.get("default_rate_approved")),
                flags=escape(", ".join(batch["alerts"]) or "estável"),
            )
        )

    features = []
    for batch in batches:
        features.extend(batch.get("psi_by_feature", {}).keys())
    features = sorted(set(features))
    heat_head = "<th>feature</th>" + "".join(
        f"<th>{escape(b['batch'])}</th>" for b in batches
    )
    heat_rows = []
    for feature in features:
        cells = "".join(
            "<td style='color:{color}'>{value}</td>".format(
                color=_psi_color(batch.get("psi_by_feature", {}).get(feature, 0.0)),
                value=_fmt_num(batch.get("psi_by_feature", {}).get(feature), 4),
            )
            for batch in batches
        )
        heat_rows.append(f"<tr><td>{escape(feature)}</td>{cells}</tr>")

    step_rows = []
    for step in steps:
        status = step.get("status") or "success"
        step_rows.append(
            "<tr class='{cls}'><td>{task}</td><td>{batch}</td><td>{status}</td>"
            "<td>{duration}</td><td>{rows}</td><td>{error}</td></tr>".format(
                cls="ok" if status == "success" else "critical",
                task=escape(str(step.get("task"))),
                batch=escape(str(step.get("batch") or "—")),
                status=escape(status),
                duration=_fmt_num(step.get("duration_seconds"), 3),
                rows=step.get("rows") if step.get("rows") is not None else "—",
                error=escape(str(step.get("error") or "")),
            )
        )

    metric_rows = "".join(
        "<tr><td><code>{name}</code></td><td>{source}</td><td>{question}</td></tr>".format(
            name=escape(m["name"]),
            source=escape(m["source"]),
            question=escape(m["question"]),
        )
        for m in metrics
    )

    generated = escape(str(health.get("generated_at") or ""))
    style = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#0f1419;color:#e7ecf3}
main{max-width:1100px;margin:0 auto;padding:2rem}
h1,h2{font-weight:600}
h1{font-size:1.6rem;margin:0 0 .3rem}
.sub{color:#9aa7b8;margin-bottom:1.5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}
.kpi{background:#1a2330;border:1px solid #2a3848;border-radius:12px;padding:1rem}
.kpi .label{margin:0;color:#9aa7b8;font-size:.85rem}
.kpi .value{margin:.3rem 0 0;font-size:1.6rem}
.ok{--tone:#12b76a}.warning{--tone:#f79009}.critical{--tone:#f04438}
.kpi.ok{border-color:#12b76a33}.kpi.warning{border-color:#f7900933}.kpi.critical{border-color:#f0443833}
.kpi .value{color:var(--tone)}
section{background:#1a2330;border:1px solid #2a3848;border-radius:12px;padding:1.2rem;margin-top:1.2rem}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{text-align:left;padding:.45rem .5rem;border-bottom:1px solid #2a3848}
tr.warning td:first-child{color:#f79009}
tr.critical td:first-child{color:#f04438}
tr.ok td:first-child{color:#12b76a}
.alerts{margin:.4rem 0 0;padding-left:1.2rem}
.muted{color:#9aa7b8}
code{background:#0f1419;padding:.1rem .3rem;border-radius:4px}
"""
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Observabilidade — credit scoring</title>
  <style>{style}</style>
</head>
<body>
<main>
  <h1>Etapa 3 — Observabilidade de pipelines e modelos</h1>
  <p class="sub">Gerado em {generated}. Logs JSONL + MLflow + métricas de saúde. Cruza contrato de dados, PSI da Etapa 2 e calibração com rótulos.</p>
  <div class="grid">{kpi_html}</div>
  <section>
    <h2>Alertas (Etapa 3)</h2>
    {alert_html}
    <p class="muted">Degradação silenciosa = PSI estável em todas as features e erro de calibração ≥ {config.CALIBRATION_GAP_ALERT:.0%}. Esse alerta só existe porque as duas camadas de monitoramento se cruzam.</p>
  </section>
  <section>
    <h2>Saúde por lote</h2>
    <table>
      <thead><tr><th>lote</th><th>linhas</th><th>bloqueadas</th><th>PSI máx</th><th>erro de calibração</th><th>ROC-AUC</th><th>inadimplência dos aprovados</th><th>alertas</th></tr></thead>
      <tbody>{"".join(batch_rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>PSI por feature × lote</h2>
    <p class="muted">Verde &lt; {config.PSI_MODERATE:.2f} (estável); âmbar até {config.PSI_SIGNIFICANT:.2f} (moderado); vermelho acima (significativo).</p>
    <table>
      <thead><tr>{heat_head}</tr></thead>
      <tbody>{"".join(heat_rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>Execução do pipeline (logs)</h2>
    <p class="muted">Fonte: <code>logs/pipeline.jsonl</code>. Última ocorrência de cada tarefa.</p>
    <table>
      <thead><tr><th>tarefa</th><th>lote</th><th>status</th><th>duração (s)</th><th>linhas</th><th>erro</th></tr></thead>
      <tbody>{"".join(step_rows) or "<tr><td colspan='6' class='muted'>Nenhum log ainda. Rode o pipeline e depois <code>uv run python -m credit_monitoring.observability</code>.</td></tr>"}</tbody>
    </table>
  </section>
  <section>
    <h2>Métricas de saúde documentadas (Etapa 3)</h2>
    <table>
      <thead><tr><th>métrica</th><th>origem</th><th>pergunta que responde</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
