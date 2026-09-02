"""
SignalStory — Streamlit Community Cloud entry point
Compatibility version for the current Phase 4.2 backend.

The current src.server.execute_decision_analysis() accepts one request
dictionary. This UI intentionally does not import the removed
_build_signal_story helper and derives the presentation directly from
the backend response.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from html import escape
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.server import execute_decision_analysis, OFFICIAL_SCENARIOS


st.set_page_config(
    page_title="Signal Story — Decision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#0b0f19; color:#e2e8f0; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; }
.card {
  background:linear-gradient(145deg,#131b2e,#0f172a);
  border:1px solid rgba(255,255,255,.10);
  border-radius:14px; padding:20px; margin-bottom:16px;
}
.badge { display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:11px; font-weight:700; margin-right:6px; }
.green { background:rgba(34,197,94,.15); color:#4ade80; border:1px solid rgba(34,197,94,.35); }
.blue { background:rgba(59,130,246,.15); color:#60a5fa; border:1px solid rgba(59,130,246,.35); }
.purple { background:rgba(168,85,247,.15); color:#c084fc; border:1px solid rgba(168,85,247,.35); }
.amber { background:rgba(245,158,11,.15); color:#fbbf24; border:1px solid rgba(245,158,11,.35); }
.red { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.35); }
.big { font-size:34px; font-weight:800; margin:4px 0; }
.muted { color:#94a3b8; font-size:13px; }
.section-title { font-size:18px; font-weight:750; margin:4px 0 12px; }
</style>
""", unsafe_allow_html=True)


def _scenario_label(s: Dict[str, Any]) -> str:
    title = str(s.get("title", s.get("scenario_id", "Scenario")))
    return f"{s.get('scenario_id', 'CUSTOM')} — {title.split('—', 1)[-1].strip()}"


def _fmt_money(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:,.2f}"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _driver_name(driver: Any) -> str:
    names = {
        "DRIVER_01_INVENTORY": "Inventory / Supply",
        "DRIVER_02_PRICING": "Competitor Pricing",
        "DRIVER_03_MARKETING": "Marketing Inefficiency",
        "DRIVER_04_RETURNS": "Returns",
        "DRIVER_05_SUPPORT": "Customer Support",
        "DRIVER_06_CUSTOMER": "Customer / Account",
        "DRIVER_07_MARKET": "Market Shift",
        "DRIVER_08_PRODUCT_MIX": "Product Mix",
    }
    return names.get(str(driver), str(driver or "Unknown"))


def _build_story(res: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    event = res.get("phase3a", {}).get("event", {}) or {}
    candidates = res.get("phase3a", {}).get("candidate_drivers", []) or []
    p3b = res.get("phase3b", {}) or {}
    diagnosis = (
        p3b.get("diagnosis", {})
        or res.get("phase3a", {}).get("diagnosis", {})
        or {}
    )

    driver = diagnosis.get("driver") or diagnosis.get("established_driver")
    status = (
        diagnosis.get("status")
        or diagnosis.get("overall_status")
        or res.get("phase3a", {}).get("overall_status", "NOT_ESTABLISHED")
    )
    confidence = diagnosis.get("confidence", "NONE")

    current = event.get("current_value")
    baseline = event.get("baseline_value")
    change = event.get("change_percent")
    kpi = event.get("kpi") or scenario.get("kpi_name", "Gross Sales")

    supporting = p3b.get("supporting_evidence", []) or []
    contradictory = p3b.get("contradictory_evidence", []) or []
    next_steps = p3b.get("recommended_next_steps", []) or []
    uncertainties = p3b.get("uncertainties", []) or []

    if status == "STRONGLY_SUPPORTED":
        state = "SUPPORTED"
    elif status == "NOT_ESTABLISHED":
        state = "ABSTENTION"
    else:
        state = "PLAUSIBLE"

    if driver:
        headline = f"{_driver_name(driver)} is the leading explanation for the {kpi} movement."
    else:
        headline = f"No single causal driver is established for the {kpi} movement."

    return {
        "event": event,
        "candidates": candidates,
        "driver": driver,
        "status": status,
        "state": state,
        "confidence": confidence,
        "current": current,
        "baseline": baseline,
        "change": change,
        "kpi": kpi,
        "headline": headline,
        "supporting": supporting,
        "contradictory": contradictory,
        "next_steps": next_steps,
        "uncertainties": uncertainties,
    }


@st.cache_data(show_spinner=False, ttl=60)
def run_analysis_cached(req: Dict[str, Any]) -> Dict[str, Any]:
    # Current backend signature: execute_decision_analysis(req_data)
    return execute_decision_analysis(req)


with st.sidebar:
    st.markdown("## ⚙️ SignalStory")
    st.caption("AI Decision Investigator")

    scenario_ids = [s["scenario_id"] for s in OFFICIAL_SCENARIOS]
    selected_id = st.selectbox(
        "Business Scenario",
        scenario_ids,
        index=0,
        format_func=lambda x: _scenario_label(
            next(s for s in OFFICIAL_SCENARIOS if s["scenario_id"] == x)
        ),
    )
    scenario = next(s for s in OFFICIAL_SCENARIOS if s["scenario_id"] == selected_id)

    st.info(
        f"**Target:** {scenario.get('market')} • "
        f"{scenario.get('product_code') or scenario.get('category') or 'All Products'}\n\n"
        f"**Period:** {scenario.get('period')}"
    )

    provider = st.radio(
        "Reasoning Engine",
        ["mock", "gemini"],
        index=0,
        format_func=lambda x: (
            "Deterministic Mock (recommended)"
            if x == "mock"
            else "Google Gemini"
        ),
    )

    if provider == "gemini":
        gemini_key = st.text_input(
            "Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", "")
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

    analyze = st.button("⚡ Analyze", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Human oversight remains required for consequential decisions.")


request = {
    "scenario_id": selected_id,
    "market": scenario.get("market"),
    "product_code": scenario.get("product_code"),
    "category": scenario.get("category"),
    "date": scenario.get("date"),
    "kpi": scenario.get("kpi", "gross_sales"),
    "provider_mode": provider,
}

if "analysis" not in st.session_state or analyze:
    with st.spinner("Analyzing signals, evidence and candidate drivers…"):
        try:
            st.session_state["analysis"] = run_analysis_cached(request)
            st.session_state["analysis_request"] = request
        except Exception as exc:
            st.error("The analysis pipeline could not be completed.")
            st.exception(exc)
            st.stop()

ui_resp = st.session_state["analysis"]
story = _build_story(ui_resp, scenario)

provider_name = ui_resp.get("metadata", {}).get("provider", provider).upper()
trust = ui_resp.get("data_trust", {}) or {}
trust_score = trust.get("overall_score")
validation = ui_resp.get("metadata", {}).get("validation_status", "PASSED")

st.markdown(f"""
<div class="card">
  <div class="muted">ACCENTURE DECISION INTELLIGENCE PLATFORM</div>
  <h1 style="margin:4px 0;">Signal Story</h1>
  <div class="muted">{escape(str(scenario.get("title", "")))}</div>
  <div style="margin-top:12px;">
    <span class="badge purple">SIGNAL → INVESTIGATION → EVIDENCE → DECISION</span>
    <span class="badge blue">PROVIDER: {escape(provider_name)}</span>
    <span class="badge green">VALIDATION: {escape(str(validation))}</span>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("KPI", story["kpi"])
with c2:
    st.metric("Current", _fmt_money(story["current"]))
with c3:
    st.metric("3-Month Baseline", _fmt_money(story["baseline"]))
with c4:
    st.metric("Change vs Baseline", _fmt_pct(story["change"]))

state_badge = {
    "SUPPORTED": "green",
    "PLAUSIBLE": "purple",
    "ABSTENTION": "red",
}.get(story["state"], "amber")

st.markdown(f"""
<div class="card">
  <span class="badge {state_badge}">STATUS: {escape(story["state"])}</span>
  <div class="section-title" style="margin-top:10px;">Primary Signal / Finding</div>
  <div class="big">{escape(story["headline"])}</div>
  <div class="muted">Backend diagnosis status: {escape(str(story["status"]))} • Confidence: {escape(str(story["confidence"]))}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Diagnostic Breakdown",
    "🔗 Evidence",
    "⚖️ Candidate Drivers",
    "🛡️ Data Trust",
    "✍️ Decision Review",
])

with tab1:
    st.subheader("What happened")
    e = story["event"]
    a, b, c = st.columns(3)
    a.metric("Observed", _fmt_money(e.get("current_value")))
    b.metric("Baseline", _fmt_money(e.get("baseline_value")))
    c.metric("Baseline Change", _fmt_pct(e.get("change_percent")))

    st.markdown("### Executive Summary")
    summary = ui_resp.get("phase3b", {}).get("executive_summary")
    st.write(summary or story["headline"])

    if ui_resp.get("phase3b", {}).get("what_happened"):
        st.info(ui_resp["phase3b"]["what_happened"])

with tab2:
    st.subheader("Evidence Chain")
    evidence = story["supporting"]

    if evidence:
        for ev in evidence:
            st.markdown(
                f"**{ev.get('metric', 'Evidence')}**  \n"
                f"{ev.get('finding', '')}  \n"
                f"<span class='muted'>Source: {ev.get('source_dataset', 'Analytics')} • "
                f"ID: {ev.get('evidence_id', '—')}</span>",
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info("No supporting evidence records were returned for this diagnosis.")

    if story["contradictory"]:
        st.subheader("Contradictory Evidence")
        st.dataframe(
            pd.DataFrame(story["contradictory"]),
            use_container_width=True,
            hide_index=True,
        )

    if story["uncertainties"]:
        st.subheader("Known Uncertainties")
        for item in story["uncertainties"]:
            st.warning(item)

with tab3:
    st.subheader("Candidate Driver Arbitration")
    candidates = story["candidates"]

    if candidates:
        rows = []
        for c in candidates:
            rows.append({
                "Rank": c.get("rank"),
                "Driver": _driver_name(c.get("driver")),
                "Score": c.get("score"),
                "Status": c.get("status"),
                "Confidence": c.get("confidence"),
                "Supporting Evidence": c.get("supporting_evidence_count"),
                "Contradictory Evidence": c.get("contradictory_evidence_count"),
                "Temporal Alignment": c.get("temporal_alignment"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No candidate drivers were generated. This may indicate insufficient "
            "history or telemetry."
        )

with tab4:
    st.subheader("Data Trust & Quality Audit")
    q1, q2, q3 = st.columns(3)
    q1.metric(
        "Trust Score",
        f"{trust_score}/100" if trust_score is not None else "—",
    )
    q2.metric("Trust Status", trust.get("trust_status", "—"))
    q3.metric(
        "Records Assessed",
        f"{trust.get('records_assessed', 0):,}"
        if trust.get("records_assessed") is not None
        else "—",
    )

    if trust.get("dimension_scores"):
        st.json(trust["dimension_scores"])

    st.caption("The analysis is observational; correlation is not proof of causality.")

with tab5:
    st.subheader("Decision Governance")
    dg = ui_resp.get("decision_governance", {}) or {}

    if dg:
        st.json(dg)
    else:
        st.info("No separate decision-governance payload was returned.")

    st.markdown("### Recommended Next Steps")
    if story["next_steps"]:
        for step in story["next_steps"]:
            st.markdown(f"- {step}")
    else:
        st.info("No next steps returned.")

    st.markdown("### Human Review")
    if story["state"] == "ABSTENTION":
        st.error(
            "Do not act on a causal conclusion. Route this case for analyst investigation."
        )
    else:
        st.info(
            "Validate the evidence and business context before taking action."
        )

st.markdown("---")
st.caption(
    "SignalStory • Evidence-grounded decision intelligence • Human-in-the-loop"
)
