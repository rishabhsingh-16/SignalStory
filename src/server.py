"""
Phase 4.2 Decision Intelligence API Server.
Provides REST endpoints and serves the Decision Intelligence Portal UI.
Consumes the frozen Phase 3A deterministic engine and Phase 3B reasoning pipeline.
Zero secrets are exposed to the frontend.
"""

import os
import sys
import json
import time
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.run_analysis import run_analysis
from src.phase3b.engine import Phase3BReasoningEngine
from src.phase3b.mock_reasoning_provider import MockReasoningProvider
from src.phase3b.llm_provider import LLMReasoningProvider, LLMConfig
from src.governance.data_quality import evaluate_data_trust
from src.governance.decision_governance import evaluate_decision_governance, record_analyst_review, get_decision_governance_engine

DOTENV_PATH = PROJECT_ROOT / ".env"

OFFICIAL_SCENARIOS = [
    {
        "scenario_id": "S003",
        "title": "S003 — China / A2520150501 (Marketing Inefficiency Showcase)",
        "market": "China",
        "category": None,
        "product_code": "A2520150501",
        "date": "2021-04-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "April 2021",
        "badge": "PRIMARY SHOWCASE",
        "description": "Gross sales anomaly of -72.1% with supporting evidence of marketing ad spend surge and conversion efficiency drop."
    },
    {
        "scenario_id": "S001",
        "title": "S001 — South Korea / A6519160401 (Return Volume Surge)",
        "market": "South Korea",
        "category": None,
        "product_code": "A6519160401",
        "date": "2021-05-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "May 2021",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Gross sales drop corroborated by customer return rate anomalies and product defect CRM notes."
    },
    {
        "scenario_id": "S002",
        "title": "S002 — South Korea / All Products (Customer Support Escalation)",
        "market": "South Korea",
        "category": None,
        "product_code": None,
        "date": "2021-01-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "January 2021",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Market-wide gross sales drop with surging support ticket volume and sentiment drop."
    },
    {
        "scenario_id": "S004",
        "title": "S004 — China / A0621150308 (Competitor Price Undercut)",
        "market": "China",
        "category": None,
        "product_code": "A0621150308",
        "date": "2021-01-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "January 2021",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Gross sales decline following aggressive competitor pricing discounts."
    },
    {
        "scenario_id": "S005",
        "title": "S005 — Indonesia / All Products (Support Crisis)",
        "market": "Indonesia",
        "category": None,
        "product_code": None,
        "date": "2020-03-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "March 2020",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Market-wide gross sales drop driven by unresolved support escalations."
    },
    {
        "scenario_id": "S006",
        "title": "S006 — India / Processors (Product Mix Shift)",
        "market": "India",
        "category": "Processors",
        "product_code": None,
        "date": "2020-03-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "March 2020",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Category-level gross sales drop with product mix cannibalization."
    },
    {
        "scenario_id": "S007",
        "title": "S007 — Portugal / Wi-Fi Extenders (Category Share Shift)",
        "market": "Portugal",
        "category": "Wi fi extender",
        "product_code": None,
        "date": "2019-09-01",
        "kpi": "category_share",
        "kpi_name": "Category Share",
        "period": "September 2019",
        "badge": "OFFICIAL BENCHMARK",
        "description": "Category share shift and product mix changes in European networking."
    },
    {
        "scenario_id": "S008",
        "title": "S008 — Germany / All Products (Uncertainty & Graceful Fallback)",
        "market": "Germany",
        "category": None,
        "product_code": None,
        "date": "2020-03-01",
        "kpi": "gross_sales",
        "kpi_name": "Gross Sales",
        "period": "March 2020",
        "badge": "UNCERTAINTY TEST",
        "description": "Conflicting multi-source signals correctly preserving NOT_ESTABLISHED uncertainty."
    }
]


def load_api_key_securely() -> Optional[str]:
    """Loads GEMINI_API_KEY from environment or .env without exposing it."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if key and key.strip():
        return key.strip()
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in {"GEMINI_API_KEY", "LLM_API_KEY"}:
                val = v.strip().strip("'\"")
                if val:
                    return val
    return None


SEMANTIC_CONTRACT_PATH = PROJECT_ROOT / "Data" / "semantic" / "kpi_contract.json"

# Bounded caches prevent repeated expensive work during the competition demo.
_ANALYSIS_CACHE = {}
_DATA_TRUST_CACHE = {}
_MAX_ANALYSIS_CACHE = 12
_MAX_DATA_TRUST_CACHE = 12


def load_kpi_contract(kpi_id: Optional[str] = None) -> Dict[str, Any]:
    """Loads KPI semantic contract safely with zero secrets."""
    if not SEMANTIC_CONTRACT_PATH.exists():
        return {"error": "KPI semantic contract file not found", "status": 404}
    try:
        data = json.loads(SEMANTIC_CONTRACT_PATH.read_text(encoding="utf-8"))
        if kpi_id:
            kpis = data.get("kpis", {})
            if kpi_id in kpis:
                return {
                    "version": data.get("version", "1.0.0"),
                    "schema": data.get("schema"),
                    "kpi": kpis[kpi_id]
                }
            return {
                "error": f"KPI '{kpi_id}' not found in semantic contract",
                "status": 404,
                "available_kpis": list(kpis.keys())
            }
        return data
    except Exception as e:
        return {"error": f"Failed to parse semantic contract: {str(e)}", "status": 500}


def execute_decision_analysis(req_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the frozen Phase 3A and Phase 3B analytical pipelines.
    Guarantees that no secrets are returned in the response.
    """
    req = {
        "date": req_data.get("date"),
        "kpi": req_data.get("kpi")
    }
    if req_data.get("market"):
        req["market"] = req_data["market"]
    if req_data.get("category"):
        req["category"] = req_data["category"]
    if req_data.get("product_code"):
        req["product_code"] = req_data["product_code"]

    provider_mode = req_data.get("provider_mode", "mock").lower()

    cache_key = (
        req_data.get("scenario_id"),
        req.get("date"),
        req.get("kpi"),
        req.get("market"),
        req.get("category"),
        req.get("product_code"),
        provider_mode,
    )
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # 1. Phase 3A Deterministic Engine
    t0_p3a = time.perf_counter()
    p3a_payload = run_analysis(req)
    t1_p3a = time.perf_counter()
    p3a_latency_ms = round((t1_p3a - t0_p3a) * 1000, 2)

    # 2. Configure Phase 3B Provider
    api_key = load_api_key_securely()
    if provider_mode == "gemini" and api_key:
        config = LLMConfig(
            provider="gemini",
            model=os.getenv("LLM_MODEL", "gemini-3.6-flash"),
            api_key=api_key,
            temperature=0.0,
            timeout_seconds=45.0,
            enable_safe_fallback=True
        )
        provider = LLMReasoningProvider(config=config)
        provider_name = "gemini"
        model_name = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    else:
        provider = MockReasoningProvider()
        provider_name = "mock"
        model_name = "mock-causal-v1"

    # 3. Phase 3B Reasoning Pipeline
    engine = Phase3BReasoningEngine(default_provider=provider)
    p3b_payload, validation_report = engine.run(p3a_payload, provider=provider)

    # 4. Provenance Attribution
    is_fallback = (p3b_payload.get("validation_status") == "FALLBACK_PRESERVED") or p3b_payload.get("is_fallback", False)
    if provider_name == "gemini":
        provenance = "LIVE_WITH_FALLBACK" if is_fallback else "LIVE_GEMINI"
    else:
        provenance = "MOCK_PROVIDER"

    # 5. Build Safe UI Response Payload
    kpi_contract_snippet = load_kpi_contract(req.get("kpi")).get("kpi")
    trust_key = (req.get("date"), req.get("market"))
    if trust_key not in _DATA_TRUST_CACHE:
        _DATA_TRUST_CACHE[trust_key] = evaluate_data_trust(
            target_date=req.get("date"),
            target_market=req.get("market")
        )
        if len(_DATA_TRUST_CACHE) > _MAX_DATA_TRUST_CACHE:
            _DATA_TRUST_CACHE.pop(next(iter(_DATA_TRUST_CACHE)))
    data_trust_report = _DATA_TRUST_CACHE[trust_key]
    decision_gov = evaluate_decision_governance(p3a_payload, p3b_payload, scenario_id=req_data.get("scenario_id"))

    ui_response = {
        "scenario_id": req_data.get("scenario_id", "CUSTOM"),
        "request": req,
        "kpi_contract": kpi_contract_snippet,
        "data_trust": data_trust_report,
        "decision_governance": decision_gov,
        "phase3a": p3a_payload,
        "phase3b": p3b_payload,
        "metadata": {
            "provider": provider_name,
            "model": model_name,
            "provenance": provenance,
            "validation_status": p3b_payload.get("validation_status", "PASSED"),
            "validation_errors": validation_report.errors if not validation_report.is_valid else [],
            "p3a_latency_ms": p3a_latency_ms,
            "pipeline_latency_ms": p3b_payload.get("pipeline_latency_ms", 0.0),
            "gemini_configured": bool(api_key),
            "frozen_backend": True
        }
    }

    _ANALYSIS_CACHE[cache_key] = ui_response
    if len(_ANALYSIS_CACHE) > _MAX_ANALYSIS_CACHE:
        _ANALYSIS_CACHE.pop(next(iter(_ANALYSIS_CACHE)))
    return ui_response


class DecisionIntelligenceRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving UI assets and JSON API endpoints."""

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204, "text/plain")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/health":
            self._set_headers(200, "application/json")
            resp = {
                "status": "ok",
                "app": "Accenture Decision Intelligence Platform",
                "version": "4.2.0",
                "frozen_backend": True
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path == "/api/scenarios":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(OFFICIAL_SCENARIOS, indent=2).encode("utf-8"))
            return

        if path == "/api/kpi-contract":
            qs = parse_qs(parsed_path.query)
            kpi_id = qs.get("kpi_id", [None])[0]
            contract_resp = load_kpi_contract(kpi_id)
            status_code = contract_resp.get("status", 200) if "error" in contract_resp else 200
            self._set_headers(status_code, "application/json")
            self.wfile.write(json.dumps(contract_resp, indent=2).encode("utf-8"))
            return

        if path == "/api/data-trust":
            qs = parse_qs(parsed_path.query)
            target_date = qs.get("date", [None])[0]
            target_market = qs.get("market", [None])[0]
            trust_report = evaluate_data_trust(target_date=target_date, target_market=target_market)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(trust_report, indent=2).encode("utf-8"))
            return

        if path == "/api/decision-governance":
            qs = parse_qs(parsed_path.query)
            driver_id = qs.get("driver_id", [None])[0]
            engine = get_decision_governance_engine()
            gov_resp = engine.get_driver_governance(driver_id)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(gov_resp, indent=2).encode("utf-8"))
            return

        # Serve static assets
        if path == "/" or path == "":
            file_path = STATIC_DIR / "index.html"
        else:
            rel_path = path.lstrip("/")
            file_path = STATIC_DIR / rel_path

        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            self._set_headers(200, mime_type)
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._set_headers(404, "text/plain")
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/analyst-review":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                req_data = json.loads(body_bytes.decode("utf-8"))
                scenario_id = req_data.get("scenario_id", "S003")
                status = req_data.get("status", "REVIEWED")
                reviewer = req_data.get("reviewer", "Lead Commercial Analyst")
                notes = req_data.get("notes")
                record = record_analyst_review(scenario_id, status, reviewer=reviewer, notes=notes)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(record, indent=2).encode("utf-8"))
            except Exception as e:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": f"Failed to record review: {str(e)}"}).encode("utf-8"))
            return

        if path == "/api/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                req_data = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {str(e)}"}).encode("utf-8"))
                return

            try:
                print(
                    f"[ANALYZE] start scenario={req_data.get('scenario_id')} "
                    f"date={req_data.get('date')} market={req_data.get('market')} "
                    f"provider={req_data.get('provider_mode', 'mock')}",
                    flush=True
                )
                response_data = execute_decision_analysis(req_data)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(response_data, indent=2).encode("utf-8"))
                print(
                    f"[ANALYZE] success scenario={req_data.get('scenario_id')}",
                    flush=True
                )
            except Exception as e:
                print(
                    f"[ANALYZE] FAILED: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True
                )
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "error": f"Analysis execution failed: {type(e).__name__}: {str(e)}"
                }).encode("utf-8"))
            return

        self._set_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        """Clean logging output to standard stderr."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    """Starts the Decision Intelligence HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DecisionIntelligenceRequestHandler)
    print(f"\n==================================================================")
    print(f"Accenture Decision Intelligence Platform Server")
    print(f"Status: RUNNING")
    print(f"URL:    http://{host}:{port}")
    print(f"API:    http://{host}:{port}/api/scenarios")
    print(f"==================================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    run_server(port=port, host=host)
