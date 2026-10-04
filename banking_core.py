"""
Banking Customer Support AI Agent - Multi-Agent Core Engine
===========================================================

Course-End Project 3. A multi-agent GenAI system for banking customer support,
imported by both the Streamlit app (``app.py``) and the notebook.

Agents (Part 1):
  ClassifierAgent      - routes a message to Positive Feedback / Negative Feedback / Query
  FeedbackHandlerAgent - positive -> warm thank-you; negative -> new ticket + empathetic reply
  QueryHandlerAgent    - extract ticket number -> look up support_tickets -> return status
  SupervisorAgent      - orchestrates: classify -> route -> respond, logging every step

LLMOps (Part 2):
  action logging + performance_metrics() (routing success, agent success/failure)
  evaluate_classification() and evaluate_responses() (QA-based scoring)

LLM backend: Groq (set GROQ_API_KEY; free at console.groq.com/keys). No embeddings
are required for this project. A .env file is supported.
"""

from __future__ import annotations

import os
import re
import json
import random
import sqlite3
import datetime as dt
from typing import Dict, List, Optional

# ===========================================================================
# CONFIG
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("BANKING_DATA", os.path.join(BASE_DIR, "Data"))
DB_PATH = os.path.join(DATA_DIR, "support_tickets.db")
LOG_PATH = os.path.join(DATA_DIR, "agent_logs.json")

DEFAULT_MODEL = "openai/gpt-oss-20b"
CATEGORIES = ("Positive Feedback", "Negative Feedback", "Query")
STATUSES = ("Unresolved", "In Progress", "Resolved")


def configure(data_dir: str) -> None:
    """Point the engine at a data directory (folder with support_tickets.db)."""
    global DATA_DIR, DB_PATH, LOG_PATH
    DATA_DIR = data_dir
    DB_PATH = os.path.join(data_dir, "support_tickets.db")
    LOG_PATH = os.path.join(data_dir, "agent_logs.json")


# ===========================================================================
# SUPPORT DATABASE  (the support_tickets table)
# ===========================================================================
def _conn():
    return sqlite3.connect(DB_PATH)


def ensure_db() -> None:
    """Create + seed the DB if it does not exist yet."""
    if not os.path.exists(DB_PATH):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "init_db", os.path.join(DATA_DIR, "init_db.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.init_db(DB_PATH)


def list_tickets() -> List[Dict]:
    ensure_db()
    conn = _conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM support_tickets ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ticket(ticket_number: str) -> Optional[Dict]:
    ensure_db()
    conn = _conn()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM support_tickets WHERE ticket_number = ?",
        (str(ticket_number),)).fetchone()
    conn.close()
    return dict(row) if row else None


def _generate_ticket_number() -> str:
    """A unique 6-digit ticket number not already in the DB."""
    existing = {t["ticket_number"] for t in list_tickets()}
    while True:
        num = str(random.randint(100000, 999999))
        if num not in existing:
            return num


def create_ticket(customer_name: str, issue: str,
                  status: str = "Unresolved") -> Dict:
    """Insert a new unresolved ticket and return it."""
    ensure_db()
    num = _generate_ticket_number()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _conn()
    conn.execute(
        "INSERT INTO support_tickets "
        "(ticket_number, customer_name, issue, status, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (num, customer_name or "Customer", issue, status, now, now))
    conn.commit()
    conn.close()
    return get_ticket(num)


def update_ticket_status(ticket_number: str, status: str) -> Optional[Dict]:
    ensure_db()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _conn()
    conn.execute(
        "UPDATE support_tickets SET status = ?, updated_at = ? "
        "WHERE ticket_number = ?", (status, now, str(ticket_number)))
    conn.commit()
    conn.close()
    return get_ticket(ticket_number)


# ===========================================================================
# ACTION LOGGING  (Part 2 - logs & debugging)
# ===========================================================================
def log_action(agent: str, message: str, output: str,
               success: bool, detail: str = "") -> None:
    try:
        logs = json.load(open(LOG_PATH)) if os.path.exists(LOG_PATH) else []
    except Exception:
        logs = []
    logs.append({
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent,
        "input": str(message)[:300],
        "output": str(output)[:300],
        "success": bool(success),
        "detail": str(detail)[:200],
    })
    json.dump(logs, open(LOG_PATH, "w"), indent=2)


def get_logs() -> List[Dict]:
    try:
        return json.load(open(LOG_PATH)) if os.path.exists(LOG_PATH) else []
    except Exception:
        return []


def clear_logs() -> None:
    json.dump([], open(LOG_PATH, "w"))


def performance_metrics() -> Dict:
    """Routing success rate and per-agent success counts from the logs."""
    logs = get_logs()
    total = len(logs)
    ok = sum(1 for l in logs if l["success"])
    per_agent: Dict[str, Dict[str, int]] = {}
    for l in logs:
        d = per_agent.setdefault(l["agent"], {"calls": 0, "success": 0})
        d["calls"] += 1
        d["success"] += int(l["success"])
    routes = [l for l in logs if l["agent"] == "SupervisorAgent"]
    routed_ok = sum(1 for l in routes if l["success"])
    return {
        "total_actions": total,
        "overall_success_rate": round(ok / total, 3) if total else None,
        "per_agent": per_agent,
        "routing_attempts": len(routes),
        "routing_success": routed_ok,
        "routing_success_rate": round(routed_ok / len(routes), 3) if routes else None,
    }


# ===========================================================================
# LLM (Groq)
# ===========================================================================
def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """Return a Groq chat model. Requires GROQ_API_KEY."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=temperature)


def _llm_text(llm, system: str, user: str) -> str:
    """Call the chat model with a system + user message and return the text."""
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return (resp.content or "").strip()


# ===========================================================================
# AGENT 1 - CLASSIFIER
# ===========================================================================
CLASSIFIER_SYSTEM = """You are the Classifier Agent in a banking customer-support system.
Classify the customer's message into EXACTLY ONE of these three categories:
- Positive Feedback : the customer is thanking, praising, or expressing satisfaction.
- Negative Feedback : the customer is complaining, reporting a problem, or expressing dissatisfaction, WITHOUT asking about a specific existing ticket.
- Query : the customer is asking for information or a status update, especially about a ticket number.

Reply with ONLY the category name, nothing else: "Positive Feedback", "Negative Feedback", or "Query"."""


class ClassifierAgent:
    """Routes a message into Positive Feedback / Negative Feedback / Query."""

    def __init__(self, llm):
        self.llm = llm

    def classify(self, message: str) -> str:
        try:
            raw = _llm_text(self.llm, CLASSIFIER_SYSTEM, message)
            label = self._parse(raw)
            log_action("ClassifierAgent", message, label, True, f"raw={raw[:60]}")
            return label
        except Exception as e:
            log_action("ClassifierAgent", message, "error", False, str(e))
            raise

    @staticmethod
    def _parse(raw: str) -> str:
        low = raw.lower()
        if "positive" in low:
            return "Positive Feedback"
        if "negative" in low:
            return "Negative Feedback"
        if "query" in low:
            return "Query"
        # fallback heuristic
        if any(w in low for w in ["thank", "great", "appreciate", "awesome"]):
            return "Positive Feedback"
        if re.search(r"\b\d{6}\b", raw) or "status" in low:
            return "Query"
        return "Negative Feedback"


# ===========================================================================
# AGENT 2 - FEEDBACK HANDLER
# ===========================================================================
POSITIVE_SYSTEM = """You are the Feedback Handler Agent for a bank.
The customer gave POSITIVE feedback. Write a warm, short, personalized thank-you
(1-2 sentences). Address the customer by name if provided. Keep it professional and friendly."""


class FeedbackHandlerAgent:
    """Handles positive and negative feedback."""

    def __init__(self, llm):
        self.llm = llm

    def handle(self, classification: str, message: str,
               customer_name: str = "Customer") -> Dict:
        if classification == "Positive Feedback":
            return self._positive(message, customer_name)
        return self._negative(message, customer_name)

    def _positive(self, message: str, customer_name: str) -> Dict:
        try:
            user = f"Customer name: {customer_name}\nCustomer message: {message}"
            reply = _llm_text(self.llm, POSITIVE_SYSTEM, user)
            if not reply:
                reply = (f"Thank you for your kind words, {customer_name}! "
                         "We're delighted to assist you.")
            log_action("FeedbackHandlerAgent", message, reply, True, "positive")
            return {"response": reply, "ticket": None, "action": "positive_thank_you"}
        except Exception as e:
            log_action("FeedbackHandlerAgent", message, "error", False, str(e))
            reply = (f"Thank you for your kind words, {customer_name}! "
                     "We're delighted to assist you.")
            return {"response": reply, "ticket": None, "action": "positive_thank_you"}

    def _negative(self, message: str, customer_name: str) -> Dict:
        try:
            ticket = create_ticket(customer_name, message, status="Unresolved")
            num = ticket["ticket_number"]
            reply = (f"We apologize for the inconvenience. A new ticket #{num} has been "
                     "generated, and our team will follow up shortly.")
            log_action("FeedbackHandlerAgent", message, reply, True,
                       f"created ticket {num}")
            return {"response": reply, "ticket": ticket, "action": "ticket_created"}
        except Exception as e:
            log_action("FeedbackHandlerAgent", message, "error", False, str(e))
            raise


# ===========================================================================
# AGENT 3 - QUERY HANDLER
# ===========================================================================
class QueryHandlerAgent:
    """Extracts a ticket number from the message and returns its status."""

    def __init__(self, llm=None):
        self.llm = llm  # not strictly needed; extraction is deterministic

    @staticmethod
    def extract_ticket_number(message: str) -> Optional[str]:
        m = re.search(r"\b(\d{6})\b", message)
        return m.group(1) if m else None

    def handle(self, message: str) -> Dict:
        num = self.extract_ticket_number(message)
        if not num:
            reply = ("I couldn't find a 6-digit ticket number in your message. "
                     "Please share your ticket number so I can check its status.")
            log_action("QueryHandlerAgent", message, reply, False, "no ticket number")
            return {"response": reply, "ticket": None, "action": "no_ticket_number"}
        ticket = get_ticket(num)
        if not ticket:
            reply = (f"I'm sorry, I couldn't find ticket #{num} in our system. "
                     "Please double-check the number.")
            log_action("QueryHandlerAgent", message, reply, False, f"{num} not found")
            return {"response": reply, "ticket": None, "action": "ticket_not_found"}
        reply = f"Your ticket #{num} is currently marked as: {ticket['status']}."
        log_action("QueryHandlerAgent", message, reply, True, f"{num} -> {ticket['status']}")
        return {"response": reply, "ticket": ticket, "action": "status_returned"}


# ===========================================================================
# SUPERVISOR - orchestrates the multi-agent workflow
# ===========================================================================
class SupervisorAgent:
    """Coordinates the agents: classify -> route -> respond."""

    def __init__(self, llm):
        self.llm = llm
        self.classifier = ClassifierAgent(llm)
        self.feedback = FeedbackHandlerAgent(llm)
        self.query = QueryHandlerAgent(llm)

    def route(self, message: str, customer_name: str = "Customer") -> Dict:
        steps = []
        classification = self.classifier.classify(message)
        steps.append({"agent": "ClassifierAgent", "detail": f"classified as {classification}"})

        if classification == "Query":
            agent_path = "Classifier -> Query Handler"
            result = self.query.handle(message)
            steps.append({"agent": "QueryHandlerAgent", "detail": result["action"]})
        elif classification == "Positive Feedback":
            agent_path = "Classifier -> Positive Feedback Handler"
            result = self.feedback.handle(classification, message, customer_name)
            steps.append({"agent": "FeedbackHandlerAgent", "detail": result["action"]})
        else:  # Negative Feedback
            agent_path = "Classifier -> Negative Feedback Handler"
            result = self.feedback.handle(classification, message, customer_name)
            steps.append({"agent": "FeedbackHandlerAgent", "detail": result["action"]})

        out = {
            "message": message,
            "customer_name": customer_name,
            "classification": classification,
            "agent_path": agent_path,
            "response": result["response"],
            "ticket": result.get("ticket"),
            "steps": steps,
        }
        log_action("SupervisorAgent", message, result["response"], True, agent_path)
        return out


def build_system(model: str = DEFAULT_MODEL):
    """Build the supervisor (and thus all agents). Requires GROQ_API_KEY."""
    return SupervisorAgent(get_llm(model=model))


# ===========================================================================
# EVALUATION  (Part 2 - Model Evaluation)
# ===========================================================================
def classification_test_set() -> List[Dict[str, str]]:
    """Labeled messages for classification accuracy + routing coverage."""
    return [
        {"message": "Thanks for sorting out my net banking login issue.", "label": "Positive Feedback"},
        {"message": "Thank you so much, the team was very helpful!", "label": "Positive Feedback"},
        {"message": "My debit card replacement still hasn't arrived.", "label": "Negative Feedback"},
        {"message": "I was charged twice for the same transaction, this is frustrating.", "label": "Negative Feedback"},
        {"message": "Could you check the status of ticket 650932?", "label": "Query"},
        {"message": "What is the status of my ticket 412087?", "label": "Query"},
        {"message": "Appreciate the quick help with my loan application.", "label": "Positive Feedback"},
        {"message": "The mobile app keeps crashing when I try to pay.", "label": "Negative Feedback"},
    ]


def evaluate_classification(system: "SupervisorAgent", examples=None) -> Dict:
    """Run the classifier over the labeled set; return accuracy, routing success,
    and a per-example breakdown."""
    if examples is None:
        examples = classification_test_set()
    rows, correct = [], 0
    for ex in examples:
        pred = system.classifier.classify(ex["message"])
        ok = (pred == ex["label"])
        correct += ok
        rows.append({"message": ex["message"], "expected": ex["label"],
                     "predicted": pred, "correct": ok})
    n = len(examples)
    return {
        "accuracy": round(correct / n, 3) if n else None,
        "correct": correct, "total": n,
        "routing_success_rate": round(correct / n, 3) if n else None,
        "rows": rows,
    }


def response_eval_examples() -> List[Dict[str, str]]:
    """QA-style checks on generated responses (feedback accuracy, clarity)."""
    return [
        {"query": "Customer says: 'Could you check the status of ticket 650932?'",
         "answer": "Should state that ticket #650932 is currently marked as Resolved."},
        {"query": "Customer says: 'Thanks for fixing my login!'",
         "answer": "Should be a warm, personalized thank-you message."},
        {"query": "Customer says: 'My debit card still hasn't arrived.'",
         "answer": "Should apologize and provide a new 6-digit ticket number."},
    ]


def evaluate_responses(system: "SupervisorAgent", llm=None, examples=None):
    """Grade generated responses with QAEvalChain. Returns (graded, predictions)."""
    from langchain.evaluation.qa import QAEvalChain
    if examples is None:
        examples = response_eval_examples()
    if llm is None:
        llm = system.llm

    predictions = []
    for ex in examples:
        msg = ex["query"].split("'")[1] if "'" in ex["query"] else ex["query"]
        out = system.route(msg, customer_name="Customer")
        predictions.append({"result": out["response"]})

    eval_chain = QAEvalChain.from_llm(llm)
    graded = eval_chain.evaluate(
        examples, predictions,
        question_key="query", answer_key="answer", prediction_key="result")
    return graded, predictions
