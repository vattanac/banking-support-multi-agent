"""
Banking Customer Support AI Agent - Streamlit Dashboard
=======================================================

Run with:  streamlit run app.py
Requires GROQ_API_KEY in the environment or a .env file (free: console.groq.com/keys).

Implements the Part 2 UI: accept user input and simulate agent routing; display
classification, response and database interaction; browse tickets and history;
test scenarios per agent role; and a logs/debugging view with agent success/failure.
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import banking_core as bc

load_dotenv()
bc.configure(os.environ.get("BANKING_DATA",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")))
bc.ensure_db()

st.set_page_config(page_title="Banking Support Multi-Agent", page_icon="🏦", layout="wide")
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


@st.cache_resource(show_spinner="Starting the multi-agent system...")
def get_system(model: str):
    return bc.build_system(model=model)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🏦 Banking Support")
    st.caption("Multi-Agent Customer Support System")
    model = st.selectbox("Agent LLM (Groq)",
                         ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
                         index=0)
    key_present = bool(os.environ.get("GROQ_API_KEY"))
    if key_present:
        st.success("GROQ_API_KEY detected.")
    else:
        st.warning("Set GROQ_API_KEY — free at console.groq.com/keys")
        gk = st.text_input("…or paste your Groq API key", type="password")
        if gk:
            os.environ["GROQ_API_KEY"] = gk
            key_present = True

    st.divider()
    if st.button("🗑️ Reset DB & logs", use_container_width=True):
        bc.clear_logs()
        import importlib.util
        spec = importlib.util.spec_from_file_location("init_db", os.path.join(bc.DATA_DIR, "init_db.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.init_db(bc.DB_PATH)
        st.session_state.pop("history", None)
        get_system.clear()
        st.rerun()
    st.markdown(
        "**Agent roles:**\n"
        "- Classifier → routes the message\n"
        "- Feedback Handler → thanks / creates ticket\n"
        "- Query Handler → returns ticket status")


st.title("Banking Customer Support — Multi-Agent Assistant")

tickets = bc.list_tickets()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tickets", len(tickets))
k2.metric("Unresolved", sum(1 for t in tickets if t["status"] == "Unresolved"))
k3.metric("In Progress", sum(1 for t in tickets if t["status"] == "In Progress"))
k4.metric("Resolved", sum(1 for t in tickets if t["status"] == "Resolved"))

tabs = st.tabs(["💬 Assistant", "🎫 Tickets", "📊 Evaluation", "🧾 Logs & Debugging"])

SCENARIOS = {
    "Positive feedback": ("Thanks for sorting out my net banking login issue!", "Daniel"),
    "Negative feedback": ("My debit card replacement still hasn't arrived.", "Sophea"),
    "Query (ticket status)": ("Could you check the status of ticket 650932?", "Daniel"),
}

# ===========================================================================
# TAB 1 - ASSISTANT (input -> routing -> classification/response/DB)
# ===========================================================================
with tabs[0]:
    st.subheader("Send a customer message")
    st.caption("The Supervisor classifies the message and routes it to the right agent. "
               "Each reply shows the classification, the agent path, and any database action.")

    cS = st.columns(len(SCENARIOS))
    for i, (label, (msg, name)) in enumerate(SCENARIOS.items()):
        if cS[i].button(label, use_container_width=True):
            st.session_state["pending"] = (msg, name)

    with st.form("msg_form", clear_on_submit=False):
        c1, c2 = st.columns([3, 1])
        default_msg, default_name = st.session_state.get("pending", ("", "Customer"))
        message = c1.text_area("Customer message", value=default_msg, height=90,
                               placeholder="e.g. Could you check the status of ticket 650932?")
        customer = c2.text_input("Customer name", value=default_name)
        submitted = st.form_submit_button("Route message ▶", type="primary",
                                          use_container_width=True)

    if not key_present:
        st.info("Add your Groq API key in the sidebar to run the agents.")
    elif submitted and message.strip():
        st.session_state.pop("pending", None)
        system = get_system(model)
        with st.spinner("Agents working…"):
            try:
                out = system.route(message.strip(), customer_name=customer or "Customer")
            except Exception as e:
                out = None
                st.error(f"Error: {e}")
        if out:
            st.markdown(f"**Classification:** :blue[{out['classification']}]  ·  "
                        f"**Route:** {out['agent_path']}")
            st.success(out["response"])
            if out["ticket"]:
                st.markdown("**Database interaction — ticket record:**")
                st.dataframe(pd.DataFrame([out["ticket"]]), use_container_width=True, hide_index=True)
            with st.expander("🧩 Agent step trace"):
                for s in out["steps"]:
                    st.markdown(f"- **{s['agent']}** — {s['detail']}")
            st.session_state.setdefault("history", []).insert(0, out)

    if st.session_state.get("history"):
        st.divider(); st.markdown("**Recent interactions**")
        hist = [{"message": h["message"], "classification": h["classification"],
                 "response": h["response"]} for h in st.session_state["history"][:8]]
        st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 - TICKETS (support database)
# ===========================================================================
with tabs[1]:
    st.subheader("🎫 support_tickets database")
    tdf = pd.DataFrame(bc.list_tickets())
    if tdf.empty:
        st.info("No tickets.")
    else:
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            tdf["status"].value_counts().reindex(bc.STATUSES).plot(
                kind="bar", ax=ax, color=["#E45756", "#F58518", "#54A24B"])
            ax.set_title("Tickets by status", fontweight="bold"); ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=0); plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with right:
            st.markdown("**Update a ticket status** (support-team action)")
            num = st.selectbox("Ticket", tdf["ticket_number"].tolist())
            new_status = st.selectbox("New status", bc.STATUSES)
            if st.button("Update status"):
                bc.update_ticket_status(num, new_status)
                st.success(f"Ticket #{num} updated to {new_status}.")
                st.rerun()

# ===========================================================================
# TAB 3 - EVALUATION
# ===========================================================================
with tabs[2]:
    st.subheader("📊 Model evaluation")
    if not key_present:
        st.info("Add your Groq API key in the sidebar to run evaluation.")
    else:
        system = get_system(model)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Classification accuracy & routing success**")
            if st.button("Run classification evaluation"):
                with st.spinner("Classifying test set…"):
                    ev = bc.evaluate_classification(system)
                st.metric("Accuracy", f"{ev['correct']}/{ev['total']} ({ev['accuracy']:.0%})")
                st.metric("Routing success rate", f"{ev['routing_success_rate']:.0%}")
                st.dataframe(pd.DataFrame(ev["rows"]), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Response quality (QAEvalChain)**")
            if st.button("Run response evaluation"):
                with st.spinner("Grading responses…"):
                    graded, preds = bc.evaluate_responses(system)
                    exs = bc.response_eval_examples()
                correct = 0
                for ex, pred, g in zip(exs, preds, graded):
                    verdict = str(g.get("results", g.get("text", ""))).strip()
                    ok = verdict.upper().startswith("CORRECT"); correct += ok
                    with st.expander(f"{'✅' if ok else '❌'} {ex['query'][:60]}"):
                        st.write(pred["result"]); st.caption(f"Grade: {verdict}")
                st.metric("Response quality", f"{correct}/{len(exs)} ({correct/len(exs):.0%})")

# ===========================================================================
# TAB 4 - LOGS & DEBUGGING
# ===========================================================================
with tabs[3]:
    st.subheader("🧾 Logs & debugging view")
    m = bc.performance_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total actions", m["total_actions"])
    c2.metric("Overall success", f"{m['overall_success_rate']:.0%}" if m["overall_success_rate"] is not None else "—")
    c3.metric("Routing success", f"{m['routing_success_rate']:.0%}" if m["routing_success_rate"] is not None else "—")

    logs = pd.DataFrame(bc.get_logs())
    if logs.empty:
        st.caption("No agent actions logged yet — send a message on the Assistant tab.")
    else:
        st.markdown("**Agent action log** (prompt traces, classification output, ticket actions)")
        st.dataframe(logs[["timestamp", "agent", "input", "output", "success", "detail"]].iloc[::-1],
                     use_container_width=True, hide_index=True, height=300)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        g = logs.groupby("agent")["success"].agg(["sum", "count"])
        g["fail"] = g["count"] - g["sum"]
        g[["sum", "fail"]].plot(kind="bar", stacked=True, ax=ax, color=["#54A24B", "#E45756"])
        ax.set_title("Agent success vs failure", fontweight="bold"); ax.set_ylabel("Actions")
        ax.legend(["success", "failure"]); plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
