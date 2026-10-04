"""Initialize and seed the support_tickets SQLite database for the
Banking Customer Support multi-agent system. Running this file (re)creates
a clean, seeded database."""
import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_tickets.db")

SEED_TICKETS = [
    # ticket_number, customer_name, issue, status, days_ago
    ("650932", "Daniel Roberts", "Net banking login not working", "Resolved", 20),
    ("412087", "Sophea Chan",    "Debit card replacement not received", "In Progress", 8),
    ("778310", "Maria Gomez",    "Incorrect charge on credit card statement", "Unresolved", 3),
    ("205544", "John Carter",    "Unable to transfer funds to external account", "Resolved", 15),
    ("934221", "Aisha Rahman",   "Mobile app crashes on payment screen", "In Progress", 5),
    ("117659", "Raj Patel",      "Loan EMI deducted twice this month", "Unresolved", 1),
    ("560198", "Grace Mwangi",   "Request to increase credit card limit", "Resolved", 30),
    ("883402", "Thomas Weber",   "ATM withdrawal failed but amount debited", "In Progress", 2),
]

STATUSES = ("Unresolved", "In Progress", "Resolved")


def init_db(db_path: str = DB_PATH, reset: bool = True) -> None:
    if reset and os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_number TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            issue         TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'Unresolved',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )
        """
    )
    now = datetime.now()
    for num, name, issue, status, days_ago in SEED_TICKETS:
        ts = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT OR REPLACE INTO support_tickets "
            "(ticket_number, customer_name, issue, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (num, name, issue, status, ts, ts),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ticket_number, customer_name, status FROM support_tickets"
    ).fetchall()
    conn.close()
    print(f"Seeded {len(rows)} tickets into {DB_PATH}")
    for r in rows:
        print(" ", r)
