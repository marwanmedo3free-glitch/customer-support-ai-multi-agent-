"""
MCP server: the ONLY place that touches Postgres.

Safety rules enforced here (never trust the LLM to enforce them):
  * every query is scoped by customer_id
  * refunds re-check eligibility server-side
  * refunds are idempotent
  * refunds always use the DB order total
  * address changes are allowed only while an order is still 'processing'

Run standalone:
    python mcp_server.py
"""

import os
from datetime import date, datetime, timezone
from decimal import Decimal

from dotenv import load_dotenv
import psycopg
from mcp.server.fastmcp import FastMCP
from psycopg.rows import dict_row


# ============================================================
# Load environment variables from .env
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

mcp = FastMCP("support-tools")

REFUND_WINDOW_DAYS = 30


# ============================================================
# Database connection
# ============================================================

def _db():
    """
    Create a PostgreSQL connection using DATABASE_URL.

    DATABASE_URL should come from .env, for example:

    DATABASE_URL="postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require"
    """

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Make sure your .env file contains DATABASE_URL."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
    )


# ============================================================
# Helpers
# ============================================================

def _clean(row):
    """
    Convert PostgreSQL values into JSON-friendly values.
    """

    if row is None:
        return None

    return {
        k: (
            float(v)
            if isinstance(v, Decimal)
            else v.isoformat()
            if isinstance(v, (datetime, date))
            else v
        )
        for k, v in row.items()
    }


def _order(cur, order_id, customer_id):
    """
    Get an order only if it belongs to the authenticated customer.
    """

    cur.execute(
        """
        SELECT *
        FROM orders
        WHERE id = %s
          AND customer_id = %s
        """,
        (order_id, customer_id),
    )

    return cur.fetchone()


def _eligibility(cur, order_id, customer_id):
    """
    Check whether an order is eligible for a refund.
    """

    # --------------------------------------------------------
    # Get order belonging to this customer
    # --------------------------------------------------------

    o = _order(cur, order_id, customer_id)

    if not o:
        return {
            "error": "order_not_found"
        }

    # --------------------------------------------------------
    # Count previous refunds for this customer
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT count(*) AS n
        FROM refunds
        WHERE customer_id = %s
          AND status = 'issued'
        """,
        (customer_id,),
    )

    prior = cur.fetchone()["n"]

    # --------------------------------------------------------
    # Check whether this order was already refunded
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT 1
        FROM refunds
        WHERE order_id = %s
          AND status = 'issued'
        """,
        (order_id,),
    )

    already = cur.fetchone() is not None

    # --------------------------------------------------------
    # Determine eligibility
    # --------------------------------------------------------

    days = None
    eligible = False
    reason = ""

    if already:
        reason = "a refund was already issued for this order"

    elif o["status"] != "delivered" or not o["delivered_at"]:
        reason = "the order has not been delivered yet"

    else:
        days = (
            datetime.now(timezone.utc) - o["delivered_at"]
        ).days

        eligible = days <= REFUND_WINDOW_DAYS

        if not eligible:
            reason = (
                f"it was delivered {days} days ago and "
                f"our return window is {REFUND_WINDOW_DAYS} days"
            )

    return {
        "order_id": order_id,
        "eligible": eligible,
        "reason": reason,
        "amount": float(o["total"]),
        "days_since_delivery": days,
        "prior_refunds": prior,
    }


# ============================================================
# MCP TOOL: Get Order
# ============================================================

@mcp.tool()
def get_order(order_id: int, customer_id: int) -> dict:
    """
    Get status, ETA and address of one of the customer's orders.
    """

    with _db() as conn, conn.cursor() as cur:

        o = _order(
            cur,
            order_id,
            customer_id,
        )

        return (
            _clean(o)
            if o
            else {"error": "order_not_found"}
        )


# ============================================================
# MCP TOOL: Update Address
# ============================================================

@mcp.tool()
def update_address(
    order_id: int,
    customer_id: int,
    new_address: str,
) -> dict:
    """
    Change the delivery address.

    Only allowed while the order is still processing.
    """

    with _db() as conn, conn.cursor() as cur:

        # ----------------------------------------------------
        # Get the customer's order
        # ----------------------------------------------------

        o = _order(
            cur,
            order_id,
            customer_id,
        )

        if not o:
            return {
                "error": "order_not_found"
            }

        # ----------------------------------------------------
        # Only processing orders can have their address changed
        # ----------------------------------------------------

        if o["status"] != "processing":
            return {
                "error": "order_already_shipped"
            }

        # ----------------------------------------------------
        # Update address
        # ----------------------------------------------------

        clean_address = new_address.strip()[:300]

        cur.execute(
            """
            UPDATE orders
            SET address = %s
            WHERE id = %s
              AND customer_id = %s
            """,
            (
                clean_address,
                order_id,
                customer_id,
            ),
        )

        return {
            "updated": True,
            "order_id": order_id,
            "address": clean_address,
        }


# ============================================================
# MCP TOOL: Check Refund Eligibility
# ============================================================

@mcp.tool()
def check_refund_eligibility(
    order_id: int,
    customer_id: int,
) -> dict:
    """
    Check whether an order can be refunded under policy.

    This operation is read-only.
    """

    with _db() as conn, conn.cursor() as cur:

        return _eligibility(
            cur,
            order_id,
            customer_id,
        )


# ============================================================
# MCP TOOL: Create Refund
# ============================================================

@mcp.tool()
def create_refund(
    order_id: int,
    customer_id: int,
    reason: str,
    idempotency_key: str,
) -> dict:
    """
    Issue a full refund.

    Idempotent per key.
    Re-validates eligibility server-side.
    """

    with _db() as conn, conn.cursor() as cur:

        # ----------------------------------------------------
        # Check idempotency key
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT *
            FROM refunds
            WHERE idempotency_key = %s
            """,
            (idempotency_key,),
        )

        existing = cur.fetchone()

        if existing:
            return {
                **_clean(existing),
                "replayed": True,
            }

        # ----------------------------------------------------
        # Re-check refund eligibility
        # ----------------------------------------------------

        elig = _eligibility(
            cur,
            order_id,
            customer_id,
        )

        if "error" in elig:
            return elig

        if not elig["eligible"]:
            return {
                "error": "not_eligible",
                "reason": elig["reason"],
            }

        # ----------------------------------------------------
        # Create refund
        # ----------------------------------------------------

        cur.execute(
            """
            INSERT INTO refunds (
                order_id,
                customer_id,
                amount,
                reason,
                idempotency_key
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                order_id,
                customer_id,
                elig["amount"],
                reason[:500],
                idempotency_key,
            ),
        )

        refund = cur.fetchone()

        return _clean(refund)


# ============================================================
# MCP TOOL: Create Ticket
# ============================================================

@mcp.tool()
def create_ticket(
    customer_id: int,
    summary: str,
) -> dict:
    """
    Open a ticket for a human support agent.
    """

    with _db() as conn, conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO tickets (
                customer_id,
                summary
            )
            VALUES (%s, %s)
            RETURNING id
            """,
            (
                customer_id,
                summary[:1000],
            ),
        )

        ticket_id = cur.fetchone()["id"]

        return {
            "ticket_id": ticket_id
        }


# ============================================================
# Start MCP server
# ============================================================

if __name__ == "__main__":
    mcp.run()