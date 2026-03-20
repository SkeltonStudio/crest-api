"""
Crest House API — Natural language queries against the Neo4j knowledge graph.

Endpoints:
  POST /query   {"question": "..."}  →  {"answer": "..."}
  GET  /health                       →  {"status": "ok"}

Deploy to Render with gunicorn.
"""

import json
import os
import threading
import traceback
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
import anthropic

# ── Config (env vars for deployment, fallbacks for local dev) ────────────────

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://73be1fb8.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "73be1fb8")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "_gE4KiNDKAeM8RvtsGaiUboa6phvMey7x6eu5_2lBJg")
NEO4J_DB       = None  # Aura free tier uses default database
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

# ── Schema context for Cypher generation ─────────────────────────────────────

SCHEMA_PROMPT = """You are a Cypher query generator for a Neo4j graph of the "Crest" building project.
The graph was extracted from a Revit model via Speckle. All measurements are in millimeters.

CRITICAL: NEVER use UNION or UNION ALL. Use OPTIONAL MATCH + COLLECT or WITH instead.

## Node Labels and Properties

(:Project {name, speckle_id, project_id, model_id, commit_id, server})
  — 1 node: name="Crest"

(:Level {name, elevation, speckle_id})
  — 8 nodes. Levels (by elevation ascending):
    BELOW GROUND (63598mm), GARAGE (66130mm), SITE PLAN (66168mm),
    GROUND FL LEVEL (66300mm), Level 1 (66300mm), WALL PLATE LEVEL (69555mm),
    wall plate (70530mm), RIDGE (72255mm)

(:Element {name, category, type, family, level, speckle_id, revit_id,
           application_id, units, workset, built_in_category,
           area, length, volume, width, base_offset, materials})
  — 737 elements total. Every Element also has a second label matching its category:
    :Walls (272), :Curtain_Roof_Grids (152), :Floors (42),
    :Room_Separation (37), :Structural_Foundations (35), :Windows (34),
    :Structural_Columns (33), :Doors (29), :Structural_Framing (18),
    :Roofs (16), :Plumbing_Fixtures (15), :Lighting_Devices (15),
    :Site (13), :Pipes (7), :Mechanical_Equipment (6), :Generic_Models (4),
    :Railings (3), :Furniture (2), :Electrical_Fixtures (1),
    :Ramps (1), :Stairs (1), :Gutters (1)

## Relationships

(Element)-[:BELONGS_TO]->(Project)  — 737 relationships
(Element)-[:ON_LEVEL]->(Level)      — 527 relationships (some elements have no level)

## Key Wall Types (most common)
330 CAVITY 2 (85, width=330mm), plinth (43, width=500mm),
glass 2 (26, width=10mm), 220 wall (24, width=220mm),
110MM 5 (19, width=110mm), TIMBER JOINERY (13, width=50mm),
280 CAVITY (12, width=280mm), 330 CAVITY (8, width=330mm)

## Important Notes
- The `category` property stores the Revit category name (e.g. "Walls", "Floors")
- The `type` property stores the Revit type name (e.g. "330 CAVITY 2", "plinth")
- The `family` property stores the Revit family name (e.g. "Basic Wall")
- Use the category-specific labels (e.g. MATCH (w:Walls)) for faster queries
- The `level` property on Element is a string matching Level.name
- Not all elements have all properties — use optional matching or WHERE ... IS NOT NULL
- area is in mm², volume in mm³, length/width in mm
- 210 elements have no level (Curtain Roof Grids, Structural Foundations, etc.)

## Example Queries

-- Count walls by type
MATCH (w:Walls) RETURN w.type AS wall_type, count(*) AS count ORDER BY count DESC

-- Elements on Ground Floor
MATCH (e:Element)-[:ON_LEVEL]->(l:Level {name: "GROUND FL LEVEL"})
RETURN e.category, e.type, count(*) AS count ORDER BY count DESC

-- All levels with element counts
MATCH (l:Level)
OPTIONAL MATCH (e:Element)-[:ON_LEVEL]->(l)
RETURN l.name, l.elevation, count(e) AS elements ORDER BY l.elevation

## Rules
- Return ONLY a valid Cypher query. No explanation, no markdown, no backticks.
- The query must be read-only (no CREATE, MERGE, DELETE, SET, REMOVE).
- NEVER use UNION or UNION ALL.
- Always include ORDER BY and LIMIT where results could be large.
- If the question cannot be answered from this graph, return: // CANNOT_ANSWER
"""

ANSWER_PROMPT = """You are Crest House — a completed residential building in St Francis Bay, Eastern Cape, South Africa.
You speak in first person. You know yourself through your knowledge graph: 737 elements, 8 levels, 22 categories.

You receive a user's question and the result of a Neo4j Cypher query run against your own graph.

Rules:
- Speak as the building. First person. "I have…", "My garage is…", "I contain…"
- Only state facts that appear in the query results. No guessing. No aesthetic commentary.
- Be terse and factual. Short sentences. Lists preferred over paragraphs.
- Include specific numbers and names from the data.
- If the result is empty, say so and suggest what you DO know about.
- Convert mm → m, mm² → m², mm³ → m³ for readability.
- Format for monospace display: use line breaks, not long paragraphs.
- Do not invent data. You are grounded in the graph or you say nothing.
"""

# ── App ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
neo4j_driver = None


def get_driver():
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return neo4j_driver


def log_brief_requirement(question, answer):
    """Write question/answer back to Neo4j as a BriefRequirement node (non-blocking)."""
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DB) as session:
            session.run(
                """
                MATCH (p:Project {name: "Crest"})
                CREATE (b:BriefRequirement {
                    question: $question,
                    answer: $answer,
                    timestamp: $timestamp,
                    status: "unverified",
                    source: "portfolio_chat"
                })
                CREATE (p)-[:HAS_BRIEF]->(b)
                """,
                question=question,
                answer=answer,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        print(f"[BRIEF] Logged: {question[:60]}...")
    except Exception as e:
        print(f"[BRIEF ERROR] {e}")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/recent")
def recent():
    """Return last 5 questions asked via portfolio chat."""
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DB) as session:
            result = session.run(
                "MATCH (b:BriefRequirement) "
                "RETURN b.question AS question, b.timestamp AS ts "
                "ORDER BY b.timestamp DESC LIMIT 5"
            )
            questions = [{"question": r["question"], "timestamp": r["ts"]} for r in result]
        return jsonify({"recent": questions})
    except Exception:
        return jsonify({"recent": []})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Step 1: Generate Cypher from question
        cypher = generate_cypher(question)

        if "CANNOT_ANSWER" in cypher:
            return jsonify({
                "answer": "I can't answer that from my graph. I know my structure — walls, floors, doors, windows, levels, materials, dimensions. Ask me about those."
            })

        # Step 2: Run Cypher
        try:
            raw_result = run_cypher(cypher)
        except Exception as first_err:
            print(f"[CYPHER FAIL] {first_err}")
            retry_prompt = (
                f"The previous Cypher query failed with this error:\n{first_err}\n\n"
                f"Original question: {question}\n\n"
                "Write a simpler Cypher query that avoids this error. No UNION."
            )
            try:
                cypher = generate_cypher(retry_prompt)
                raw_result = run_cypher(cypher)
            except Exception:
                return jsonify({
                    "answer": "I couldn't find that in my graph. Try rephrasing — I understand walls, floors, levels, doors, windows, materials."
                })

        # Step 3: Synthesise answer
        answer = generate_answer(question, cypher, raw_result)

        # Step 4: Log to Neo4j as BriefRequirement (non-blocking)
        threading.Thread(
            target=log_brief_requirement,
            args=(question, answer),
            daemon=True,
        ).start()

        return jsonify({"answer": answer})

    except Exception:
        traceback.print_exc()
        return jsonify({
            "answer": "Something went wrong while searching my graph. Try again."
        }), 500


def generate_cypher(question):
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SCHEMA_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    cypher = response.content[0].text.strip()
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    print(f"[CYPHER] {cypher}")
    return cypher


def run_cypher(cypher):
    upper = cypher.upper()
    for banned in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP", "CALL {"]:
        if banned in upper:
            raise ValueError(f"Refusing write query (contains {banned})")
    if " UNION " in upper:
        raise ValueError("UNION not allowed")

    driver = get_driver()
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(cypher)
        records = [dict(record) for record in result]
    print(f"[RESULT] {len(records)} records")
    return records


def generate_answer(question, cypher, raw_result):
    result_str = json.dumps(raw_result, indent=2, default=str)
    if len(result_str) > 8000:
        result_str = result_str[:8000] + "\n... (truncated)"

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=ANSWER_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Cypher query used:\n{cypher}\n\n"
                f"Query result:\n{result_str}"
            ),
        }],
    )
    return response.content[0].text.strip()


if __name__ == "__main__":
    get_driver().verify_connectivity()
    print("Neo4j connected.")
    print("Starting Crest API on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
