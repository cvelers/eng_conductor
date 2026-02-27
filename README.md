# Eurocodes Chatbot — EC3 Grounded Intelligence

A single-page ChatGPT-like engineering assistant that answers questions about Eurocode 3 (steel structures) using **only** information from a local clause database and modular MCP calculator tools. Every claim is grounded with clause citations; every calculation traces back to a source.

## Architecture

```
┌────────────┐    ┌─────────────────────────────────┐    ┌──────────────┐
│  Chat UI   │───▶│  Central Intelligence           │───▶│  13 MCP      │
│  (vanilla  │    │  Orchestrator                    │    │  Calculator  │
│   JS)      │◀───│  (Gemini / configurable LLM)     │◀───│  Tools       │
└────────────┘    │                                  │    └──────────────┘
                  │  Plan → Retrieve → Execute →     │
                  │  Compose (grounded narrative)     │    ┌──────────────┐
                  │                                  │───▶│  EC3 Clause  │
                  └─────────────────────────────────┘    │  Database    │
                                                         └──────────────┘
```

**State machine:** Intake → Plan → Input Resolution → Retrieval → Tools → Compose → Output

## Features

- **Collapsible thinking** — ChatGPT-style: reasoning flow graph expands/collapses on click
- **13 modular MCP tools** — building-block calculators (beams, bolts, welds, buckling, units...)
- **Developer mode** — toggle to show a tool writer that generates new MCP tools from database clauses
- **Agentic retrieval** — iterative lexical search with LLM-driven query refinement
- **Grounded responses** — every answer cites clauses and tool references
- **Markdown rendering** — responses render with proper formatting, bold results, collapsible sections
- **Streaming** — real-time flow graph updates + chunked answer delivery
- **Example prompts** — clickable starter queries on the welcome screen

## Tools (13 total)

| Tool | Description | Reference |
|------|-------------|-----------|
| `section_classification_ec3` | I/H section class (1-4) | EC3 5.5.2 |
| `member_resistance_ec3` | M_Rd, N_Rd, V_Rd | EC3 6.2.4-6 |
| `interaction_check_ec3` | Axial + bending interaction | EC3 6.2.9 |
| `ipe_moment_resistance_ec3` | IPE-specific M_Rd | EC3 6.2.5 |
| `simple_beam_calculator` | Moment, shear, deflection (SS beam) | Beam theory |
| `cantilever_beam_calculator` | Moment, shear, deflection (cantilever) | Beam theory |
| `steel_grade_properties` | fy, fu, ε lookup | EC3 Table 3.1 |
| `effective_length_ec3` | Buckling length factor k | EC3 BB.1 |
| `column_buckling_ec3` | Nb,Rd with χ and λ̄ | EC3 6.3.1 |
| `bolt_shear_ec3` | Bolt shear resistance Fv,Rd | EC3-1-8 Table 3.4 |
| `weld_resistance_ec3` | Fillet weld Fw,Rd | EC3-1-8 4.5.3.3 |
| `deflection_check` | SLS deflection check | EC0 A1.4.3 |
| `unit_converter` | Engineering unit conversion | — |

All tools are modular Python scripts under `tools/mcp/`, each with Pydantic validation and JSON I/O.

## Quick Start

```bash
# 1. Create venv and install deps
make venv

# 2. Configure LLM keys (copy and edit)
cp .env.example .env
# Edit .env with your API keys

# 3. Run
make run
# Open http://localhost:8000

# 4. Test
make test
```

## Configuration (.env)

```ini
# Orchestrator (default: Gemini)
ORCHESTRATOR_PROVIDER=gemini
ORCHESTRATOR_MODEL=gemini-3.1-pro
ORCHESTRATOR_API_KEY=your-key-here

# Search agent (default: Kimi via OpenRouter)
SEARCH_PROVIDER=openrouter
SEARCH_MODEL=moonshotai/kimi-k2.5
SEARCH_API_KEY=your-key-here
SEARCH_DECOMPOSE_MAX_TOKENS=1200

# Feature flags
AGENTIC_SEARCH_ENABLED=true
RECURSIVE_RETRIEVAL_ENABLED=false
EMBEDDINGS_ENABLED=false
```

## Example Prompts

**Conceptual:**
- "Explain EC3 section classification rules for I-sections"
- "What are the buckling curves and when is each used?"

**Computational:**
- "Given IPE300, S355, what is the bending resistance?"
- "Simply supported beam, 6m span, 15 kN/m UDL — max moment and deflection?"
- "Calculate bolt shear resistance for 4× M20 grade 8.8 bolts"
- "Column buckling check: IPE300, S355, 5m length, pinned-pinned"
- "Fillet weld resistance: 5mm throat, 200mm length, S355"

**Developer mode:**
- Toggle "Dev Mode" in the top bar to access the tool writer
- Describe a new tool and the system generates MCP-compatible Python code grounded in database clauses

## Project Structure

```
eng_conductor/
├── backend/
│   ├── app.py                      # FastAPI application
│   ├── config.py                   # Settings from .env
│   ├── orchestrator/engine.py      # Central Intelligence Orchestrator
│   ├── retrieval/agentic_search.py # Iterative clause retrieval
│   ├── tools/runner.py             # MCP tool subprocess executor
│   ├── tools/writer.py             # Dev mode tool generator
│   ├── llm/                        # LLM provider abstraction
│   └── utils/                      # Parsing, citations
├── frontend/
│   ├── index.html                  # Single-page chat UI
│   ├── app.js                      # Streaming, flow graph, dev mode
│   └── styles.css                  # Dark theme, responsive
├── tools/
│   ├── tool_registry.json          # 13 registered tools
│   └── mcp/                        # All calculator scripts
├── data/
│   ├── document_registry.json
│   └── ec3/                        # EC3 clause data (JSON)
└── tests/
```
