# LLM Cost Autopilot

An intelligent LLM routing layer that sits in front of multiple providers, analyzes each request's complexity, and routes it to the cheapest model capable of handling it — without sacrificing quality.

**Baseline result: 66% cost reduction on 30-request load test vs all-GPT-4o routing.**

---

## What it does

Most apps send every prompt to the same model regardless of complexity. A question like "what is the capital of France?" costs the same as a multi-step reasoning task. That's wasteful.

LLM Cost Autopilot fixes that by:

1. Classifying each incoming prompt into a complexity tier (simple / moderate / complex)
2. Routing it to the cheapest model that can handle that tier
3. Verifying quality async using GPT-4o as a judge
4. Auto-escalating to a higher tier if the cheap model underperforms
5. Logging every decision so you can see exactly where money is saved

---

## Stack

- **Router:** FastAPI (async)
- **Providers:** OpenAI (GPT-4o, GPT-4o Mini), Anthropic (Claude Sonnet, Claude Haiku), Ollama (Llama 3.2 local)
- **Classifier:** Scikit-learn (logistic regression + random forest)
- **Quality Verification:** LLM-as-judge using GPT-4o
- **Logging:** SQLite + structured JSON
- **Dashboard:** Streamlit
- **Containerization:** Docker + docker-compose

---

## Architecture

```
Incoming Request
      │
      ▼
Complexity Classifier (scikit-learn)
      │
      ├── Tier 1 (simple)   → Llama 3.2 (free, local)
      ├── Tier 2 (moderate) → GPT-4o Mini or Claude Haiku
      └── Tier 3 (complex)  → Claude Sonnet or GPT-4o
                                    │
                                    ▼
                            Response to Caller
                                    │
                                    ▼
                    Async Quality Verifier (GPT-4o judge)
                                    │
                            Log + escalate if needed
```

---

## Day 1 Baseline Results

30 requests across 3 models, same prompts:

| Model | Total Cost | Avg Latency | Success |
|-------|-----------|-------------|---------|
| GPT-4o | $0.017990 | 4676ms | 10/10 |
| GPT-4o Mini | $0.000756 | 5766ms | 10/10 |
| Llama 3.2 (Local) | $0.000000 | 25560ms | 10/10 |

- **Actual cost (mixed routing):** $0.018746
- **Hypothetical all-GPT-4o cost:** $0.054830
- **Savings:** $0.036084 (~66%)

---

## Project Structure

```
llm-cost-autopilot/
├── backend/
│   ├── app.py                    # FastAPI main app
│   ├── router.py                 # Core routing logic
│   ├── classifier.py             # Complexity classifier
│   ├── verifier.py               # Async quality verifier
│   ├── providers/
│   │   ├── base.py               # Unified provider interface
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── ollama_provider.py
│   ├── models/
│   │   ├── registry.py           # ModelConfig dataclass + pricing
│   │   └── routing_config.yaml   # Tier-to-model mapping
│   ├── db/
│   │   ├── database.py           # SQLite setup
│   │   └── logger.py             # Request logging
│   ├── dashboard/
│   │   └── app.py                # Streamlit dashboard
│   ├── data/
│   │   ├── training_prompts.json # 200+ labeled prompts
│   │   └── classifier.pkl        # Trained model
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup

**1. Clone and create virtual environment**
```bash
git clone https://github.com/vaibhav-badoliasoft/llm-cost-autopilot.git
cd llm-cost-autopilot
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**2. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**3. Configure environment**
```bash
copy .env.example .env
# Add your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

**4. Start Ollama**
```bash
ollama serve
ollama pull llama3.2
```

**5. Run Day 1 test**
```bash
python test_day1.py
```

---

## Build Progress

- [x] Provider abstraction + model registry

---

## Author

Vaibhav Saini — [GitHub](https://github.com/vaibhav-badoliasoft) · [Portfolio](https://vaibhavsainiportfolio.vercel.app)
