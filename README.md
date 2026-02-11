# AgentOps AI Platform

A production-grade multi-agent AI system built with **LangChain**, **LangGraph**, and **FastAPI**. Features real-time streaming, semantic memory with vector search, and comprehensive observability.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

- 🤖 **Multi-Agent System**: Orchestrated workflow with Supervisor, Research, Execution, and Evaluator agents
- 🛠️ **Tool Integration**: Web search with DuckDuckGo (easily extensible)
- 📊 **Real-Time Streaming**: Server-Sent Events (SSE) for immediate feedback
- 🧠 **Semantic Memory**: ChromaDB vector store for intelligent memory retrieval
- 🔍 **Observability**: LangSmith and Langfuse integration for full tracing
- 🎯 **Self-Evaluation**: Automatic quality control with retry mechanism
- 🔒 **Security First**: Zero hardcoded secrets, comprehensive .gitignore
- 🚀 **Production Ready**: Clean architecture, type safety, error handling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│                  Real-time Streaming UI                      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────┴────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│              REST API + Streaming Endpoints                  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────┴────────┐       ┌────────┴────────┐
│   LangGraph    │       │  Observability  │
│   Workflow     │       │  (Traces/Logs)  │
└───────┬────────┘       └─────────────────┘
        │
        ├─► Supervisor Agent (Planning)
        ├─► Research Agent (Context Gathering)
        ├─► Execution Agent (Output Generation)
        └─► Evaluator Agent (Quality Control)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **Google Gemini API Key** ([Get Free Key](https://makersuite.google.com/app/apikey))

### Installation

**1. Clone & Navigate**
```bash
git clone <your-repo-url>
cd "AgentOps AI Platorm"
```

**2. Set Up Environment**
```bash
# Copy example env file
cp .env.example .env.local

# Edit with your API key
nano .env.local  # or use your editor
```

Add your Gemini API key:
```bash
GOOGLE_API_KEY="your_actual_key_here"
```

**3. Install Dependencies**
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

**4. Start the Application**
```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend
cd frontend && npm run dev
```

**5. Open in Browser**
```
http://localhost:3000
```

That's it! 🎉

---

## 💻 Usage

### Web Interface

1. Open **http://localhost:3000**
2. Enter your goal (e.g., "Explain vector databases")
3. Click **Execute**
4. Watch real-time streaming output
5. View evaluation score and history

### API Endpoints

#### Execute Task (Streaming)
```bash
curl -X POST 'http://localhost:8000/run?stream=true' \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explain vector databases"}'
```

#### Execute Task (Non-Streaming)
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explain vector databases"}'
```

**Response:**
```json
{
  "final_output": "Vector databases are specialized systems...",
  "evaluation": {
    "passed": true,
    "score": 9,
    "reasons": ["Clear", "Accurate", "Well-structured"]
  },
  "memory_used": false
}
```

#### Get History
```bash
curl http://localhost:8000/history
```

#### Health Check
```bash
curl http://localhost:8000/health
```

**Interactive API Docs:** http://localhost:8000/docs

---

## 📁 Project Structure

```
AgentOps AI Platform/
│
├── backend/                    # FastAPI REST API
│   ├── main.py                # Entry point + health check
│   └── routers/
│       ├── run.py             # Task execution (streaming + non-streaming)
│       └── history.py         # Memory retrieval
│
├── frontend/                   # Next.js 14 UI
│   ├── app/
│   │   ├── page.tsx           # Main page
│   │   ├── layout.tsx         # Root layout
│   │   ├── globals.css        # Styles
│   │   └── components/        # React components
│   │       ├── AgentInput.tsx
│   │       ├── ResultDisplay.tsx
│   │       └── HistoryList.tsx
│   ├── package.json
│   └── next.config.js
│
├── src/agentops_ai_platform/   # Core agent system
│   ├── agents/                 # Agent implementations
│   │   ├── supervisor_agent.py
│   │   ├── research_agent.py
│   │   ├── execution_agent.py
│   │   └── evaluator_agent.py
│   └── graphs/
│       └── main_graph.py      # LangGraph workflow
│
├── memory/                     # Memory system
│   ├── memory_store.py        # JSON storage
│   └── vector_store.py        # ChromaDB vector search
│
├── tools/                      # Agent tools
│   └── web_search.py          # DuckDuckGo integration
│
├── observability/              # Tracing & monitoring
│   ├── langsmith.py           # LangSmith client
│   ├── langfuse.py            # Langfuse client
│   └── trace_utils.py         # Helper utilities
│
├── .env.example               # Environment template ⭐
├── .gitignore                 # Comprehensive protection
├── requirements.txt           # Python dependencies
├── start_backend.sh           # Backend startup script
├── LICENSE                    # MIT License
└── README.md                  # This file
```

> **Note:** `.env.local` (your secrets) is gitignored and must be created locally.

---

## 🤖 Agent Pipeline

### Workflow

```
User Goal → Supervisor → Research → Execution → Evaluator → Response
                ↓           (if needed)      ↓            ↓
           Plan + Tools                  Uses Tools   Validates
```

### Agents

| Agent | Role | Responsibility |
|-------|------|---------------|
| **Supervisor** | Planning | Creates plan, decides if research needed, declares required tools |
| **Research** | Context Gathering | Searches for information (conditional, only if needed) |
| **Execution** | Generation | Produces output, uses tools declared by Supervisor |
| **Evaluator** | Quality Control | Scores output (1-10), validates tool usage, triggers retry if needed |

**Model:** Google Gemini 2.5 Flash (all agents)  
**Retry Logic:** Up to 5 attempts if evaluation fails  
**Memory:** Successful outputs (score ≥8) saved automatically

---

## 🛠️ Tools

### Web Search (DuckDuckGo)

- **No API Key Required** - Free and unlimited
- **Automatic** - Supervisor decides when to use
- **Safe** - Validated schemas, graceful failures
- **Transparent** - Results labeled as `[EXTERNAL INFORMATION]`

### Tool Flow

```
Supervisor declares → Execution uses → Evaluator validates
     ["web_search"]        tool              authorization
```

### Adding New Tools

1. Create tool in `tools/` with Pydantic schema
2. Register in tool registry
3. Test with integration tests
4. Tool automatically available to agents

**Easily extensible for**: APIs, databases, file operations, calculations, etc.

---

## 🧠 Memory System

**Semantic Memory with ChromaDB Vector Search**

| Feature | Details |
|---------|---------|
| **Storage** | Automatic for score ≥ 8 |
| **Search** | Semantic similarity (embeddings) |
| **Retrieval** | Supervisor finds relevant past tasks |
| **Limits** | Max 100 memories, 90-day retention |
| **Format** | JSON + ChromaDB vector index |

**How it works:**
1. High-quality output completed (score ≥ 8)
2. Saved to `memory/memory.json`
3. Embedded and indexed in ChromaDB
4. Future similar tasks find and reuse knowledge

---

## 🔍 Observability

### LangSmith (Optional)

Full tracing of agent execution, LLM calls, and timing.

```bash
# .env.local
LANGSMITH_API_KEY="your_key"
LANGSMITH_PROJECT="agentops-ai-platform"
```

**Dashboard:** https://smith.langchain.com/

### Langfuse (Optional)

Metrics, evaluation scores, and cost tracking.

```bash
# .env.local
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
```

**Dashboard:** https://cloud.langfuse.com/

**Note:** Both are optional. System works without them.

---

## 👨‍💻 Development

### Hot Reload

Both backend and frontend support hot reload for rapid development:

```bash
# Backend (auto-restarts on code changes)
cd backend && uvicorn main:app --reload

# Frontend (auto-refreshes on changes)
cd frontend && npm run dev
```

### Code Quality

```bash
# Python
black src/              # Format
ruff check src/         # Lint

# TypeScript
cd frontend && npm run lint
```

---

## 🚢 Deployment

### Vercel (Frontend)

1. Push your code to GitHub
2. Import project in Vercel dashboard
3. Set environment variables:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.com
   ```
4. Deploy!

### Google Cloud Run (Backend)

```bash
# Deploy to Cloud Run
gcloud run deploy agentops-backend \
  --source . \
  --region us-central1
  
# Set secrets via Secret Manager (recommended)
gcloud secrets create google-api-key --data-file=- <<< "your_key"
```

### Railway / Render (Backend)

Both support direct GitHub deployment:
1. Connect your repository
2. Set environment variables in dashboard
3. Deploy automatically on push

**⚠️ Security: Never commit `.env.local` - use platform secret managers!**

---

## 🔒 Security

**Critical: NEVER commit API keys or secrets!**

### Protected Files (Already in .gitignore)

✅ `.env.local` - Your secrets  
✅ `memory/memory.json` - User data  
✅ `memory/chroma_db/` - Vector database  
✅ All `.env.*` files  

### Before Pushing to GitHub

```bash
# 1. Verify no secrets will be committed
git status
# Should NOT show .env.local, .env, or any secret files

# 2. Double-check .gitignore is working
git check-ignore .env.local .env
# Should output the file names (means they're ignored)

# 3. Search for accidental API keys in staged files
git diff --cached | grep -iE "(AIza|sk-|pk-lf|sk-lf)"
# Should return nothing

# 4. Safe to push
git add . && git commit -m "Your message" && git push
```

### Deployment Secrets

| Platform | Secret Management |
|----------|------------------|
| **Vercel** | Dashboard → Settings → Environment Variables |
| **GCP** | Secret Manager (`gcloud secrets create`) |
| **Railway** | Dashboard → Variables |
| **Render** | Dashboard → Environment |

### If Secrets Are Leaked

1. **Immediately** revoke/rotate ALL exposed keys
2. Check Google API Console for unauthorized usage
3. Update `.env.local` with new keys
4. Consider using [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) to remove from git history

### Security Checklist

- [x] `.env.example` has placeholders only
- [x] `.gitignore` blocks all secret files  
- [x] No hardcoded API keys in code
- [ ] Rotate keys monthly (recommended)
- [ ] Use platform secret managers in production

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Streaming First Byte** | 3-7 seconds |
| **Total Execution** | 10-20 seconds |
| **Cost per Task** | ~$0.002 (Gemini 2.5 Flash) |
| **Monthly Cost** (1000 tasks) | ~$2 |

---

## 🐛 Troubleshooting

### Backend Won't Start

```bash
# Check port 8000
lsof -i :8000

# Kill process if occupied
lsof -ti:8000 | xargs kill -9

# Restart
./start_backend.sh
```

### Missing GOOGLE_API_KEY

1. Verify `.env.local` exists
2. Check API key format: `GOOGLE_API_KEY="AIza..."`
3. Restart backend

### Frontend Not Loading

```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

### Streaming Issues

Streaming auto-falls-back to non-streaming. Check:
- Browser console (F12)
- Backend terminal logs
- Network tab for errors

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Framework**: [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLM**: [Google Gemini](https://ai.google.dev/)
- **Observability**: [LangSmith](https://smith.langchain.com/) & [Langfuse](https://langfuse.com/)
- **Frontend**: [Next.js](https://nextjs.org/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)

---

## 📚 Resources

- **API Documentation**: http://localhost:8000/docs
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Gemini API**: https://ai.google.dev/docs

---

## 💬 Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)

---

<div align="center">

**Built with ❤️ for production-grade agentic AI systems**

⭐ Star this repo if you find it helpful!

</div>
