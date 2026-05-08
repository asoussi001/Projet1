# Tech Office Cockpit

**Copilote Intelligent pour l'Architecture d'Entreprise**

Plateforme d'analyse et de gouvernance des projets Jira, enrichie par un pipeline RAG
(Retrieval-Augmented Generation) croisant tickets, livrables PDF et référentiels
d'architecture internes.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Modules fonctionnels](#modules-fonctionnels)
- [Stack technologique](#stack-technologique)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Déploiement](#déploiement)
- [Contribuer](#contribuer)

---

## Vue d'ensemble

Le **Tech Office Cockpit** est une application de gouvernance intelligente qui :

- **Extrait** les tickets Jira (Epics, Stories, Bugs, Tasks) via l'API Atlassian
- **Ingère** les documents internes (PDF, Word) — notes de service, DAT, ADR, LLD/HLD
- **Indexe** sémantiquement ces documents dans une base vectorielle (Qdrant)
- **Analyse** via LLM l'alignement architectural, la dette technique et les failles de sécurité
- **Orchestre** un agent conversationnel Rovo Tech Office pour des requêtes en langage naturel

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Tech Office Cockpit                           │
│                                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Jira API   │    │  Rovo Agent  │    │  Document Store       │  │
│  │  Connector  │    │  Connector   │    │  (PDF/Word/ADR/DAT)   │  │
│  └──────┬──────┘    └──────┬───────┘    └──────────┬────────────┘  │
│         │                  │                        │               │
│         └──────────────────┴────────────────────────┘               │
│                            │                                         │
│                    ┌───────▼────────┐                               │
│                    │   Data Layer   │                               │
│                    │  PostgreSQL    │                               │
│                    │  + Qdrant VDB  │                               │
│                    └───────┬────────┘                               │
│                            │                                         │
│              ┌─────────────┼─────────────┐                         │
│              │             │             │                           │
│    ┌─────────▼──┐  ┌───────▼──┐  ┌──────▼──────────┐             │
│    │Architectural│  │Governance│  │   RAG Pipeline  │             │
│    │ Alignment  │  │Remediation│  │  LangChain+LLM  │             │
│    └─────────┬──┘  └───────┬──┘  └──────┬──────────┘             │
│              └─────────────┴─────────────┘                          │
│                            │                                         │
│                    ┌───────▼────────┐                               │
│                    │  FastAPI REST  │                               │
│                    │  + WebSocket   │                               │
│                    └───────┬────────┘                               │
│                            │                                         │
│                    ┌───────▼────────┐                               │
│                    │  Streamlit UI  │                               │
│                    │  Dashboard     │                               │
│                    └────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

Voir [docs/architecture/HLD.md](docs/architecture/HLD.md) pour le design détaillé.

---

## Modules fonctionnels

| Module | Description |
|--------|-------------|
| `architectural_alignment` | Croise les Epics/Stories Jira avec les DAT/ADR pour détecter les écarts |
| `governance_remediation` | Analyse la dette technique et propose des plans de remédiation priorisés |
| `rovo_agent` | Agent conversationnel Rovo pour requêtes en langage naturel |
| `rag_pipeline` | Pipeline RAG : ingestion PDF → chunking → embedding → retrieval → génération |

---

## Stack technologique

| Couche | Technologie |
|--------|-------------|
| Langage | Python 3.11+ |
| Orchestration LLM | LangChain 0.2+ / LlamaIndex |
| LLM Backend | Anthropic Claude 3.5 Sonnet (via API) ou Azure OpenAI |
| Vector DB | Qdrant (self-hosted ou Cloud) |
| Base relationnelle | PostgreSQL 16 |
| OCR | Unstructured.io + Tesseract |
| API REST | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| Orchestration K8s | AKS / Kubernetes 1.28+ |
| Monitoring | Prometheus + Grafana |
| Traçabilité LLM | LangSmith / Langfuse |

---

## Prérequis

- Python 3.11+
- Docker 24+ et Docker Compose v2
- Accès à une instance Jira Cloud ou Jira Data Center
- Token API Jira (PAT ou OAuth 2.0)
- Clé API pour le LLM (Anthropic ou Azure OpenAI)
- Instance Qdrant (locale via Docker ou cloud)

---

## Installation

```bash
# Cloner le dépôt
git clone <repo-url> && cd tech-office-cockpit

# Créer l'environnement virtuel
python -m venv .venv && source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Copier et renseigner les variables d'environnement
cp .env.example .env
# Éditer .env avec vos credentials

# Démarrer les services d'infrastructure
docker compose -f docker/docker-compose.yml up -d qdrant postgres
```

---

## Configuration

Toutes les configurations passent par des variables d'environnement (voir `.env.example`).
Les secrets ne doivent jamais être commités.

```bash
# Vérifier la configuration
python -m src.config.settings --validate
```

---

## Utilisation

```bash
# Lancer l'API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Lancer le dashboard Streamlit
streamlit run src/ui/dashboard.py

# Exécuter le PoC complet
python poc/poc_jira_rag.py --project MY_PROJECT --issue ARCH-123
```

---

## Déploiement

```bash
# Build et push de l'image
docker build -t tech-office-cockpit:latest -f docker/Dockerfile .

# Déploiement Kubernetes
kubectl apply -f k8s/
```

---

## Contribuer

1. Créer une branche feature : `git checkout -b feature/mon-module`
2. Lancer les tests : `pytest tests/ -v`
3. Soumettre une PR avec description d'impact architectural
