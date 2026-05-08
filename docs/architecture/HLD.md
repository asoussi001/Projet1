# High-Level Design — Tech Office Cockpit

**Version :** 1.0  
**Date :** 2026-05-08  
**Auteur :** Architecture d'Entreprise / Tech Office  
**Statut :** Draft

---

## 1. Contexte et Objectifs

### 1.1 Problème adressé

Les équipes du Tech Office gèrent un portefeuille de projets complexe (Core Banking, Cloud
Migration, API Gateway, …) dans Jira. Les enjeux actuels sont :

- Absence de visibilité croisée entre les tickets Jira et les référentiels d'architecture
- Détection tardive des dérives architecturales et des non-conformités
- Suivi manuel et coûteux de la dette technique
- Difficultés à corréler les livrables projets (DAT, LLD, rapports d'audit) avec les exigences

### 1.2 Solution proposée

Un **Cockpit Tech Office** : plateforme intelligente combinant extraction Jira, ingestion RAG de
documents internes, et agents LLM pour automatiser l'analyse de conformité et la gouvernance.

---

## 2. Architecture Globale

### 2.1 Vue Layered

```
┌────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — PRÉSENTATION                                                     │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │  Streamlit       │  │  FastAPI REST /      │  │  Rovo Agent          │  │
│  │  Dashboard       │  │  WebSocket API       │  │  (Atlassian UI)      │  │
│  └──────────────────┘  └─────────────────────┘  └──────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — ORCHESTRATION & AGENTS                                           │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │  LangChain Agent     │  │  RAG Chain       │  │  Governance Engine  │  │
│  │  Orchestrator        │  │  (Retrieval +    │  │  (Scoring +         │  │
│  │                      │  │   Generation)    │  │   Remediation)      │  │
│  └──────────────────────┘  └──────────────────┘  └─────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — TRAITEMENT & INDEXATION                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Pipeline d'Ingestion Documents                                        │ │
│  │  OCR (Unstructured) → Chunking → Embedding → Qdrant                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — DONNÉES                                                           │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  PostgreSQL   │  │  Qdrant        │  │  Redis Cache │  │  MinIO     │  │
│  │  (Metadata,   │  │  Vector DB     │  │  (Sessions,  │  │  (PDF      │  │
│  │   Audit logs) │  │  (Embeddings)  │  │   LLM cache) │  │   Storage) │  │
│  └───────────────┘  └────────────────┘  └──────────────┘  └────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — CONNECTEURS (Sources)                                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐   │
│  │  Jira Cloud /  │  │  Rovo API      │  │  Système de Fichiers /     │   │
│  │  Data Center   │  │  (Knowledge    │  │  SharePoint / Confluence   │   │
│  │  REST API v3   │  │   Graph)       │  │  (Documents internes)      │   │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flux de données principaux

#### Flux 1 — Analyse d'alignement architectural (synchrone)
```
User Request
    │
    ▼
FastAPI /analyze/alignment
    │
    ├──► JiraConnector.get_issue(issue_key)
    │         └──► Jira REST API v3
    │
    ├──► JiraConnector.get_attachments(issue_key)
    │         └──► Download PDF → MinIO
    │
    ├──► RAGPipeline.retrieve(query=epic_description, collection="architecture")
    │         └──► Qdrant Similarity Search (top-k=10)
    │
    └──► LLMChain.invoke(issue_data + retrieved_chunks)
              └──► Claude 3.5 Sonnet / Azure OpenAI
                        └──► AlignmentReport (JSON)
```

#### Flux 2 — Ingestion de documents (asynchrone, batch)
```
Document Upload (PDF/Word)
    │
    ▼
Celery Task Queue
    │
    ├──► Unstructured.io → extraction texte + OCR si scan
    ├──► RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    ├──► EmbeddingModel.embed_documents() → vectors[1536]
    └──► Qdrant.upsert(collection, vectors, metadata)
              └──► PostgreSQL: document_index table (audit trail)
```

#### Flux 3 — Agent Rovo Tech Office (conversationnel)
```
User NL Query
    │
    ▼
RovoAgent / LangChain ReAct Agent
    │
    ├──[Tool: search_jira_projects] ──► Jira JQL API
    ├──[Tool: search_architecture_docs] ──► Qdrant RAG
    ├──[Tool: get_delivery_status] ──► Jira + PostgreSQL
    └──[Tool: generate_compliance_report] ──► LLM Chain
              │
              ▼
         Structured Response (Markdown + JSON)
```

---

## 3. Choix Technologiques — Décision et Justification

### 3.1 LLM Backend

| Option | Avantages | Inconvénients | Décision |
|--------|-----------|---------------|----------|
| **Anthropic Claude 3.5 Sonnet** | Contexte 200k tokens, excellent sur tâches analytiques, API stable | Coût par token, dépendance cloud | **RECOMMANDÉ** pour production |
| Azure OpenAI GPT-4o | Conformité Azure, SLA entreprise | Latence variable, nécessite Azure abonnement | Alternatif si contrainte Azure |
| Mistral Large (self-hosted) | On-premise, pas de fuite données | Infrastructure GPU lourde, moins performant | Pour données ultra-sensibles |

**Critique architecturale :** L'abstraction LangChain permet de switcher de provider sans
refactoring applicatif. Prévoir un `LLMProviderFactory` dès le départ.

### 3.2 Vector Database

| Option | Avantages | Inconvénients | Décision |
|--------|-----------|---------------|----------|
| **Qdrant** | Performance, filtrage payload riche, Rust, K8s-ready | Moins mature que Pinecone | **RETENU** |
| Milvus | Très scalable, multi-tenant | Complexité opérationnelle élevée | Si >10M vecteurs |
| pgvector | Simplicité (même BDD) | Performance limitée au-delà de 1M vecteurs | Dev/test uniquement |
| Pinecone | SaaS managé, simple | Coût élevé, vendor lock-in, données hors SI | Exclu (gouvernance) |

**Critique :** Qdrant est le bon choix pour un périmètre initial de 100K à 5M vecteurs avec
filtrage métadonnées (projet, type_doc, version). Migrer vers Milvus si le corpus dépasse 50M.

### 3.3 Embedding Model

| Option | Dimension | Avantages |
|--------|-----------|-----------|
| `text-embedding-3-large` (OpenAI) | 3072 | Meilleure qualité, multilingue |
| `text-embedding-3-small` (OpenAI) | 1536 | Bon rapport qualité/coût |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Local, gratuit, rapide |
| **`intfloat/multilingual-e5-large`** | 1024 | **RECOMMANDÉ** : multilingue FR/EN, local |

**Critique :** Les documents internes en français nécessitent un modèle multilingue. Éviter
les embeddings OpenAI pour les documents sensibles (fuite de données).

### 3.4 OCR et Extraction de Documents

```
Unstructured.io (open-source)
    ├── Stratégie "hi_res" pour PDFs scannés (via Tesseract + detectron2)
    ├── Stratégie "fast" pour PDFs natifs
    └── Extraction des tableaux et diagrammes (via pdfplumber)
```

**Critique :** Pour les DAT contenant des schémas Visio/Draw.io, prévoir un preprocessing
spécifique. Les images de diagrammes doivent être décrites via un modèle vision (Claude Vision).

---

## 4. Modèle de Données

### 4.1 PostgreSQL — Tables principales

```sql
-- Index des documents ingérés
CREATE TABLE document_index (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    source_type     TEXT NOT NULL,  -- 'DAT' | 'ADR' | 'NOTE_SERVICE' | 'LLD' | 'HLD'
    project_key     TEXT,           -- Clé Jira associée
    qdrant_collection TEXT NOT NULL,
    chunk_count     INTEGER,
    embedding_model TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    checksum        TEXT NOT NULL,  -- SHA256 pour déduplication
    metadata        JSONB
);

-- Cache des analyses LLM (évite re-calculs coûteux)
CREATE TABLE analysis_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key       TEXT UNIQUE NOT NULL,  -- hash(issue_key + doc_versions)
    analysis_type   TEXT NOT NULL,         -- 'alignment' | 'governance' | 'compliance'
    result          JSONB NOT NULL,
    llm_model       TEXT NOT NULL,
    tokens_used     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ
);

-- Audit trail des analyses
CREATE TABLE analysis_audit (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_key       TEXT NOT NULL,
    analysis_type   TEXT NOT NULL,
    triggered_by    TEXT NOT NULL,  -- user ou 'scheduler'
    status          TEXT NOT NULL,  -- 'success' | 'error' | 'partial'
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.2 Qdrant — Collections

| Collection | Contenu | Metadata clés |
|------------|---------|---------------|
| `architecture_docs` | DAT, HLD, LLD, ADR | project, version, doc_type, date |
| `governance_docs` | Notes de service, COBIT, ITIL | category, priority, effective_date |
| `jira_tickets` | Descriptions enrichies des tickets | issue_key, project, type, status |
| `deliverables` | Livrables PDF attachés aux tickets | issue_key, filename, upload_date |

---

## 5. Considérations de Sécurité et Gouvernance

### 5.1 Gestion des secrets
- Tous les secrets via **HashiCorp Vault** (ou Azure Key Vault)
- Rotation automatique des tokens Jira
- Jamais de secrets dans les variables d'environnement du conteneur en prod (utiliser K8s Secrets + Vault Agent Injector)

### 5.2 Isolation des données
- Les embeddings de documents confidentiels doivent être dans des collections Qdrant séparées
- Contrôle d'accès par `project_key` via PostgreSQL Row-Level Security (RLS)
- Les PDFs bruts stockés dans MinIO avec chiffrement at-rest (AES-256)

### 5.3 Traçabilité LLM
- Chaque appel LLM loggé dans `analysis_audit` avec `tokens_used`
- Intégration **Langfuse** pour observabilité complète des chains LangChain
- Alertes si dérive de coûts (tokens/jour > seuil)

---

## 6. Scalabilité et Haute Disponibilité

### 6.1 Composants stateless (scalables horizontalement)
- API FastAPI : 3 réplicas minimum derrière un Ingress K8s
- Workers Celery (ingestion) : autoscaling HPA basé sur la longueur de la queue Redis

### 6.2 Composants stateful (réplication)
- Qdrant : mode cluster (3 nœuds, replication_factor=2)
- PostgreSQL : Patroni + PgBouncer (connection pooling)
- Redis : Sentinel ou Redis Cluster

### 6.3 SLA cibles
| Composant | Disponibilité cible | RTO | RPO |
|-----------|--------------------|----|-----|
| API REST | 99.9% | 5 min | 1 min |
| Qdrant | 99.5% | 15 min | 1 h |
| PostgreSQL | 99.9% | 10 min | 5 min |
| Pipeline ingestion | Best effort | 30 min | 24 h |

---

## 7. Plan de Migration et Phases

### Phase 1 — MVP (6 semaines)
- [ ] Connecteur Jira (lecture tickets + pièces jointes)
- [ ] Pipeline ingestion PDF basique (Unstructured + pgvector)
- [ ] Module alignement architectural (LLM simple)
- [ ] API REST basique + UI Streamlit

### Phase 2 — Production (10 semaines)
- [ ] Migration vers Qdrant cluster
- [ ] Agent Rovo complet avec tools LangChain
- [ ] Module gouvernance + remédiation
- [ ] Déploiement AKS + monitoring Prometheus/Grafana

### Phase 3 — Enrichissement (continu)
- [ ] Intégration Confluence (Knowledge Base)
- [ ] Fine-tuning embedding model sur corpus interne
- [ ] Alerting proactif (webhook Jira → analyse auto)
- [ ] Tableau de bord exécutif (KPIs Architecture)
