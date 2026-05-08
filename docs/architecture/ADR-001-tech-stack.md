# ADR-001 : Choix du Stack Technologique — Tech Office Cockpit

**Date :** 2026-05-08  
**Statut :** Accepté  
**Décideurs :** Architecte d'Entreprise, Lead Tech Office

---

## Contexte

Sélection du stack pour la première version production du Cockpit Tech Office.
Contraintes : budget LLM maîtrisé, données sensibles (ne pas sortir du SI), équipe Python.

## Décision

| Composant | Choix | Alternatives rejetées |
|-----------|-------|-----------------------|
| LLM principal | Anthropic Claude 3.5 Sonnet | GPT-4o (Azure), Gemini Pro |
| Embedding | `multilingual-e5-large` (local) | `text-embedding-3-large` (OpenAI) |
| Orchestration | LangChain 0.2 + LCEL | LlamaIndex, Haystack |
| Vector DB | Qdrant | Milvus, pgvector, Pinecone |
| Backend | FastAPI + Python 3.11 | Django, Flask |
| Queue | Celery + Redis | RQ, Dramatiq |
| Stockage objets | MinIO | S3, Azure Blob |
| Monitoring | Prometheus + Grafana + Langfuse | Datadog |

## Conséquences positives
- Stack 100% open-source pour les composants d'infrastructure (maîtrise des coûts)
- LangChain permet d'abstraire le LLM provider (pivot facilité)
- Modèle d'embedding local = aucune fuite de documents sensibles

## Conséquences négatives
- Maintenance de l'infrastructure Qdrant et Celery en interne
- `multilingual-e5-large` moins performant que OpenAI sur l'anglais technique

## Révision prévue
Réévaluer à 6 mois si le volume de documents dépasse 5M chunks.
