# Stratégie d'Intégration Jira & Rovo — Tech Office Cockpit

**Version :** 1.0  
**Date :** 2026-05-08

---

## 1. Intégration Jira — API REST v3

### 1.1 Méthodes d'authentification

#### Option A : Personal Access Token (PAT) — Recommandé pour Data Center
```
Authorization: Bearer <PAT>
```
- Généré par l'utilisateur technique dédié (`svc-techoffice-cockpit`)
- Durée de vie configurable, révocable immédiatement
- **Limites :** Lié à un compte utilisateur (risque si départ)

#### Option B : OAuth 2.0 (3-legged) — Recommandé pour Cloud
```
POST https://auth.atlassian.com/oauth/token
  grant_type: authorization_code
  client_id: <APP_CLIENT_ID>
  client_secret: <SECRET>
```
- Application Jira enregistrée sur developer.atlassian.com
- Scopes requis : `read:jira-work`, `read:jira-user`, `read:attachment:jira`
- **Avantage :** Token de service indépendant de tout utilisateur

#### Option C : API Token (Basic Auth) — Dev/Test uniquement
```
Authorization: Basic base64(email:api_token)
```
**Ne jamais utiliser en production.**

### 1.2 Endpoints clés utilisés

| Endpoint | Usage |
|----------|-------|
| `GET /rest/api/3/issue/{issueKey}` | Récupérer un ticket complet (champs + custom fields) |
| `GET /rest/api/3/issue/{issueKey}/attachments` | Lister les pièces jointes |
| `GET /rest/api/3/attachment/content/{id}` | Télécharger un fichier binaire |
| `POST /rest/api/3/search` | Recherche JQL (filtres projet, statut, date) |
| `GET /rest/api/3/project/{projectKey}/versions` | Versions/Releases d'un projet |
| `GET /rest/api/3/field` | Récupérer la liste des champs custom |
| `GET /rest/agile/1.0/board/{boardId}/sprint` | Sprints actifs (Scrum/Kanban) |

### 1.3 Pagination et Rate Limiting

```python
# Pattern de pagination recommandé
params = {
    "jql": "project = ARCH AND issuetype = Epic",
    "startAt": 0,
    "maxResults": 100,  # max 100 par page
    "fields": ["summary", "status", "assignee", "customfield_10xxx"]
}

# Rate limiting Jira Cloud : ~300 req/min par token
# Utiliser un semaphore asyncio pour respecter les limites
```

**Critique architecturale :** Implémenter un circuit breaker (bibliothèque `tenacity`) pour
gérer les 429 (rate limit) et les 503 (indisponibilité Jira). Ne jamais appeler Jira
synchroniquement depuis un endpoint API critique.

### 1.4 Champs Custom — Mapping

Les champs custom Jira ont des IDs numériques (`customfield_10XXX`) spécifiques à chaque
instance. Le mapping doit être externalisé en configuration :

```yaml
# config/jira_field_mapping.yaml
field_mapping:
  architectural_impact: "customfield_10201"
  technical_debt_score: "customfield_10202"
  dat_reference: "customfield_10203"
  security_classification: "customfield_10204"
  remediation_priority: "customfield_10205"
```

---

## 2. Intégration Rovo

### 2.1 Qu'est-ce que Jira Rovo ?

Rovo est la couche IA d'Atlassian (GA depuis 2024) offrant :
- Un **graphe de connaissances** indexant Jira, Confluence, Slack (via connecteurs)
- Des **Rovo Agents** : agents conversationnels personnalisables déployables dans l'UI Atlassian
- Une **API Rovo** (Beta) pour l'interrogation programmatique du graphe

### 2.2 Stratégie d'utilisation — Deux approches

#### Approche A : Rovo Agent natif (Low-Code)
Créer un Rovo Agent directement dans l'UI Atlassian avec des instructions personnalisées.

**Avantages :**
- Déploiement immédiat sans code
- Accès natif au graphe de connaissances Atlassian
- Présent dans l'interface Jira (accessibilité utilisateurs)

**Inconvénients :**
- Personnalisation limitée (pas de tools custom Python)
- Pas d'accès aux bases documentaires hors-Atlassian (PDFs internes)
- Dépendant de l'écosystème Atlassian

**Configuration d'un Rovo Agent Tech Office :**
```
Nom : Tech Office Assistant
Instructions système :
  Tu es l'assistant du Tech Office. Tu analyses les projets Jira pour identifier
  les risques architecturaux, les retards de livraison et la dette technique.
  
  Quand on te demande l'état d'un projet :
  1. Consulte les Epics du projet et leur statut
  2. Identifie les tickets bloqués ou en retard (due_date < today)
  3. Recherche les tickets liés à la dette technique ou aux failles de sécurité
  4. Synthétise en un rapport structuré avec niveau de risque (RAG : Rouge/Amber/Vert)

Actions autorisées : search_jira, read_confluence_page
```

#### Approche B : API Rovo programmatique (Recommandée pour le Cockpit)
Interroger l'API Rovo depuis notre backend Python pour enrichir les analyses.

```python
# Exemple d'appel API Rovo (endpoint Beta)
POST https://api.atlassian.com/rovo/v1/search
Headers:
  Authorization: Bearer <ROVO_API_TOKEN>
  Content-Type: application/json
Body:
{
  "query": "projets impactant le Core Banking avec retard de livraison DAT",
  "sources": ["jira", "confluence"],
  "limit": 20
}
```

**Avantages :**
- Intégrable dans nos LangChain Tools
- Combinable avec notre RAG sur documents internes
- Contrôle total du prompt et du post-traitement

**Inconvénients :**
- API en Beta (risque de breaking changes)
- Nécessite un abonnement Rovo actif

### 2.3 Architecture de l'Agent Rovo Hybride (Recommandé)

```
User Query (NL)
      │
      ▼
LangChain ReAct Agent (notre backend)
      │
      ├──[Tool: rovo_search] ──────────► Rovo API
      │                                     └──► Graphe Atlassian (Jira + Confluence)
      │
      ├──[Tool: search_architecture_docs] ─► Qdrant (nos PDFs internes)
      │
      ├──[Tool: jira_jql_search] ──────────► Jira REST API
      │
      └──[Tool: generate_report] ──────────► LLM Claude (analyse finale)
                │
                ▼
         Réponse enrichie (NL + JSON structuré)
```

---

## 3. Gouvernance des Données — Data Privacy

### 3.1 Classification des données

| Type de donnée | Niveau | Stockage autorisé | LLM autorisé |
|----------------|--------|-------------------|--------------|
| Descriptions tickets Jira (sans données perso) | Interne | PostgreSQL + Qdrant | Cloud LLM OK |
| Pièces jointes (DAT, LLD) | Confidentiel | MinIO chiffré | LLM local préféré |
| Notes de service sécurité | Très confidentiel | MinIO chiffré + RLS | LLM local UNIQUEMENT |
| Données personnelles (assignees) | RGPD sensible | Masquage avant stockage | Interdit |

### 3.2 Pseudonymisation RGPD

```python
# Masquer les données personnelles avant embedding/LLM
def anonymize_issue(issue: dict) -> dict:
    """Remplace emails et noms par des identifiants opaques."""
    issue["assignee"] = hash_user(issue.get("assignee", {}).get("accountId"))
    issue["reporter"] = hash_user(issue.get("reporter", {}).get("accountId"))
    # Supprimer les commentaires (peuvent contenir des données perso)
    issue.pop("comments", None)
    return issue
```

### 3.3 Contrôle d'accès aux collections Qdrant

```python
# Chaque requête Qdrant doit inclure un filtre par project_key
# basé sur les permissions Jira de l'utilisateur appelant
filter = models.Filter(
    must=[
        models.FieldCondition(
            key="project_key",
            match=models.MatchAny(any=user_authorized_projects)
        )
    ]
)
```

### 3.4 Audit trail

Tout appel LLM qui analyse un ticket ou un document sensible doit être loggé dans
`analysis_audit` avec : `user_id`, `issue_key`, `doc_ids_retrieved`, `timestamp`.
Rétention : 2 ans (conformité COBIT/ITIL).

---

## 4. Gestion des Erreurs et Résilience

### 4.1 Pattern de retry Jira
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((JiraRateLimitError, JiraServiceUnavailableError))
)
async def get_issue_with_retry(issue_key: str) -> dict:
    ...
```

### 4.2 Fallback si Rovo indisponible
Si l'API Rovo est indisponible (Beta), le système doit dégrader gracieusement vers
une recherche JQL native Jira sans interrompre l'analyse principale.

### 4.3 Cache LLM
Les analyses coûteuses (compliance report) doivent être cachées dans `analysis_cache`
pendant 24h. La clé de cache est : `SHA256(issue_key + sorted(doc_checksums))`.
