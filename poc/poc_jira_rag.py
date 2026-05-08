#!/usr/bin/env python3
"""
Proof of Concept — Tech Office Cockpit
=======================================

Démontre le flux complet :
1. Extraction d'un ticket Jira (Epic ou Story)
2. Téléchargement et extraction du texte d'une pièce jointe PDF
3. Chargement de documents d'architecture de référence (simulés ou réels)
4. Vérification LLM : le livrable mentionne-t-il les directives d'architecture ?

Usage :
    python poc/poc_jira_rag.py --issue ARCH-123
    python poc/poc_jira_rag.py --issue ARCH-123 --demo   # Mode démo sans Jira réel
    python poc/poc_jira_rag.py --project COREBANK --analyze-all

Prérequis :
    pip install -r requirements.txt
    cp .env.example .env  # Renseigner les credentials
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

# Ajouter le dossier racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("poc_techoffice")


# ─── Données de démonstration ─────────────────────────────────────────────────

DEMO_ISSUE = {
    "key": "ARCH-123",
    "fields": {
        "summary": "Migration API Gateway vers Kong Enterprise — Core Banking",
        "description": {
            "content": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Cette Epic couvre la migration de l'API Gateway Nginx "
                                "vers Kong Enterprise pour l'ensemble des services Core Banking. "
                                "L'objectif est d'améliorer la gestion des policies de sécurité, "
                                "le rate limiting et l'observabilité des flux API. "
                                "Le déploiement cible est Azure Kubernetes Service (AKS) "
                                "avec Helm charts. La solution doit respecter les standards "
                                "d'architecture API définis dans le DAT-API-2024."
                            ),
                        }
                    ]
                }
            ]
        },
        "issuetype": {"name": "Epic"},
        "status": {"name": "In Progress"},
        "priority": {"name": "High"},
        "labels": ["api-gateway", "core-banking", "migration"],
        "customfield_10201": "DAT-API-2024",
    },
}

DEMO_PDF_CONTENT = """
DOSSIER D'ARCHITECTURE TECHNIQUE — DAT-API-2024
Migration API Gateway — Core Banking

1. CONTEXTE ET OBJECTIFS
La migration vers Kong Enterprise répond aux exigences de sécurité ANSSI
et aux recommandations OWASP API Security Top 10.

2. STANDARDS OBLIGATOIRES
- Authentification : OAuth 2.0 / JWT obligatoire sur tous les endpoints
- Chiffrement : TLS 1.3 minimum, certificats Let's Encrypt ou PKI interne
- Rate Limiting : 1000 req/min par consumer par défaut, configurable par plugin Kong
- Logging : OpenTelemetry pour la traçabilité, export vers Elastic APM
- Haute Disponibilité : 3 réplicas Kong minimum, déploiement AKS avec PodDisruptionBudget

3. PATTERNS D'INTÉGRATION APPROUVÉS
- Backend-for-Frontend (BFF) pour les clients mobiles et web
- Circuit Breaker pattern (Kong plugin) pour les backends instables
- API Versioning : /api/v1/, /api/v2/ — déprécation avec 6 mois de préavis

4. EXIGENCES DE CONFORMITÉ
- Toute API exposée doit avoir une fiche de sécurité validée par le RSSI
- Les APIs Core Banking doivent respecter la réglementation DORA (Digital Operational Resilience Act)
- Tests de performance obligatoires (seuil : P99 < 200ms pour les APIs transactionnelles)

5. DÉPLOIEMENT
- Infrastructure as Code : Terraform + Helm Charts versionnés dans Git
- Pipeline CI/CD : GitLab CI avec stages validate, test, deploy-staging, deploy-prod
- Rollback automatique si erreur > 1% dans les 5 premières minutes post-déploiement
"""

DEMO_ARCH_DOCS = [
    {
        "title": "DAT-API-2024 — Standards API Gateway",
        "content": DEMO_PDF_CONTENT,
        "doc_type": "DAT",
    },
    {
        "title": "ADR-015 — Adoption Kong Enterprise",
        "content": """
ADR-015 : Adoption de Kong Enterprise comme API Gateway standard

Statut : Accepté (2024-01-15)

Contexte : L'entreprise utilise des solutions hétérogènes (Nginx, AWS API Gateway,
Apigee) entraînant une fragmentation de la gouvernance API.

Décision : Kong Enterprise est adopté comme solution standard pour l'ensemble
des APIs du SI. Kong Data Plane sera déployé sur AKS, Kong Control Plane sur
une instance dédiée.

Conséquences :
- Tous les nouveaux projets DOIVENT utiliser Kong
- Migration des APIs existantes sur 24 mois
- Formation obligatoire Kong pour les équipes DevOps
        """,
        "doc_type": "ADR",
    },
    {
        "title": "NOTE-SECURITE-2025-03 — Exigences DORA",
        "content": """
NOTE DE SERVICE — Direction Technique
Référence : NOTE-SECURITE-2025-03
Objet : Mise en conformité DORA — Digital Operational Resilience Act

Les projets impactant les systèmes d'information financiers critiques
(Core Banking, Paiements, Gestion de risque) doivent :

1. Réaliser une analyse d'impact DORA avant toute mise en production
2. Documenter les dépendances aux fournisseurs ICT tiers
3. Implémenter des tests de résilience (chaos engineering) semestriels
4. Maintenir un RTO < 4h et RPO < 1h pour les systèmes critiques
        """,
        "doc_type": "NOTE_SERVICE",
    },
]


# ─── PoC Principal ────────────────────────────────────────────────────────────

async def run_poc_demo(issue_key: str) -> None:
    """Mode démonstration : utilise des données simulées, pas de Jira réel."""

    print("\n" + "═" * 70)
    print("  TECH OFFICE COCKPIT — Proof of Concept (Mode Démonstration)")
    print("═" * 70 + "\n")

    # ── Étape 1 : Simulation données Jira ────────────────────────────────────
    print(f"[1/4] Ticket Jira simulé : {DEMO_ISSUE['key']}")
    print(f"      Titre  : {DEMO_ISSUE['fields']['summary']}")
    print(f"      Type   : {DEMO_ISSUE['fields']['issuetype']['name']}")
    print(f"      Statut : {DEMO_ISSUE['fields']['status']['name']}")
    print()

    # ── Étape 2 : Indexation des documents de référence ──────────────────────
    print("[2/4] Indexation des documents d'architecture dans le Vector DB...")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("      Modèle d'embedding : all-MiniLM-L6-v2 (demo, 384 dim)")

        doc_chunks = []
        doc_vectors = []
        for doc in DEMO_ARCH_DOCS:
            text = f"{doc['title']}\n{doc['content']}"
            vector = model.encode([text], normalize_embeddings=True)[0].tolist()
            doc_chunks.append({"text": text, "metadata": {"title": doc["title"], "doc_type": doc["doc_type"]}})
            doc_vectors.append(vector)
            print(f"      ✓ Indexé : [{doc['doc_type']}] {doc['title']}")

    except ImportError:
        print("      ⚠ sentence-transformers non installé — simulation embedding (demo only)")
        doc_chunks = [{"text": d["title"] + "\n" + d["content"], "metadata": d} for d in DEMO_ARCH_DOCS]
        doc_vectors = None

    print()

    # ── Étape 3 : Extraction du texte du livrable PDF ─────────────────────────
    print("[3/4] Extraction du texte du livrable PDF simulé...")
    pdf_text = DEMO_PDF_CONTENT
    print(f"      ✓ Texte extrait : {len(pdf_text)} caractères")
    print()

    # ── Étape 4 : Analyse LLM ────────────────────────────────────────────────
    print("[4/4] Analyse LLM — Alignement architectural...")

    issue_desc = DEMO_ISSUE["fields"]["description"]["content"][0]["content"][0]["text"]
    combined_context = "\n\n---\n\n".join(
        f"[{d['metadata']['doc_type']}] {d['metadata']['title']}\n{d['text'][:500]}"
        for d in doc_chunks
    )

    llm_prompt = f"""Analyse l'alignement architectural du ticket Jira suivant :

TICKET : {DEMO_ISSUE['key']}
TITRE : {DEMO_ISSUE['fields']['summary']}
DESCRIPTION : {issue_desc}

CONTENU DU LIVRABLE PDF (extrait) :
{pdf_text[:1500]}

RÉFÉRENCES ARCHITECTURALES INTERNES :
{combined_context[:2000]}

Réponds en JSON avec :
{{
  "risk_level": "VERT|AMBER|ROUGE",
  "compliance_score": (0-100),
  "aligned_standards": ["liste des standards respectés"],
  "gaps": [{{"description": "écart détecté", "severity": "HIGH|MEDIUM|LOW"}}],
  "recommendations": ["actions recommandées"],
  "summary": "Synthèse en 2-3 phrases"
}}"""

    # Appel LLM réel si configuré
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": llm_prompt,
                    }
                ],
                system="Tu es un Architecte d'Entreprise Senior. Analyse les tickets Jira pour détecter les écarts architecturaux. Réponds uniquement en JSON valide.",
            )
            llm_response = message.content[0].text
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
            print(f"      ✓ Analyse LLM réelle (Claude Sonnet) — {tokens_used} tokens utilisés")
        except Exception as exc:
            print(f"      ⚠ Erreur LLM API : {exc}")
            llm_response = _generate_demo_response()
            tokens_used = 0
    else:
        print("      ℹ ANTHROPIC_API_KEY non définie — réponse simulée")
        llm_response = _generate_demo_response()
        tokens_used = 0

    # ── Résultats ─────────────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  RÉSULTATS DE L'ANALYSE")
    print("─" * 70)

    try:
        start = llm_response.find("{")
        end = llm_response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(llm_response[start:end])
            _print_alignment_report(result, DEMO_ISSUE["key"], tokens_used)
        else:
            print(llm_response)
    except json.JSONDecodeError:
        print(llm_response)

    print()
    print("═" * 70)
    print("  PoC terminé. Voir docs/architecture/HLD.md pour l'architecture complète.")
    print("═" * 70 + "\n")


async def run_poc_real(issue_key: str) -> None:
    """Mode réel : connexion à Jira et LLM configurés dans .env."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.config.settings import get_settings
    from src.connectors.jira_connector import JiraConnector
    from src.pipeline.document_ingestion import DocumentIngestionPipeline
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.modules.architectural_alignment import ArchitecturalAlignmentModule

    settings = get_settings()

    print("\n" + "═" * 70)
    print(f"  TECH OFFICE COCKPIT — Analyse réelle : {issue_key}")
    print("═" * 70 + "\n")

    async with JiraConnector(settings.jira) as jira:
        rag = RAGPipeline(settings)
        await rag.initialize()

        module = ArchitecturalAlignmentModule(settings, jira, rag)
        report = await module.analyze_issue(issue_key)

        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def _generate_demo_response() -> str:
    return json.dumps({
        "risk_level": "AMBER",
        "compliance_score": 72,
        "aligned_standards": [
            "OAuth 2.0 / JWT mentionné dans le livrable",
            "Déploiement AKS conforme avec ADR-015",
            "TLS 1.3 spécifié",
        ],
        "gaps": [
            {
                "description": "Absence de mention des tests de performance (P99 < 200ms requis par DAT-API-2024 §4)",
                "severity": "HIGH",
            },
            {
                "description": "Analyse d'impact DORA non référencée malgré l'impact Core Banking (NOTE-SECURITE-2025-03)",
                "severity": "HIGH",
            },
            {
                "description": "Stratégie de rollback automatique non détaillée dans le livrable",
                "severity": "MEDIUM",
            },
        ],
        "recommendations": [
            "Ajouter un plan de tests de performance (k6 ou Gatling) avec seuil P99 < 200ms",
            "Compléter l'analyse d'impact DORA et la soumettre au RSSI avant mise en production",
            "Documenter la stratégie de rollback dans le plan de déploiement CI/CD",
            "Vérifier la conformité Kong avec la réglementation DORA sur la gestion des fournisseurs ICT",
        ],
        "summary": (
            "Le livrable est globalement aligné avec les standards Kong Enterprise (ADR-015) "
            "et les exigences TLS/OAuth. Cependant, deux gaps critiques bloquent la validation : "
            "l'absence de plan de tests de performance et l'omission de l'analyse d'impact DORA, "
            "obligatoire pour tout projet Core Banking. Statut AMBER — validation conditionnelle."
        ),
    }, ensure_ascii=False, indent=2)


def _print_alignment_report(result: dict, issue_key: str, tokens: int) -> None:
    COLORS = {"VERT": "\033[92m", "AMBER": "\033[93m", "ROUGE": "\033[91m"}
    RESET = "\033[0m"

    risk = result.get("risk_level", "?")
    color = COLORS.get(risk, "")
    score = result.get("compliance_score", 0)

    print(f"\n  Ticket        : {issue_key}")
    print(f"  Niveau risque : {color}● {risk}{RESET}")
    print(f"  Score conformité : {score}/100")
    if tokens:
        print(f"  Tokens LLM    : {tokens}")

    print("\n  Standards respectés :")
    for std in result.get("aligned_standards", []):
        print(f"    ✓ {std}")

    print("\n  Écarts détectés :")
    for gap in result.get("gaps", []):
        severity_icon = {"HIGH": "✗", "MEDIUM": "⚠", "LOW": "ℹ"}.get(gap.get("severity"), "•")
        print(f"    {severity_icon} [{gap.get('severity', '?')}] {gap.get('description', '')}")

    print("\n  Recommandations :")
    for i, rec in enumerate(result.get("recommendations", []), 1):
        print(f"    {i}. {rec}")

    print(f"\n  Synthèse :\n    {result.get('summary', '')}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tech Office Cockpit — Proof of Concept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python poc/poc_jira_rag.py --demo
  python poc/poc_jira_rag.py --issue ARCH-123
  python poc/poc_jira_rag.py --issue ARCH-123 --demo
        """,
    )
    parser.add_argument("--issue", default="ARCH-123", help="Clé du ticket Jira à analyser")
    parser.add_argument("--demo", action="store_true", help="Mode démonstration (données simulées)")
    parser.add_argument("--project", help="Analyser tous les Epics d'un projet")
    args = parser.parse_args()

    if args.demo or not os.getenv("JIRA_BASE_URL"):
        asyncio.run(run_poc_demo(args.issue))
    else:
        asyncio.run(run_poc_real(args.issue))


if __name__ == "__main__":
    main()
