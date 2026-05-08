"""
API REST — Tech Office Cockpit.

FastAPI avec gestion du cycle de vie des connexions (Qdrant, LLM, Jira)
et endpoints pour les 4 cas d'usage principaux.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.connectors.jira_connector import JiraConnector, JiraNotFoundError, JiraAuthError
from src.connectors.rovo_connector import RovoConnector
from src.modules.architectural_alignment import ArchitecturalAlignmentModule
from src.modules.governance_remediation import GovernanceRemediationModule
from src.modules.rovo_agent import TechOfficeAgent
from src.pipeline.document_ingestion import DocumentIngestionPipeline
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ─── Application state ────────────────────────────────────────────────────────

class AppState:
    settings: Settings
    jira: JiraConnector
    rovo: RovoConnector
    rag: RAGPipeline
    ingestion: DocumentIngestionPipeline
    alignment_module: ArchitecturalAlignmentModule
    governance_module: GovernanceRemediationModule
    agent: TechOfficeAgent


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise et ferme toutes les connexions au démarrage/arrêt."""
    state.settings = get_settings()

    state.jira = JiraConnector(state.settings.jira)
    await state.jira._init_client()

    state.rovo = RovoConnector(state.settings.rovo)
    await state.rovo.__aenter__()

    state.rag = RAGPipeline(state.settings)
    await state.rag.initialize()

    state.ingestion = DocumentIngestionPipeline(state.settings)
    await state.ingestion.initialize()

    state.alignment_module = ArchitecturalAlignmentModule(
        state.settings, state.jira, state.rag, state.rovo
    )
    state.governance_module = GovernanceRemediationModule(
        state.settings, state.jira, state.rag
    )
    state.agent = TechOfficeAgent(
        state.settings, state.jira, state.rag, state.rovo
    )
    await state.agent.initialize()

    logger.info("Tech Office Cockpit démarré.")
    yield

    await state.jira.close()
    await state.rovo.__aexit__(None, None, None)
    logger.info("Tech Office Cockpit arrêté.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tech Office Cockpit",
        description="Copilote intelligent pour l'Architecture d'Entreprise",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(alignment_router, prefix="/api/v1/alignment", tags=["Alignment"])
    app.include_router(governance_router, prefix="/api/v1/governance", tags=["Governance"])
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
    app.include_router(ingestion_router, prefix="/api/v1/ingest", tags=["Ingestion"])

    return app


# ─── Routers ──────────────────────────────────────────────────────────────────

from fastapi import APIRouter

health_router = APIRouter()
alignment_router = APIRouter()
governance_router = APIRouter()
agent_router = APIRouter()
ingestion_router = APIRouter()


# ─── Health ───────────────────────────────────────────────────────────────────

@health_router.get("/")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "services": {
            "jira": "connected",
            "rovo": "connected" if state.rovo.is_available else "unavailable",
            "qdrant": "connected",
        },
    }


# ─── Alignment ────────────────────────────────────────────────────────────────

class AlignmentRequest(BaseModel):
    issue_key: str = Field(..., description="Clé du ticket Jira (ex: ARCH-123)")
    include_attachments: bool = Field(True, description="Analyser les pièces jointes PDF")


@alignment_router.post("/analyze")
async def analyze_alignment(request: AlignmentRequest) -> dict:
    """
    Analyse l'alignement architectural d'un ticket Jira avec les référentiels internes.
    """
    try:
        report = await state.alignment_module.analyze_issue(
            request.issue_key,
            include_attachments=request.include_attachments,
        )
        return report.to_dict()
    except JiraNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ticket {request.issue_key} introuvable dans Jira.")
    except JiraAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        logger.error("Erreur analyse alignement %s : %s", request.issue_key, exc)
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'analyse.")


class ProjectAlignmentRequest(BaseModel):
    project_key: str = Field(..., description="Clé du projet Jira (ex: COREBANK)")


@alignment_router.post("/project")
async def analyze_project_alignment(request: ProjectAlignmentRequest) -> dict:
    """Analyse l'alignement architectural de tous les Epics d'un projet."""
    reports = await state.alignment_module.analyze_project_epics(request.project_key)
    return {
        "project_key": request.project_key,
        "total_epics": len(reports),
        "reports": [r.to_dict() for r in reports],
        "risk_summary": {
            "ROUGE": sum(1 for r in reports if r.risk_level.value == "ROUGE"),
            "AMBER": sum(1 for r in reports if r.risk_level.value == "AMBER"),
            "VERT": sum(1 for r in reports if r.risk_level.value == "VERT"),
        },
    }


# ─── Governance ───────────────────────────────────────────────────────────────

class GovernanceRequest(BaseModel):
    project_key: str
    max_issues: int = Field(50, ge=1, le=200)


@governance_router.post("/analyze")
async def analyze_governance(request: GovernanceRequest) -> dict:
    """Analyse la dette technique et les failles de sécurité d'un projet."""
    report = await state.governance_module.analyze_project(
        request.project_key, max_issues=request.max_issues
    )
    return report.to_dict()


@governance_router.get("/security/{project_key}")
async def security_posture(project_key: str) -> dict:
    """Retourne le score de sécurité et les failles critiques d'un projet."""
    return await state.governance_module.analyze_security_posture(project_key)


# ─── Agent ────────────────────────────────────────────────────────────────────

class AgentQuery(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)


@agent_router.post("/ask")
async def ask_agent(query: AgentQuery) -> dict:
    """
    Pose une question en langage naturel au Tech Office Agent.

    Exemples :
    - "Quels projets impactent le Core Banking et ont un retard de livraison ?"
    - "Génère un rapport de conformité pour le projet MIGCLOUD"
    - "Quelles sont les failles de sécurité P1 non résolues ?"
    """
    try:
        result = await state.agent.ask(query.question)
        return result
    except Exception as exc:
        logger.error("Erreur agent : %s", exc)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la question.")


# ─── Ingestion ────────────────────────────────────────────────────────────────

class IngestionRequest(BaseModel):
    collection: str = Field(..., description="Collection Qdrant cible")
    project_key: Optional[str] = None
    doc_type: Optional[str] = Field(None, description="Type: DAT | ADR | LLD | HLD | NOTE_SERVICE")
    version: Optional[str] = None


@ingestion_router.post("/document")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_key: Optional[str] = None,
    collection: str = "architecture_docs",
    doc_type: Optional[str] = None,
) -> dict:
    """
    Ingère un document PDF/Word dans la base vectorielle.

    L'ingestion est effectuée en arrière-plan pour ne pas bloquer l'API.
    """
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Type de fichier non supporté : {file.content_type}",
        )

    content = await file.read()
    metadata = {
        "project_key": project_key,
        "doc_type": doc_type,
        "uploaded_by": "api",
    }

    background_tasks.add_task(
        state.ingestion.ingest_document,
        content=content,
        filename=file.filename,
        collection=collection,
        metadata=metadata,
    )

    return {
        "status": "accepted",
        "filename": file.filename,
        "collection": collection,
        "message": "Ingestion démarrée en arrière-plan.",
    }


app = create_app()
