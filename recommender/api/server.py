"""FastAPI server for personal recommendation system."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path

from ..models import ModelFactory
from ..database import DuckDBStore, QdrantVectorStore, EventType
from ..candidate_generation import CandidateGenerator
from ..reranking import Reranker
from ..exploration import ThompsonSamplingExplorer
from ..feedback import FeedbackProcessor
from ..enrichment import ContentEnricher


# Pydantic models for API
class RecommendRequest(BaseModel):
    """Request for recommendations."""

    user_id: str = "default"
    num_items: int = 20
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Feedback event."""

    item_id: str
    event_type: str
    dwell_seconds: Optional[float] = None
    completion_fraction: Optional[float] = None
    session_id: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response with recommendations."""

    items: List[Dict[str, Any]]
    session_id: str


def create_app(config_path: str = "./config.yaml") -> FastAPI:
    """Create FastAPI application with all components."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create app
    app = FastAPI(title="Personal Recommender", version="0.1.0")

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["server"].get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components on startup
    @app.on_event("startup")
    async def startup():
        """Initialize all components."""
        # Create data directory
        Path(config["storage"]["data_dir"]).mkdir(parents=True, exist_ok=True)

        # Initialize models
        app.state.embedding_model = ModelFactory.create_embedding_model(
            config["models"]["embedding"]
        )
        app.state.reranker_model = ModelFactory.create_reranker(
            config["models"]["reranker"]
        )
        app.state.llm = ModelFactory.create_llm(config["models"]["llm"])

        # Initialize databases
        db_path = config["relational_db"]["path"]
        app.state.db_store = DuckDBStore(db_path)

        vector_db_config = config["vector_db"]
        app.state.vector_store = QdrantVectorStore(
            host=vector_db_config["host"],
            port=vector_db_config["port"],
            collection_name=vector_db_config["collection_name"],
            dimension=app.state.embedding_model.get_dimension(),
            distance_metric=vector_db_config["distance_metric"],
        )

        # Initialize recommendation components
        app.state.candidate_generator = CandidateGenerator(
            vector_store=app.state.vector_store,
            db_store=app.state.db_store,
            config=config["recommendation"]["candidate_generation"],
        )

        app.state.reranker = Reranker(
            vector_store=app.state.vector_store,
            db_store=app.state.db_store,
            cross_encoder=app.state.reranker_model,
            config=config["recommendation"]["reranking"],
        )

        app.state.explorer = ThompsonSamplingExplorer(
            db_store=app.state.db_store,
            config=config["recommendation"]["exploration"],
        )

        app.state.feedback_processor = FeedbackProcessor(
            db_store=app.state.db_store, config=config["recommendation"]["feedback"]
        )

        app.state.enricher = ContentEnricher(
            llm=app.state.llm, config=config["enrichment"]
        )

        print("🚀 Recommender system initialized!")

    # === API Endpoints ===

    @app.get("/")
    async def root():
        """Health check."""
        return {"status": "ok", "message": "Personal Recommender API"}

    @app.post("/recommend", response_model=RecommendationResponse)
    async def get_recommendations(request: RecommendRequest):
        """Get personalized recommendations.

        This is the main endpoint that:
        1. Generates candidates
        2. Reranks them
        3. Adds exploration items
        4. Returns top-k with explanations
        """
        try:
            # Get user profile
            user_profile = app.state.db_store.get_user_profile(request.user_id)

            # Get recent session items if session_id provided
            session_items = []
            if request.session_id:
                recent_events = app.state.db_store.get_recent_events(limit=20)
                session_events = [
                    e for e in recent_events if e.session_id == request.session_id
                ]
                session_item_ids = [e.item_id for e in session_events[-5:]]
                session_items = [
                    app.state.db_store.get_item(item_id)
                    for item_id in session_item_ids
                ]
                session_items = [item for item in session_items if item]

            # 1. Generate candidates
            candidates = app.state.candidate_generator.generate_candidates(
                user_profile=user_profile,
                session_items=session_items,
                num_candidates=500,
                context=request.context,
            )

            # 2. Rerank candidates
            reranked = app.state.reranker.rerank(
                candidates=candidates,
                user_profile=user_profile,
                session_items=session_items,
                context=request.context,
                top_k=request.num_items - 5,  # Reserve slots for exploration
            )

            # 3. Add exploration items
            if config["recommendation"]["exploration"]["enabled"]:
                exploration_items = app.state.explorer.get_exploration_items(
                    user_profile=user_profile, num_items=5
                )
                # Mix exploration items into results
                all_results = reranked + [
                    {**item, "score": 0.5, "score_components": {}}
                    for item in exploration_items
                ]
            else:
                all_results = reranked

            # 4. Format response
            items = []
            for result in all_results[: request.num_items]:
                item = app.state.db_store.get_item(result["item_id"])
                if item:
                    items.append(
                        {
                            "id": item.id,
                            "title": item.title,
                            "url": item.url,
                            "creator": item.creator,
                            "summary": item.summary,
                            "media_type": item.media_type.value,
                            "length_seconds": item.length_seconds,
                            "topics": item.topics,
                            "score": result.get("score", 0.0),
                            "strategy": result.get("strategy", "unknown"),
                            "arm": result.get("arm"),
                        }
                    )

            return RecommendationResponse(
                items=items, session_id=request.session_id or "new"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/feedback")
    async def log_feedback(request: FeedbackRequest):
        """Log user feedback event."""
        try:
            # Parse event type
            event_type = EventType(request.event_type.lower())

            # Log event
            app.state.feedback_processor.log_event(
                item_id=request.item_id,
                event_type=event_type,
                dwell_seconds=request.dwell_seconds,
                completion_fraction=request.completion_fraction,
                session_id=request.session_id,
            )

            return {"status": "ok"}

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/stats")
    async def get_stats():
        """Get user statistics."""
        try:
            user_profile = app.state.db_store.get_user_profile()

            return {
                "total_items_viewed": user_profile.total_items_viewed,
                "total_items_completed": user_profile.total_items_completed,
                "top_topics": user_profile.top_topics,
                "top_creators": user_profile.top_creators,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/items/{item_id}")
    async def get_item(item_id: str):
        """Get item details."""
        item = app.state.db_store.get_item(item_id)

        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        return item.to_dict()

    return app


# For running with uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
