"""Vector store interface and implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    id: str
    score: float
    payload: Dict[str, Any]


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get a vector by ID."""
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of vector store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "items",
        dimension: int = 1024,
        distance_metric: str = "Cosine",
    ):
        """Initialize Qdrant client."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension

        # Map distance metric
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance_metric, Distance.COSINE)

        # Create collection if it doesn't exist
        self._create_collection_if_needed()

    def _create_collection_if_needed(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import VectorParams

        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=self.distance),
            )

    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to Qdrant."""
        from qdrant_client.models import PointStruct

        if payloads is None:
            payloads = [{} for _ in ids]

        points = [
            PointStruct(id=idx, vector=vector.tolist(), payload=payload)
            for idx, (vector, payload) in enumerate(zip(vectors, payloads))
        ]

        # Also store the original ID in payload
        for point, item_id in zip(points, ids):
            point.payload["item_id"] = item_id

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # OR condition for lists
                    for v in value:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

            if conditions:
                qdrant_filter = Filter(should=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            VectorSearchResult(
                id=result.payload.get("item_id", str(result.id)),
                score=result.score,
                payload=result.payload,
            )
            for result in results
        ]

    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get a vector by ID."""
        # Search by item_id in payload
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="item_id", match=MatchValue(value=vector_id))]
            ),
            limit=1,
        )

        if results[0]:
            return np.array(results[0][0].vector)

        return None

    def delete_vectors(self, ids: List[str]):
        """Delete vectors by item IDs."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Delete by item_id in payload
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                should=[
                    FieldCondition(key="item_id", match=MatchValue(value=item_id))
                    for item_id in ids
                ]
            ),
        )


class FAISSVectorStore(VectorStore):
    """FAISS implementation of vector store (lighter alternative to Qdrant)."""

    def __init__(self, dimension: int = 1024, index_type: str = "Flat"):
        """Initialize FAISS index."""
        import faiss

        self.dimension = dimension

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage (simple dict)
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.next_idx = 0

    def add_vectors(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to FAISS."""
        if payloads is None:
            payloads = [{} for _ in ids]

        # Ensure float32
        vectors = vectors.astype(np.float32)

        # Add to index
        self.index.add(vectors)

        # Store mappings
        for i, (item_id, payload) in enumerate(zip(ids, payloads)):
            idx = self.next_idx + i
            self.id_to_idx[item_id] = idx
            self.idx_to_id[idx] = item_id
            self.payloads[item_id] = payload

        self.next_idx += len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        # Ensure float32 and 2D
        query_vector = query_vector.astype(np.float32)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vector, top_k)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            item_id = self.idx_to_id.get(idx)
            if not item_id:
                continue

            payload = self.payloads.get(item_id, {})

            # Apply filters if provided
            if filters:
                matches = all(
                    payload.get(key) == value or payload.get(key) in value
                    if isinstance(value, list)
                    else payload.get(key) == value
                    for key, value in filters.items()
                )
                if not matches:
                    continue

            # Convert L2 distance to similarity score
            score = 1.0 / (1.0 + float(dist))

            results.append(
                VectorSearchResult(id=item_id, score=score, payload=payload)
            )

        return results

    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get a vector by ID."""
        idx = self.id_to_idx.get(vector_id)
        if idx is None:
            return None

        # FAISS doesn't have direct vector retrieval, so we store separately
        # For production, you'd want to maintain a separate storage
        return None

    def delete_vectors(self, ids: List[str]):
        """Delete vectors (FAISS doesn't support deletion easily)."""
        # FAISS doesn't support deletion, so just remove from metadata
        for item_id in ids:
            idx = self.id_to_idx.pop(item_id, None)
            if idx is not None:
                self.idx_to_id.pop(idx, None)
                self.payloads.pop(item_id, None)
