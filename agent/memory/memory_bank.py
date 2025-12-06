"""Memory Bank for storing and managing agent experiences."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pickle

import numpy as np
import torch


class MemoryBank:
    """Memory Bank for storing and managing agent experiences.

    Supports memory storage, retrieval, and management with optimized numpy-based embeddings.
    Uses an internal ID mapping system to handle deletions without breaking embeddings.
    """

    def __init__(self, memory_dir: str = "agent_memories",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k: int = 3,
                 device: str = "cpu") -> None:
        self.memory_dir = memory_dir
        self.embedding_model = embedding_model
        self.device = device
        self.top_k = top_k

        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)

        # Memory storage files
        self.memory_file = os.path.join(memory_dir, "memories.json")
        self.embedding_file = os.path.join(memory_dir, "embeddings.npy")
        self.metadata_file = os.path.join(memory_dir, "metadata.json")
        self.id_mapping_file = os.path.join(memory_dir, "id_mapping.json")
        self.embeddings = None

        # Load existing memories
        self.memories = self._load_memories()
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()
        self.id_mapping = self._load_id_mapping()

        # Initialize embedding manager (lazy loading)
        self._embedding_manager = None

        # Cache reverse mapping for search optimization
        self._emb_id_to_mem_id = None
        self._mapping_dirty = True  # Flag to indicate if mapping needs update

    def _get_embedding_manager(self):
        """Lazy load embedding manager."""
        if self._embedding_manager is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_manager = SentenceTransformer(self.embedding_model, device=self.device)
            except ImportError:
                print("Warning: sentence-transformers not installed. Using simple text similarity.")
                self._embedding_manager = None
        return self._embedding_manager

    def _load_memories(self) -> Dict[str, Any]:
        """Load memories from JSON file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memories: {e}")
                return {}
        return {}

    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from numpy file."""
        if os.path.exists(self.embedding_file):
            try:
                embeddings = np.load(self.embedding_file, allow_pickle=True)
                # Convert to proper numpy array if loaded as object array
                if embeddings.dtype == object:
                    embeddings = np.array([np.array(e) for e in embeddings])
                return embeddings
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return None
        return None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {"total_memories": 0, "last_updated": None, "next_mem_id": 0}
        return {"total_memories": 0, "last_updated": None, "next_mem_id": 0}

    def _load_id_mapping(self) -> Dict[int, int]:
        """Load ID mapping from JSON file."""
        if os.path.exists(self.id_mapping_file):
            try:
                with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    # Convert string keys back to int
                    return {int(k): v for k, v in mapping.items()}
            except Exception as e:
                print(f"Error loading ID mapping: {e}")

        # If no mapping exists, create one-to-one mapping for existing memories
        mapping = {}
        mem_ids = [int(id_str) for id_str in self.memories.keys()]
        mem_ids.sort()
        for i, mem_id in enumerate(mem_ids):
            mapping[mem_id] = i
        return mapping

    def _save_id_mapping(self) -> None:
        """Save ID mapping to JSON file."""
        try:
            with open(self.id_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.id_mapping, f, indent=2)
        except Exception as e:
            print(f"Error saving ID mapping: {e}")

    def _save_memories(self) -> None:
        """Save memories to JSON file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memories: {e}")

    def _save_embeddings(self) -> None:
        """Save embeddings to numpy file."""
        if self.embeddings is not None:
            try:
                np.save(self.embedding_file, self.embeddings)
            except Exception as e:
                print(f"Error saving embeddings: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        try:
            self.metadata["total_memories"] = len(self.memories)
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create embedding for text."""
        manager = self._get_embedding_manager()
        if manager is None:
            return None

        try:
            embedding = manager.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None

    def _update_reverse_mapping(self) -> None:
        """Update the cached reverse mapping from embedding index to memory ID."""
        self._emb_id_to_mem_id = {v: k for k, v in self.id_mapping.items()}
        self._mapping_dirty = False

    def add_memory(self,
                   title: str,
                   description: str,
                   content: str,
                   success_rate: float = 0.0,) -> str:
        """Add a new memory to the bank.

        Args:
            title: Short title for the memory
            description: Brief description of the memory
            content: Detailed content/experience
            success_rate: Success rate for this type of task (0.0-1.0)

        Returns:
            Memory ID (integer starting from 0)
        """
        # Get next memory ID
        mem_id = self.metadata.get("next_mem_id", 0)

        memory = {
            "title": title,
            "description": description,
            "content": content,
            "success_rate": success_rate,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }

        # Create embedding for search
        search_text = f"{title} {description} {content}"
        embedding = self._create_embedding(search_text)

        # Add embedding to numpy array
        if embedding is not None:
            if self.embeddings is None:
                self.embeddings = np.array([embedding])
                embedding_index = 0
            else:
                embedding_index = len(self.embeddings)
                self.embeddings = np.vstack([self.embeddings, embedding])

            # Update ID mapping
            self.id_mapping[mem_id] = embedding_index
            # Mark mapping as dirty since it changed
            self._mapping_dirty = True

        # Add to memories
        self.memories[str(mem_id)] = memory

        # Update next memory ID
        self.metadata["next_mem_id"] = mem_id + 1

        # Save changes
        self._save_memories()
        self._save_embeddings()
        self._save_metadata()
        self._save_id_mapping()

        return str(mem_id)

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get memory by ID."""
        memory = self.memories.get(str(memory_id))
        if memory is None:
            return None

        # Update access statistics
        memory["access_count"] += 1
        memory["last_accessed"] = datetime.now().isoformat()
        self._save_memories()

        return memory

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories."""
        return list(self.memories.values())

    def update_memory(self, memory_id: int, updates: Dict[str, Any]) -> bool:
        """Update memory with new data."""
        memory = self.memories.get(str(memory_id))
        if memory is None:
            return False

        # Update fields
        for key, value in updates.items():
            if key in memory:
                memory[key] = value

        # Update embedding if content changed
        if "title" in updates or "description" in updates or "content" in updates:
            search_text = f"{memory['title']} {memory['description']} {memory['content']}"
            embedding = self._create_embedding(search_text)
            if embedding is not None and self.embeddings is not None and memory_id in self.id_mapping:
                embedding_idx = self.id_mapping[memory_id]
                if 0 <= embedding_idx < len(self.embeddings):
                    self.embeddings[embedding_idx] = embedding

        self._save_memories()
        self._save_embeddings()
        return True

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Search memories by text using PyTorch-optimized vector operations."""
        if not self.memories or self.embeddings is None:
            return []

        # Update reverse mapping cache if needed
        if self._mapping_dirty or self._emb_id_to_mem_id is None:
            self._update_reverse_mapping()

        # Create query embedding
        query_embedding = self._create_embedding(query)
        if query_embedding is None:
            return []

        # Convert to PyTorch tensors
        query_tensor = torch.from_numpy(query_embedding).float().to(self.device)
        embeddings_tensor = torch.from_numpy(self.embeddings).float().to(self.device)

        # Ensure proper shape
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)  # [1, embedding_dim]

        # Normalize using PyTorch operations for GPU acceleration
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

        # Calculate cosine similarities using matrix multiplication
        similarities = torch.mm(query_norm, embeddings_norm.T).squeeze(0)

        # Get top-k values and indices
        top_k = min(self.top_k, len(similarities))
        top_values, top_indices = torch.topk(similarities, top_k)

        # Convert to CPU for processing
        top_indices_cpu = top_indices.cpu().numpy()
        similarities_cpu = top_values.cpu().numpy()

        # Return corresponding memories
        result_memories = []
        for idx, embedding_idx in enumerate(top_indices_cpu):
            if similarities_cpu[idx] > 0:  # Only return memories with some similarity
                mem_id = self._emb_id_to_mem_id.get(int(embedding_idx))
                if mem_id is not None:
                    memory = self.get_memory(mem_id)
                    if memory:
                        # Add similarity score to memory
                        memory_copy = memory.copy()
                        memory_copy['similarity_score'] = float(similarities_cpu[idx])
                        result_memories.append(memory_copy)

        return result_memories


    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory from the bank.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if memory was deleted, False if memory was not found
        """
        memory_str_id = str(memory_id)
        if memory_str_id not in self.memories:
            return False

        # Remove from memories
        del self.memories[memory_str_id]

        # Update embeddings and ID mapping
        if memory_id in self.id_mapping and self.embeddings is not None:
            embedding_idx = self.id_mapping[memory_id]

            # Remove embedding from the array
            self.embeddings = np.delete(self.embeddings, embedding_idx, axis=0)

            # Remove from ID mapping
            del self.id_mapping[memory_id]

            # Update all ID mappings that were after the deleted embedding
            for mem_id_key, emb_idx in list(self.id_mapping.items()):
                if emb_idx > embedding_idx:
                    self.id_mapping[mem_id_key] = emb_idx - 1

            # Mark mapping as dirty since it changed
            self._mapping_dirty = True

        # Save all changes
        self._save_memories()
        self._save_embeddings()
        self._save_metadata()
        self._save_id_mapping()

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            "total_memories": len(self.memories),
            "total_embeddings": self.embeddings.shape[0] if self.embeddings is not None else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "last_updated": self.metadata.get("last_updated"),
            "next_mem_id": self.metadata.get("next_mem_id", 0)
        }

