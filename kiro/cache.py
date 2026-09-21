# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Model metadata cache for Kiro Gateway.

Thread-safe storage for available model information
with TTL and lazy loading support.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import MODEL_CACHE_TTL, MODEL_VERIFICATION_TTL, DEFAULT_MAX_INPUT_TOKENS


class ModelInfoCache:
    """
    Thread-safe cache for storing model metadata.
    
    Uses Lazy Loading for population - data is loaded
    only on first access or when cache is stale.
    
    Attributes:
        cache_ttl: Cache time-to-live in seconds
    
    Example:
        >>> cache = ModelInfoCache()
        >>> await cache.update([{"modelId": "claude-sonnet-4", "tokenLimits": {...}}])
        >>> info = cache.get("claude-sonnet-4")
        >>> max_tokens = cache.get_max_input_tokens("claude-sonnet-4")
    """
    
    def __init__(
        self,
        cache_ttl: int = MODEL_CACHE_TTL,
        verified_models: Optional[Dict[str, float]] = None,
        verification_ttl: int = MODEL_VERIFICATION_TTL,
    ):
        """
        Initializes the model cache.

        Args:
            cache_ttl: Metadata cache time-to-live in seconds.
            verified_models: Shared account-scoped map of model IDs to the
                timestamp of their last successful completion.
            verification_ttl: Time a successful model verification remains
                visible in the model list.
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl
        self._verified_models = verified_models if verified_models is not None else {}
        self._verification_ttl = verification_ttl
    
    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Updates the model cache.
        
        Thread-safely replaces cache contents with new data.
        
        Args:
            models_data: List of dictionaries with model information.
                        Each dictionary must contain the "modelId" key.
        """
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns model information.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dictionary with model information or None if model not found
        """
        return self._cache.get(model_id)
    
    def is_valid_model(self, model_id: str) -> bool:
        """
        Check if model exists in dynamic cache.
        
        Used by ModelResolver to verify if a model is available.
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if model exists in cache, False otherwise
        """
        return model_id in self._cache
    
    def add_hidden_model(self, display_name: str, internal_id: str) -> None:
        """
        Add a hidden model to the cache.
        
        Hidden models are not returned by Kiro /ListAvailableModels API
        but are still functional. They are added to the cache so they
        appear in our /v1/models endpoint.
        
        Args:
            display_name: Model name to display (e.g., "claude-3.7-sonnet")
            internal_id: Internal Kiro ID (e.g., "CLAUDE_3_7_SONNET_20250219_V1_0")
        """
        if display_name not in self._cache:
            self._cache[display_name] = {
                "modelId": display_name,
                "modelName": display_name,
                "description": f"Hidden model (internal: {internal_id})",
                "tokenLimits": {"maxInputTokens": DEFAULT_MAX_INPUT_TOKENS},
                "_internal_id": internal_id,  # Store internal ID for reference
                "_is_hidden": True,  # Mark as hidden model
            }
            logger.debug(f"Added hidden model: {display_name} → {internal_id}")
    
    def get_max_input_tokens(self, model_id: str) -> int:
        """
        Returns maxInputTokens for the model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Maximum number of input tokens or DEFAULT_MAX_INPUT_TOKENS
        """
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or DEFAULT_MAX_INPUT_TOKENS
        return DEFAULT_MAX_INPUT_TOKENS
    
    def is_empty(self) -> bool:
        """
        Checks if the cache is empty.
        
        Returns:
            True if cache is empty
        """
        return not self._cache
    
    def is_stale(self) -> bool:
        """
        Checks if the cache is stale.
        
        Returns:
            True if cache is stale (more than cache_ttl seconds have passed)
            or if cache was never updated
        """
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl
    
    def get_all_model_ids(self) -> List[str]:
        """
        Returns a list of all model IDs in the cache.
        
        Returns:
            List of model IDs
        """
        return list(self._cache.keys())
    
    def mark_verified(self, model_id: str, verified_at: Optional[float] = None) -> None:
        """Record a successful real completion for this account's model list."""
        self._verified_models[model_id] = verified_at if verified_at is not None else time.time()

    def invalidate_verified(self, model_id: str) -> None:
        """Remove a model from the verified list after an invalid-model response."""
        self._verified_models.pop(model_id, None)

    def is_verified(self, model_id: str, now: Optional[float] = None) -> bool:
        """Return whether a model has a non-expired successful verification."""
        verified_at = self._verified_models.get(model_id)
        if verified_at is None:
            return False
        current_time = now if now is not None else time.time()
        return current_time - verified_at <= self._verification_ttl

    def get_verified_model_ids(self, now: Optional[float] = None) -> List[str]:
        """Return model IDs with a non-expired successful verification."""
        current_time = now if now is not None else time.time()
        return sorted(
            model_id
            for model_id, verified_at in self._verified_models.items()
            if current_time - verified_at <= self._verification_ttl
        )

    @property
    def verified_models(self) -> Dict[str, float]:
        """Return account-scoped verification timestamps for persistence."""
        return self._verified_models

    @property
    def size(self) -> int:
        """Number of models in the cache."""
        return len(self._cache)
    
    @property
    def last_update_time(self) -> Optional[float]:
        """Last update time (timestamp) or None."""
        return self._last_update