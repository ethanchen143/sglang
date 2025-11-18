"""
Basic test for Tail-Optimized LRU (TLRU) eviction policy.

This test verifies that TLRU correctly:
1. Calculates safe budgets based on conversation length
2. Trims tails of conversations when evicting
3. Prioritizes already-trimmed nodes for full eviction
"""

import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class TestTLRUBasic(unittest.TestCase):
    """Basic tests for TLRU eviction policy."""

    def test_tlru_trims_tail_when_exceeding_budget(self):
        """TLRU should trim the tail of a conversation when it exceeds the safe budget."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=10,  # Need 10 tokens of headroom
            tlru_next_prompt_estimate=5,  # Estimate next prompt is 5 tokens
        )

        # Insert a conversation with 20 tokens
        # convo_length = 20
        # safe_budget = max(20 + 5 - 10, 0) = max(15, 0) = 15
        # So we should keep 15 tokens and evict 5 tokens
        tokens = list(range(20))
        values = torch.tensor(tokens, dtype=torch.int64) * 10
        cache.insert(RadixKey(tokens), values)

        # Evict 1 token (this should trigger tail trimming)
        mock_allocator.free.reset_mock()
        cache.evict(1)

        # Should have called free once with 5 tokens (trimming to safe budget)
        self.assertEqual(mock_allocator.free.call_count, 1)
        freed = mock_allocator.free.call_args_list[0][0][0]
        self.assertEqual(freed.numel(), 5)

        # Verify the remaining cache is exactly 15 tokens
        result = cache.match_prefix(RadixKey(tokens))
        self.assertEqual(len(result.device_indices), 15)

    def test_tlru_evicts_entire_node_when_budget_is_zero(self):
        """TLRU should evict the entire node when safe budget is 0."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=100,  # Very high threshold
            tlru_next_prompt_estimate=10,
        )

        # Insert a short conversation (5 tokens)
        # convo_length = 5
        # safe_budget = max(5 + 10 - 100, 0) = max(-85, 0) = 0
        # So the entire node should be evicted
        tokens = list(range(5))
        values = torch.tensor(tokens, dtype=torch.int64) * 10
        cache.insert(RadixKey(tokens), values)

        mock_allocator.free.reset_mock()
        cache.evict(1)

        # Should have freed the entire 5 tokens
        self.assertEqual(mock_allocator.free.call_count, 1)
        freed = mock_allocator.free.call_args_list[0][0][0]
        self.assertEqual(freed.numel(), 5)

        # Cache should be empty
        result = cache.match_prefix(RadixKey(tokens))
        self.assertEqual(len(result.device_indices), 0)

    def test_tlru_prioritizes_trimmed_nodes(self):
        """Already-trimmed nodes should be prioritized for eviction."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=5,
            tlru_next_prompt_estimate=2,
        )

        # Insert two conversations
        # Conversation 1: 10 tokens (safe_budget = max(10 + 2 - 5, 0) = 7)
        conv1_tokens = list(range(100, 110))
        conv1_values = torch.tensor([i * 10 for i in range(10)], dtype=torch.int64)
        cache.insert(RadixKey(conv1_tokens), conv1_values)

        # Conversation 2: 8 tokens (safe_budget = max(8 + 2 - 5, 0) = 5)
        conv2_tokens = list(range(200, 208))
        conv2_values = torch.tensor([i * 10 for i in range(8)], dtype=torch.int64)
        cache.insert(RadixKey(conv2_tokens), conv2_values)

        # First eviction: should trim conv1 to 7 tokens (evict 3)
        mock_allocator.free.reset_mock()
        cache.evict(3)

        # Now conv1 should be marked as tel_trimmed
        # Second eviction: should target the already-trimmed conv1 first
        mock_allocator.free.reset_mock()
        cache.evict(1)

        # Should evict from conv1 (the trimmed one)
        # Verify that conv2 is still intact
        result2 = cache.match_prefix(RadixKey(conv2_tokens))
        self.assertEqual(len(result2.device_indices), 8)

    def test_lru_does_not_apply_tail_trimming(self):
        """Standard LRU should not apply tail trimming logic."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="lru",  # Standard LRU, not TLRU
        )

        # Insert a conversation
        tokens = list(range(10))
        values = torch.tensor(tokens, dtype=torch.int64) * 10
        cache.insert(RadixKey(tokens), values)

        mock_allocator.free.reset_mock()
        cache.evict(1)

        # LRU should evict the entire node (10 tokens), not trim it
        self.assertEqual(mock_allocator.free.call_count, 1)
        freed = mock_allocator.free.call_args_list[0][0][0]
        self.assertEqual(freed.numel(), 10)


if __name__ == "__main__":
    unittest.main()