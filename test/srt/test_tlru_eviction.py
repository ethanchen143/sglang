"""
Comprehensive unit tests for the Tail-LRU (TLRU) eviction policy.

This module tests the TLRU implementation in the RadixCache, focusing on:
- cached_tokens tracking (cumulative cached tokens from root to leaf)
- convo_length tracking (logical conversation length, never changes)
- Safe budget calculation and trimming behavior
- Per-conversation eviction with shared prefixes
- Interaction between TLRU trimming and standard LRU fallback

Test Coverage:
- Basic cached_tokens and convo_length tracking
- Shared prefix scenarios
- TLRU safe budget calculation
- Trimming behavior and cached_tokens updates
- convo_length immutability after trimming
- Per-conversation eviction semantics

Usage:
    python test_tlru_eviction.py
    python -m pytest test_tlru_eviction.py -v
    python -m pytest test_tlru_eviction.py::TestTLRUEviction::test_cached_tokens_tracking_simple -v
"""

import time
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


class TestTLRUEviction(unittest.TestCase):
    """Test cases for Tail-LRU (TLRU) eviction policy."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_cached_tokens_tracking_simple(self):
        """Test that cached_tokens is correctly tracked during insertion."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )

        # Insert a simple sequence
        key1 = RadixKey([1, 2, 3, 4])
        value1 = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        cache.insert(key1, value1)

        # Get the leaf node
        result = cache.match_prefix(key1)
        leaf = result.last_device_node

        # cached_tokens should equal the total number of cached tokens from root to leaf
        self.assertEqual(leaf.cached_tokens, 4)
        # convo_length should also equal the logical conversation length
        self.assertEqual(leaf.convo_length, 4)

    def test_cached_tokens_with_shared_prefix(self):
        """Test cached_tokens with shared prefixes in the tree."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )

        # Insert first sequence
        key1 = RadixKey([1, 2, 3, 4])
        value1 = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        cache.insert(key1, value1)

        # Insert second sequence with shared prefix [1, 2]
        key2 = RadixKey([1, 2, 5, 6])
        value2 = torch.tensor([10, 20, 50, 60], dtype=torch.int64)
        cache.insert(key2, value2)

        # Get both leaf nodes
        result1 = cache.match_prefix(key1)
        leaf1 = result1.last_device_node
        result2 = cache.match_prefix(key2)
        leaf2 = result2.last_device_node

        # Both should have correct cached_tokens (cumulative from root)
        self.assertEqual(leaf1.cached_tokens, 4)
        self.assertEqual(leaf1.convo_length, 4)
        self.assertEqual(leaf2.cached_tokens, 4)
        self.assertEqual(leaf2.convo_length, 4)

    def test_cached_tokens_after_node_split(self):
        """Test that cached_tokens is correctly updated after node splits."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )

        # Insert a long sequence that will create a compressed node
        key1 = RadixKey([1, 2, 3, 4, 5, 6, 7, 8])
        value1 = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64)
        cache.insert(key1, value1)

        # Insert a diverging sequence that will split the node
        key2 = RadixKey([1, 2, 3, 9, 10])
        value2 = torch.tensor([10, 20, 30, 90, 100], dtype=torch.int64)
        cache.insert(key2, value2)

        # Get both leaves and verify cached_tokens
        result1 = cache.match_prefix(key1)
        leaf1 = result1.last_device_node
        result2 = cache.match_prefix(key2)
        leaf2 = result2.last_device_node

        self.assertEqual(leaf1.cached_tokens, 8)
        self.assertEqual(leaf1.convo_length, 8)
        self.assertEqual(leaf2.cached_tokens, 5)
        self.assertEqual(leaf2.convo_length, 5)

    def test_tlru_safe_budget_calculation(self):
        """Test TLRU safe budget calculation logic."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=100,  # ξ = 100
            tlru_next_prompt_estimate=20,  # Q̂ = 20
        )

        # Insert a conversation with 150 tokens
        key = RadixKey(list(range(150)))
        value = torch.arange(150, dtype=torch.int64)
        cache.insert(key, value)

        # Get the leaf node
        result = cache.match_prefix(key)
        leaf = result.last_device_node

        # Verify initial state
        self.assertEqual(leaf.convo_length, 150)
        self.assertEqual(leaf.cached_tokens, 150)

        # Safe budget = max(convo_length + next_prompt_estimate - threshold, 0)
        #              = max(150 + 20 - 100, 0) = 70
        # So we should trim 150 - 70 = 80 tokens

        # Trigger eviction (request more space than available to force TLRU trimming)
        cache.evict(80)

        # After trimming, cached_tokens should be reduced
        # The total cache size should be exactly the safe budget
        self.assertEqual(cache.total_size(), 70)

    def test_convo_length_unchanged_after_trim(self):
        """Test that convo_length remains constant after trimming.

        This is a critical property: convo_length represents the logical
        conversation length and should NEVER change, even when we evict tokens.
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert a conversation
        key = RadixKey(list(range(100)))
        value = torch.arange(100, dtype=torch.int64)
        cache.insert(key, value)

        # Get the leaf node before trimming
        result_before = cache.match_prefix(key)
        leaf_before = result_before.last_device_node
        convo_length_before = leaf_before.convo_length

        # Trigger eviction to force TLRU trimming
        cache.evict(40)  # Evict 40 tokens

        # Get the leaf node after trimming (if it still exists)
        result_after = cache.match_prefix(key)

        # convo_length should remain the same even though tokens were evicted
        if result_after.last_device_node is not None:
            self.assertEqual(
                result_after.last_device_node.convo_length,
                convo_length_before,
                "convo_length should never change after trimming"
            )

    def test_tlru_trimming_updates_cached_tokens(self):
        """Test that TLRU trimming correctly updates cached_tokens.

        When we trim tokens from a node, cached_tokens should decrease
        accordingly, while convo_length stays constant.
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert a long conversation
        key = RadixKey(list(range(100)))
        value = torch.arange(100, dtype=torch.int64)
        cache.insert(key, value)

        # Get initial state
        result = cache.match_prefix(key)
        leaf = result.last_device_node
        initial_cached = leaf.cached_tokens
        initial_convo = leaf.convo_length

        # Safe budget = max(100 + 10 - 50, 0) = 60
        # Should trim 100 - 60 = 40 tokens
        cache.evict(40)

        # Check that cached_tokens was updated but convo_length stayed the same
        result_after = cache.match_prefix(key)
        self.assertEqual(cache.total_size(), 60)

        # If the node still exists, verify cached_tokens decreased
        if result_after.last_device_node is not None:
            leaf_after = result_after.last_device_node
            self.assertEqual(leaf_after.convo_length, initial_convo)
            self.assertLess(leaf_after.cached_tokens, initial_cached)

    def test_tlru_per_conversation_eviction(self):
        """Test that TLRU evicts on a per-conversation basis.

        Each conversation (leaf node) should be evaluated independently
        for its safe budget, even when they share common prefixes.
        """
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert two conversations with shared prefix
        key1 = RadixKey([1, 2] + list(range(100)))  # Long conversation
        value1 = torch.arange(102, dtype=torch.int64)
        cache.insert(key1, value1)

        key2 = RadixKey([1, 2, 999, 888])  # Short conversation sharing [1, 2]
        value2 = torch.tensor([0, 1, 999, 888], dtype=torch.int64)
        cache.insert(key2, value2)

        initial_total = cache.total_size()

        # Trigger eviction - should evaluate each conversation's safe budget independently
        # Conversation 1: convo_length=102, safe_budget = max(102 + 10 - 50, 0) = 62
        # Conversation 2: convo_length=4, safe_budget = max(4 + 10 - 50, 0) = 0
        cache.evict(50)

        # Verify that eviction happened
        self.assertLess(cache.total_size(), initial_total)

    def test_tlru_safe_budget_zero_evicts_all(self):
        """Test that when safe budget is 0, the entire node is evicted."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=100,
            tlru_next_prompt_estimate=10,
        )

        # Insert a short conversation
        # convo_length=20, safe_budget = max(20 + 10 - 100, 0) = 0
        key = RadixKey(list(range(20)))
        value = torch.arange(20, dtype=torch.int64)
        cache.insert(key, value)

        initial_size = cache.total_size()
        self.assertEqual(initial_size, 20)

        # Trigger eviction - should evict the entire node
        cache.evict(20)

        # The node should be completely evicted
        self.assertEqual(cache.total_size(), 0)

    def test_tlru_multiple_conversations_different_budgets(self):
        """Test TLRU with multiple conversations having different safe budgets."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=60,
            tlru_next_prompt_estimate=10,
        )

        # Insert three conversations with different lengths
        # Conv 1: length=100, safe_budget = max(100 + 10 - 60, 0) = 50
        key1 = RadixKey(list(range(100)))
        value1 = torch.arange(100, dtype=torch.int64)
        cache.insert(key1, value1)

        # Conv 2: length=80, safe_budget = max(80 + 10 - 60, 0) = 30
        key2 = RadixKey(list(range(200, 280)))
        value2 = torch.arange(200, 280, dtype=torch.int64)
        cache.insert(key2, value2)

        # Conv 3: length=40, safe_budget = max(40 + 10 - 60, 0) = 0 (evict all)
        key3 = RadixKey(list(range(300, 340)))
        value3 = torch.arange(300, 340, dtype=torch.int64)
        cache.insert(key3, value3)

        initial_total = cache.total_size()
        self.assertEqual(initial_total, 220)  # 100 + 80 + 40

        # Trigger significant eviction
        cache.evict(100)

        # After eviction:
        # - Conv 3 should be completely evicted (safe_budget=0, evict 40)
        # - Conv 2 should be trimmed to 30 (evict 50)
        # - Conv 1 should be trimmed to 50 (evict 50)
        # Total evicted: at least 100
        # Remaining: at most 120 (could be less depending on LRU order)
        self.assertLessEqual(cache.total_size(), 120)

    def test_tlru_tel_trimmed_flag(self):
        """Test that tel_trimmed flag is set correctly after trimming."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert a conversation that will be trimmed
        key = RadixKey(list(range(100)))
        value = torch.arange(100, dtype=torch.int64)
        cache.insert(key, value)

        # Get the node before trimming
        result_before = cache.match_prefix(key)
        leaf_before = result_before.last_device_node
        self.assertFalse(leaf_before.tel_trimmed)

        # Trigger trimming
        cache.evict(40)

        # After trimming, the tel_trimmed flag should be set
        result_after = cache.match_prefix(key)
        if result_after.last_device_node is not None:
            leaf_after = result_after.last_device_node
            self.assertTrue(leaf_after.tel_trimmed)

    def test_tlru_fallback_to_lru(self):
        """Test that TLRU falls back to standard LRU when trimming isn't enough."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=200,  # High threshold = minimal trimming
            tlru_next_prompt_estimate=10,
        )

        # Insert multiple short conversations that won't be trimmed
        # Each has safe_budget = max(20 + 10 - 200, 0) = 0, so they'll be evicted entirely
        for i in range(5):
            key = RadixKey(list(range(i * 100, i * 100 + 20)))
            value = torch.arange(i * 100, i * 100 + 20, dtype=torch.int64)
            cache.insert(key, value)
            time.sleep(0.001)  # Ensure different access times

        initial_total = cache.total_size()
        self.assertEqual(initial_total, 100)  # 5 * 20

        # Request eviction that requires LRU fallback
        cache.evict(60)

        # Should evict multiple nodes using LRU
        self.assertLessEqual(cache.total_size(), 40)


class TestTLRUWithPageSize(unittest.TestCase):
    """Test TLRU behavior with different page sizes."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_tlru_with_page_size_4(self):
        """Test TLRU with page_size=4."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=4,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert a conversation
        # Note: With page_size=4, only complete pages are cached
        key = RadixKey(list(range(100)))
        value = torch.arange(100, dtype=torch.int64)
        cache.insert(key, value)

        # cached_tokens should be aligned to page boundaries
        result = cache.match_prefix(key)
        leaf = result.last_device_node
        self.assertEqual(leaf.cached_tokens % 4, 0)

        # Trigger eviction
        cache.evict(40)

        # Verify eviction happened
        self.assertLess(cache.total_size(), 100)


class TestTLRUvsLRUBehavior(unittest.TestCase):
    """Test that TLRU and LRU make different eviction decisions."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_tlru_vs_lru_eviction_difference(self):
        """Verify that TLRU and LRU make different eviction decisions on same data.

        Setup:
        - Long conversation (200 tokens) accessed first
        - Short conversation (20 tokens) accessed second (more recent)

        Expected behavior:
        - LRU: Evicts the older (long) conversation first based on access time
        - TLRU: Trims the long conversation to its safe budget, keeps more of it
        """
        # Create two caches with the same configuration except eviction policy
        mock_allocator_lru = unittest.mock.Mock()
        mock_allocator_lru.device = torch.device("cpu")

        mock_allocator_tlru = unittest.mock.Mock()
        mock_allocator_tlru.device = torch.device("cpu")

        lru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_lru,
            page_size=1,
            eviction_policy="lru",
        )

        tlru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_tlru,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=100,  # ξ = 100
            tlru_next_prompt_estimate=20,  # Q̂ = 20
        )

        # Insert a long conversation into both caches
        long_key = RadixKey(list(range(200)))
        long_value = torch.arange(200, dtype=torch.int64)
        lru_cache.insert(long_key, long_value)
        tlru_cache.insert(long_key, long_value)

        # Small delay to ensure different access times
        time.sleep(0.01)

        # Insert a short conversation into both caches (accessed more recently)
        short_key = RadixKey(list(range(1000, 1020)))
        short_value = torch.arange(1000, 1020, dtype=torch.int64)
        lru_cache.insert(short_key, short_value)
        tlru_cache.insert(short_key, short_value)

        # Both caches should start with the same total
        self.assertEqual(lru_cache.total_size(), 220)
        self.assertEqual(tlru_cache.total_size(), 220)

        # Trigger eviction of 100 tokens
        lru_cache.evict(100)
        tlru_cache.evict(100)

        # Get final sizes
        lru_final = lru_cache.total_size()
        tlru_final = tlru_cache.total_size()

        print(f"\n=== Eviction Behavior Comparison ===")
        print(f"Initial: 220 tokens (200 long + 20 short)")
        print(f"Evict: 100 tokens")
        print(f"LRU final size: {lru_final} tokens")
        print(f"TLRU final size: {tlru_final} tokens")

        # TLRU should behave differently than LRU
        # TLRU calculates safe budget for long conversation:
        # safe_budget = max(200 + 20 - 100, 0) = 120
        # So it should trim 200 - 120 = 80 tokens from long conversation
        # Then evict the short conversation (20 tokens) to reach 100 total
        # Final TLRU: 120 tokens (just the trimmed long conversation)

        # LRU evicts entire leaf nodes (not partial)
        # When asked to evict 100 tokens, it evicts the oldest (long conversation, 200 tokens)
        # Final LRU: 20 tokens (just the short conversation)

        # The key difference: TLRU trims, LRU evicts entire nodes
        self.assertEqual(tlru_final, 120,
            "TLRU should trim long conversation to safe budget (120 tokens)")

        # LRU evicts entire oldest node (200 tokens), leaving only the short one
        self.assertEqual(lru_final, 20,
            "LRU should evict entire oldest node, leaving only short conversation")

        # Verify what's actually cached to show the behavioral difference
        lru_long_result = lru_cache.match_prefix(long_key)
        tlru_long_result = tlru_cache.match_prefix(long_key)
        lru_short_result = lru_cache.match_prefix(short_key)
        tlru_short_result = tlru_cache.match_prefix(short_key)

        print(f"LRU - Long conversation remaining: {len(lru_long_result.device_indices)} tokens")
        print(f"TLRU - Long conversation remaining: {len(tlru_long_result.device_indices)} tokens")
        print(f"LRU - Short conversation remaining: {len(lru_short_result.device_indices)} tokens")
        print(f"TLRU - Short conversation remaining: {len(tlru_short_result.device_indices)} tokens")

        # Key behavioral difference:
        # LRU: Evicted long (oldest), kept short (newest)
        self.assertEqual(len(lru_long_result.device_indices), 0,
            "LRU should have evicted the long conversation")
        self.assertEqual(len(lru_short_result.device_indices), 20,
            "LRU should have kept the short conversation")

        # TLRU: Trimmed long to safe budget, evicted short
        self.assertEqual(len(tlru_long_result.device_indices), 120,
            "TLRU should keep exactly safe_budget tokens of long conversation")
        self.assertEqual(len(tlru_short_result.device_indices), 0,
            "TLRU should have evicted the short conversation")

        print(f"\n🎯 BEHAVIORAL DIFFERENCE CONFIRMED!")
        print(f"LRU (time-based): Keeps newest (short), evicts oldest (long)")
        print(f"TLRU (tail-optimized): Trims long to safe budget, evicts short")

    def test_tlru_trims_lru_evicts_entirely(self):
        """Verify TLRU trims tails while LRU evicts entire nodes.

        This test ensures TLRU's core behavior: trimming long conversations
        rather than evicting them entirely like LRU does.
        """
        mock_allocator_lru = unittest.mock.Mock()
        mock_allocator_lru.device = torch.device("cpu")

        mock_allocator_tlru = unittest.mock.Mock()
        mock_allocator_tlru.device = torch.device("cpu")

        lru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_lru,
            page_size=1,
            eviction_policy="lru",
        )

        tlru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_tlru,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=50,
            tlru_next_prompt_estimate=10,
        )

        # Insert a single long conversation
        key = RadixKey(list(range(150)))
        value = torch.arange(150, dtype=torch.int64)
        lru_cache.insert(key, value)
        tlru_cache.insert(key, value)

        # Evict 40 tokens
        lru_cache.evict(40)
        tlru_cache.evict(40)

        # LRU: Evicts entire leaf nodes, so it will evict all 150 tokens
        # TLRU: Trims based on safe budget
        # safe_budget = max(150 + 10 - 50, 0) = 110
        # Should trim to 110 tokens (evict 40)

        lru_result = lru_cache.match_prefix(key)
        tlru_result = tlru_cache.match_prefix(key)

        print(f"\n=== Trimming vs Eviction Comparison ===")
        print(f"Original: 150 tokens")
        print(f"Evict: 40 tokens")
        print(f"LRU remaining: {len(lru_result.device_indices)} tokens")
        print(f"TLRU remaining: {len(tlru_result.device_indices)} tokens")
        print(f"TLRU safe budget: 110 tokens")

        # TLRU should have exactly the safe budget
        self.assertEqual(len(tlru_result.device_indices), 110,
            "TLRU should trim to exactly safe_budget")

        # LRU evicts entire nodes, so when asked to evict 40, it evicts the whole 150-token node
        self.assertEqual(len(lru_result.device_indices), 0,
            "LRU evicts entire nodes, so all 150 tokens are evicted")

    def test_tlru_prioritizes_long_conversations(self):
        """Verify TLRU keeps prefixes of long conversations over short ones."""
        mock_allocator_lru = unittest.mock.Mock()
        mock_allocator_lru.device = torch.device("cpu")

        mock_allocator_tlru = unittest.mock.Mock()
        mock_allocator_tlru.device = torch.device("cpu")

        lru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_lru,
            page_size=1,
            eviction_policy="lru",
        )

        tlru_cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator_tlru,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=100,
            tlru_next_prompt_estimate=20,
        )

        # Insert multiple conversations of different lengths
        # All accessed at different times to create clear LRU ordering

        # Conversation 1: Very short (30 tokens) - oldest
        key1 = RadixKey(list(range(30)))
        value1 = torch.arange(30, dtype=torch.int64)
        lru_cache.insert(key1, value1)
        tlru_cache.insert(key1, value1)
        time.sleep(0.01)

        # Conversation 2: Very long (200 tokens) - middle
        key2 = RadixKey(list(range(1000, 1200)))
        value2 = torch.arange(1000, 1200, dtype=torch.int64)
        lru_cache.insert(key2, value2)
        tlru_cache.insert(key2, value2)
        time.sleep(0.01)

        # Conversation 3: Short (40 tokens) - newest
        key3 = RadixKey(list(range(2000, 2040)))
        value3 = torch.arange(2000, 2040, dtype=torch.int64)
        lru_cache.insert(key3, value3)
        tlru_cache.insert(key3, value3)

        initial_total = lru_cache.total_size()
        self.assertEqual(initial_total, 270)  # 30 + 200 + 40

        # Evict 150 tokens
        lru_cache.evict(150)
        tlru_cache.evict(150)

        print(f"\n=== Priority Comparison ===")
        print(f"Initial: Conv1(30) + Conv2(200) + Conv3(40) = 270 tokens")
        print(f"Evict: 150 tokens")

        # Check what each cache kept
        lru_1 = len(lru_cache.match_prefix(key1).device_indices)
        lru_2 = len(lru_cache.match_prefix(key2).device_indices)
        lru_3 = len(lru_cache.match_prefix(key3).device_indices)

        tlru_1 = len(tlru_cache.match_prefix(key1).device_indices)
        tlru_2 = len(tlru_cache.match_prefix(key2).device_indices)
        tlru_3 = len(tlru_cache.match_prefix(key3).device_indices)

        print(f"LRU kept: Conv1={lru_1}, Conv2={lru_2}, Conv3={lru_3}")
        print(f"TLRU kept: Conv1={tlru_1}, Conv2={tlru_2}, Conv3={tlru_3}")

        # TLRU should prioritize keeping the long conversation (trimmed)
        # Conv1: safe_budget = max(30 + 20 - 100, 0) = 0 (evict all)
        # Conv2: safe_budget = max(200 + 20 - 100, 0) = 120 (trim to 120)
        # Conv3: safe_budget = max(40 + 20 - 100, 0) = 0 (evict all)
        # TLRU should keep: 0 + 120 + 0 = 120 tokens (evicted 150)

        self.assertEqual(tlru_1, 0, "TLRU should evict short conversation 1")
        self.assertEqual(tlru_2, 120, "TLRU should trim long conversation to safe budget")
        self.assertEqual(tlru_3, 0, "TLRU should evict short conversation 3")

        # LRU should evict oldest first (Conv1, Conv2 partially)
        # After evicting 150: 270 - 150 = 120 remaining
        # Should evict Conv1 (30) and 120 from Conv2, leaving Conv2(80) + Conv3(40)

        print(f"\nTLRU correctly prioritizes long conversations by keeping safe budget!")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)