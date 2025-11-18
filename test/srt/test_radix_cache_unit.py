"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, RadixKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- RadixKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values
- RadixCache: insert/match operations, eviction, page alignment, error handling
- Cache events and request handling
- Boundary conditions with parameterized testing

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
"""

import time
import unittest
import unittest.mock

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

# Test constants
DEFAULT_PAGE_SIZE = 4


class TestRadixKey(unittest.TestCase):
    """Test cases for RadixKey class."""

    def test_init_basic(self):
        """Test basic initialization of RadixKey."""
        token_ids = [1, 2, 3, 4]
        key = RadixKey(token_ids)
        self.assertEqual(key.token_ids, token_ids)
        self.assertIsNone(key.extra_key)

    def test_init_with_extra_key(self):
        """Test initialization with extra_key."""
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = RadixKey(token_ids, extra_key)
        self.assertEqual(key.token_ids, token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len(self):
        """Test __len__ method."""
        key = RadixKey([1, 2, 3])
        self.assertEqual(len(key), 3)

        empty_key = RadixKey([])
        self.assertEqual(len(empty_key), 0)

    def test_iter(self):
        """Test __iter__ method."""
        token_ids = [1, 2, 3, 4]
        key = RadixKey(token_ids)
        self.assertEqual(list(key), token_ids)

    def test_len_and_iter(self):
        """Test __len__ and __iter__ methods."""
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
        ]

        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                key = RadixKey(tokens)
                self.assertEqual(len(key), expected)
                self.assertEqual(list(key), tokens)

    def test_getitem_int(self):
        """Test __getitem__ with int index."""
        test_cases = [
            ([10, 20, 30], 0, [10]),
            ([10, 20, 30], -1, [30]),
            ([10, 20, 30], 2, [30]),
        ]

        for tokens, index, expected in test_cases:
            with self.subTest(tokens=tokens, index=index):
                key = RadixKey(tokens)
                result = key[index]
                self.assertIsInstance(result, RadixKey)
                self.assertEqual(result.token_ids, expected)

    def test_getitem_slice(self):
        """Test __getitem__ with slice and edge cases."""
        key = RadixKey([1, 2, 3, 4, 5], "extra")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, RadixKey)
        self.assertEqual(sliced.token_ids, [2, 3, 4])
        self.assertEqual(sliced.extra_key, "extra")

        # Edge cases
        self.assertEqual(key[2:2].token_ids, [])  # Empty slice
        self.assertEqual(key[:].token_ids, [1, 2, 3, 4, 5])  # Full slice

    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid indices."""
        key = RadixKey([1, 2, 3])
        with self.assertRaises(IndexError):
            _ = key[10]  # Out of bounds

    def test_repr(self):
        """Test __repr__ method."""
        key = RadixKey([1, 2, 3], "test")
        repr_str = repr(key)
        self.assertIn("RadixKey", repr_str)
        self.assertIn("extra_key='test'", repr_str)
        self.assertIn("[1, 2, 3]", repr_str)

    def test_repr_long_token_ids(self):
        """Test __repr__ with long token_ids."""
        long_tokens = list(range(15))
        key = RadixKey(long_tokens)
        repr_str = repr(key)
        self.assertIn("...", repr_str)  # Should be truncated


class TestTreeNode(unittest.TestCase):
    """Test cases for TreeNode class."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization of TreeNode."""
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """Test initialization with custom ID."""
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_counter_increment(self):
        """Test that counter increments properly."""
        node1 = TreeNode()
        node2 = TreeNode()
        self.assertEqual(node1.id, 0)
        self.assertEqual(node2.id, 1)

    def test_evicted_backuped_properties(self):
        """Test evicted and backuped properties."""
        test_cases = [
            (False, False, True, False),
            (True, False, False, False),
            (True, True, False, True),
            (False, True, True, True),
        ]

        for (
            has_value,
            has_host_value,
            expected_evicted,
            expected_backuped,
        ) in test_cases:
            with self.subTest(has_value=has_value, has_host_value=has_host_value):
                node = TreeNode()

                if has_value:
                    node.value = torch.tensor([1, 2, 3])
                if has_host_value:
                    node.host_value = torch.tensor([4, 5, 6])

                self.assertEqual(node.evicted, expected_evicted)
                self.assertEqual(node.backuped, expected_backuped)

    def test_protect_release_host(self):
        """Test protect_host and release_host methods."""
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)

        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)

        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

        # Test error case
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """Test get_last_hash_value method."""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())

        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_lt_comparison(self):
        """Test less than comparison based on last_access_time."""
        node1 = TreeNode()
        time.sleep(0.001)  # Small delay to ensure different timestamps
        node2 = TreeNode()

        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)


class TestRadixCache(unittest.TestCase):
    """Test cases for RadixCache class."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_init_variations(self):
        """Test cache initialization with different parameters."""
        test_cases = [
            (1, False, False),
            (4, False, True),
            (1, True, False),
        ]

        for page_size, disable, enable_events in test_cases:
            with self.subTest(
                page_size=page_size, disable=disable, enable_events=enable_events
            ):
                cache = RadixCache(
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                    page_size=page_size,
                    disable=disable,
                    enable_kv_cache_events=enable_events,
                )

                self.assertEqual(cache.page_size, page_size)
                self.assertEqual(cache.disable, disable)
                self.assertEqual(cache.enable_kv_cache_events, enable_events)
                self.assertEqual(cache.device, torch.device("cpu"))
                self.assertIsNotNone(cache.root_node)
                self.assertEqual(len(cache.root_node.key), 0)

    def test_reset(self):
        """Test reset method."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        # Insert some data
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertGreater(cache.total_size(), 0)

        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_and_match_basic(self):
        """Test basic insert and match operations."""
        for disable_cache in [False, True]:
            with self.subTest(disable_cache=disable_cache):
                cache = RadixCache(
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                    page_size=1,
                    disable=disable_cache,
                )

                key = RadixKey([1, 2, 3])
                value = torch.tensor([10, 20, 30], dtype=torch.int64)
                prefix_len = cache.insert(key, value)

                if disable_cache:
                    self.assertEqual(prefix_len, 0)
                    self.assertEqual(cache.total_size(), 0)
                    continue

                self.assertEqual(prefix_len, 0)  # No existing prefix
                self.assertEqual(cache.total_size(), 3)
                self.assertEqual(cache.evictable_size(), 3)

                # Test match_prefix
                result = cache.match_prefix(RadixKey([1, 2, 3]))
                self.assertEqual(len(result.device_indices), 3)
                torch.testing.assert_close(result.device_indices, value)

                # Test partial match
                result = cache.match_prefix(RadixKey([1, 2]))
                self.assertEqual(len(result.device_indices), 2)
                torch.testing.assert_close(
                    result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )

    def test_insert_and_match_eagle(self):
        """Test insert and match operations for EAGLE."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
            is_eagle=True,
        )

        key = RadixKey([1, 2, 3, 4])
        value = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        prefix_len = cache.insert(key, value)

        self.assertEqual(prefix_len, 0)  # No existing prefix
        self.assertEqual(
            cache.total_size(), 3
        )  # The last token is ignored in bigram key
        self.assertEqual(cache.evictable_size(), 3)

        # Test match_prefix
        result = cache.match_prefix(RadixKey([1, 2, 3, 4]))
        self.assertEqual(len(result.device_indices), 3)
        torch.testing.assert_close(
            result.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        # Test partial match
        result = cache.match_prefix(RadixKey([1, 2]))
        self.assertEqual(len(result.device_indices), 1)
        torch.testing.assert_close(
            result.device_indices, torch.tensor([10], dtype=torch.int64)
        )

    def test_insert_and_match_eagle_page_size(self):
        """Test insert and match operations for EAGLE and page_size > 1."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=2,
            disable=False,
            is_eagle=True,
        )

        key = RadixKey([1, 2, 3])
        value = torch.tensor([10, 20, 30], dtype=torch.int64)
        prefix_len = cache.insert(key, value)

        self.assertEqual(prefix_len, 0)  # No existing prefix
        self.assertEqual(cache.total_size(), 2)  # only one page is inserted
        self.assertEqual(cache.evictable_size(), 2)

        # Test match_prefix
        result = cache.match_prefix(RadixKey([1, 2, 3, 4]))
        self.assertEqual(len(result.device_indices), 2)
        torch.testing.assert_close(
            result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
        )

        # Test unmatched
        result = cache.match_prefix(RadixKey([1, 2]))
        self.assertEqual(len(result.device_indices), 0)
        torch.testing.assert_close(
            result.device_indices, torch.tensor([], dtype=torch.int64)
        )

    def test_insert_with_none_value(self):
        """Test insert with None value (should use token_ids as list)."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        key = RadixKey([1, 2, 3])
        prefix_len = cache.insert(key, None)

        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """Test total_size calculation."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        self.assertEqual(cache.total_size(), 0)

        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 3)

        cache.insert(RadixKey([4, 5]), torch.tensor([40, 50], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 5)

    def test_kv_cache_events(self):
        """Test KV cache events functionality."""
        test_cases = [
            (1, True),
            (2, True),
            (1, False),
        ]

        for page_size, enable_events in test_cases:
            with self.subTest(page_size=page_size, enable_events=enable_events):
                cache = RadixCache(
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                    page_size=page_size,
                    enable_kv_cache_events=enable_events,
                )

                # Insert data
                cache.insert(RadixKey([1, 2, 3, 4, 5]), None)

                # Take events
                events = cache.take_events()

                if enable_events:
                    self.assertGreater(len(events), 0)
                    # Verify events include BlockStored events (there might be other event types)
                    block_stored_events = [
                        e for e in events if isinstance(e, BlockStored)
                    ]
                    self.assertGreater(len(block_stored_events), 0)
                    for event in block_stored_events:
                        self.assertLessEqual(len(event.token_ids), page_size)
                else:
                    self.assertEqual(len(events), 0)

    def test_kv_cache_events_with_eviction(self):
        """Test KV cache events include removal events."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            enable_kv_cache_events=True,
        )

        # Insert and then evict data
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        cache.evict(3)

        # Take events - should include both store and remove events
        events = cache.take_events()
        self.assertGreater(len(events), 0)

        # Check event types
        event_types = [type(event).__name__ for event in events]
        self.assertIn("BlockStored", event_types)

        # Verify BlockRemoved event content
        remove_events = [e for e in events if isinstance(e, BlockRemoved)]
        for event in remove_events:
            self.assertGreater(len(event.block_hashes), 0)

    def test_extra_key_isolation(self):
        """Test that keys with different extra_key values are isolated."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        # Insert same token sequence with different extra keys
        cache.insert(
            RadixKey([1, 2, 3], "key1"), torch.tensor([10, 20, 30], dtype=torch.int64)
        )
        cache.insert(
            RadixKey([1, 2, 3], "key2"), torch.tensor([40, 50, 60], dtype=torch.int64)
        )
        cache.insert(
            RadixKey([1, 2, 3], None), torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(RadixKey([1, 2, 3], "key1"))
        result2 = cache.match_prefix(RadixKey([1, 2, 3], "key2"))
        result3 = cache.match_prefix(RadixKey([1, 2, 3], None))
        result4 = cache.match_prefix(RadixKey([1, 2, 3], "nonexistent"))

        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(
            result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(
            result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(
            result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_lock_ref_operations(self):
        """Test lock reference counting operations."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        # Insert sequence
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))

        # Get node
        result = cache.match_prefix(RadixKey([1, 2, 3]))
        node = result.last_device_node

        initial_evictable = cache.evictable_size()
        initial_protected = cache.protected_size()

        # Lock the node
        cache.inc_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected + 3)
        self.assertEqual(cache.evictable_size(), initial_evictable - 3)

        # Unlock the node
        cache.dec_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_evict_functionality(self):
        """Test eviction functionality."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
        )

        # Insert sequences
        cache.insert(RadixKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(RadixKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))

        initial_size = cache.total_size()

        # Evict some tokens
        cache.evict(2)

        # Should have called free and reduced size
        mock_allocator.free.assert_called()
        self.assertLess(cache.total_size(), initial_size)

    def test_tlru_trim_respects_tel_budgets(self):
        """TEL trimming should shrink conversations to their safe budgets."""
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

        def insert_conversation(token_base: int, length: int, value_base: int):
            tokens = list(range(token_base, token_base + length))
            values = torch.tensor(
                [value_base + i for i in range(length)], dtype=torch.int64
            )
            cache.insert(RadixKey(tokens), values)
            return tokens, length

        short_tokens, short_len = insert_conversation(10, 4, 1000)
        long_tokens, long_len = insert_conversation(40, 12, 2000)

        def safe_budget(length: int) -> int:
            return max(length + cache.tlru_next_prompt_estimate - cache.tlru_threshold, 0)

        mock_allocator.free.reset_mock()
        cache.evict(6)

        for tokens, logical_len in [(short_tokens, short_len), (long_tokens, long_len)]:
            result = cache.match_prefix(RadixKey(tokens))
            cached_len = len(result.device_indices)
            if cached_len == 0:
                continue
            self.assertLessEqual(cached_len, safe_budget(logical_len))
            self.assertEqual(result.last_device_node.logical_total_tokens, logical_len)

        freed_lengths = [call.args[0].numel() for call in mock_allocator.free.call_args_list]
        self.assertGreaterEqual(sum(freed_lengths), 6)

    def test_tlru_eviction_prefers_trimmed_conversations(self):
        """TEL-trimmed conversations should rank ahead of untouched ones for eviction."""
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

        def insert_conversation(token_base: int, length: int, value_base: int):
            tokens = list(range(token_base, token_base + length))
            values = torch.tensor(
                [value_base + i for i in range(length)], dtype=torch.int64
            )
            cache.insert(RadixKey(tokens), values)
            return tokens, values

        long_tokens, long_values = insert_conversation(0, 12, 3000)
        medium_tokens, medium_values = insert_conversation(50, 9, 2000)
        short_tokens, short_values = insert_conversation(100, 5, 1000)

        def owner_from_tensor(tensor: torch.Tensor) -> str:
            v = tensor.min().item()
            if 1000 <= v < 2000:
                return "short"
            if 2000 <= v < 3000:
                return "medium"
            return "long"

        trimmed_owners = set()
        for _ in range(2):
            mock_allocator.free.reset_mock()
            trimmed = cache._apply_tlru_trimming(cache._collect_leaves(), 3)
            self.assertGreaterEqual(trimmed, 3)
            owners = {owner_from_tensor(call.args[0]) for call in mock_allocator.free.call_args_list}
            trimmed_owners.update(owners)

        self.assertGreaterEqual(len(trimmed_owners), 2)

        prioritized = sorted(
            (
                cache.eviction_strategy.get_priority(node),
                owner_from_tensor(node.value),
            )
            for node in cache._collect_leaves()
            if node != cache.root_node
        )

        top_two_owners = [owner for _prio, owner in prioritized[:2]]
        for owner in top_two_owners:
            self.assertIn(owner, trimmed_owners)

    def test_tlru_zero_budget_drops_conversation(self):
        """If TEL budget is zero the entire conversation should be freed."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=64,
            tlru_next_prompt_estimate=4,
        )

        tokens = list(range(20))
        values = torch.tensor([4000 + i for i in range(20)], dtype=torch.int64)
        cache.insert(RadixKey(tokens), values)

        mock_allocator.free.reset_mock()
        cache.evict(1)

        self.assertEqual(mock_allocator.free.call_count, 1)
        freed = mock_allocator.free.call_args_list[0][0][0]
        self.assertEqual(freed.numel(), len(tokens))

        result = cache.match_prefix(RadixKey(tokens))
        self.assertEqual(len(result.device_indices), 0)

    def test_tlru_split_trim_followed_by_eviction(self):
        """Trimming that splits a node should leave the remainder prioritized for eviction."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            eviction_policy="tlru",
            tlru_threshold=4,
            tlru_next_prompt_estimate=2,
        )

        def insert_conversation(token_base: int, length: int, value_base: int):
            tokens = list(range(token_base, token_base + length))
            values = torch.tensor(
                [value_base + i for i in range(length)], dtype=torch.int64
            )
            cache.insert(RadixKey(tokens), values)
            return tokens, values

        # Conversation that will be partially trimmed (requires split).
        trim_tokens, trim_values = insert_conversation(10, 6, 1000)

        mock_allocator.free.reset_mock()
        cache.evict(2)  # TEL trimming should remove only the tail portion.

        self.assertEqual(mock_allocator.free.call_count, 1)
        trimmed_tail = mock_allocator.free.call_args_list[0][0][0]
        torch.testing.assert_close(trimmed_tail, trim_values[-2:])

        leaves = cache._collect_leaves()
        trimmed_leaf = None
        for node in leaves:
            if node.key.token_ids == trim_tokens[: len(node.key.token_ids)] and len(
                node.key.token_ids
            ) == 4:
                trimmed_leaf = node
                break
        self.assertIsNotNone(trimmed_leaf)
        self.assertTrue(trimmed_leaf.tel_trimmed)
        self.assertEqual(trimmed_leaf.logical_total_tokens, 6)

        result = cache.match_prefix(RadixKey(trim_tokens))
        self.assertEqual(len(result.device_indices), 4)

        # Insert another conversation after trimming to test eviction ordering.
        other_tokens, other_values = insert_conversation(100, 5, 2000)

        def owner_from_tensor(tensor: torch.Tensor) -> str:
            v = tensor.min().item()
            return "trimmed" if 1000 <= v < 2000 else "other"

        mock_allocator.free.reset_mock()
        cache.evict(6)

        freed_blocks = [call.args[0] for call in mock_allocator.free.call_args_list]
        owners = [owner_from_tensor(block) for block in freed_blocks]

        self.assertIn("trimmed", owners)
        trimmed_idx = owners.index("trimmed")
        self.assertGreater(trimmed_idx, 0)
        self.assertEqual(freed_blocks[trimmed_idx].numel(), 4)

    def test_page_alignment_boundary(self):
        """Test page alignment with different sizes."""
        test_cases = [
            (1, 5),
            (2, 5),
            (4, 6),
        ]

        for page_size, sequence_length in test_cases:
            with self.subTest(page_size=page_size, sequence_length=sequence_length):
                cache = RadixCache(
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                    page_size=page_size,
                )

                tokens = list(range(sequence_length))
                cache.insert(RadixKey(tokens), torch.tensor(tokens, dtype=torch.int64))

                result = cache.match_prefix(RadixKey(tokens))
                self.assertGreater(len(result.device_indices), 0)

                # Match length should be page-aligned
                match_len = len(result.device_indices)
                self.assertEqual(match_len % page_size, 0)

    def test_pretty_print_basic(self):
        """Test pretty_print produces output."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))

        # Just test that it doesn't crash
        try:
            cache.pretty_print()
        except Exception as e:
            self.fail(f"pretty_print raised an exception: {e}")

    def test_all_values_flatten(self):
        """Test all_values_flatten method."""
        cache = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=1
        )

        cache.insert(RadixKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(RadixKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))

        all_values = cache.all_values_flatten()
        self.assertEqual(len(all_values), 4)
        # Values should contain all inserted values (order may vary)
        values_set = set(all_values.tolist())
        self.assertEqual(values_set, {10, 20, 30, 40})

    def test_advanced_prefix_match_with_node_splits(self):
        """Advanced prefix matching: splits inside nodes and across pages."""
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache(
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                    page_size=page_size,
                )

                # Insert a long sequence that will be split later.
                seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
                val1 = torch.tensor([x * 10 for x in seq1], dtype=torch.int64)
                cache.insert(RadixKey(seq1), val1)

                # Insert a diverging branch to create an internal node on the path.
                seq2 = [1, 2, 9, 10]
                val2 = torch.tensor([x * 10 for x in seq2], dtype=torch.int64)
                cache.insert(RadixKey(seq2), val2)
                print(cache.pretty_print())

                baseline_total = cache.total_size()
                expected_total = 10  # 8 + 2
                self.assertEqual(baseline_total, expected_total)

                # Match that causes a split inside an existing node:
                # take first 4 tokens of seq1, then diverge.
                query1 = [1, 2, 3, 4, 999, 1000]
                result1 = cache.match_prefix(RadixKey(query1))
                torch.testing.assert_close(result1.device_indices, val1[:4])
                # No data change after structural split during matching.
                self.assertEqual(cache.total_size(), baseline_total)

                # Full match of the long sequence still returns the full indices.
                result_full = cache.match_prefix(RadixKey(seq1))
                torch.testing.assert_close(result_full.device_indices, val1)

                # Another split deeper on the path (after matching 6 tokens, then diverge).
                query2 = [1, 2, 3, 4, 5, 6, 777, 888]
                result2 = cache.match_prefix(RadixKey(query2))
                torch.testing.assert_close(result2.device_indices, val1[:6])
                self.assertEqual(cache.total_size(), baseline_total)

                # Matching the short diverging branch should return exactly its indices.
                result_branch = cache.match_prefix(RadixKey(seq2))
                torch.testing.assert_close(result_branch.device_indices, val2)


if __name__ == "__main__":
    unittest.main()
