import itertools
import random
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """
    Covering design C(v, k, t): find as few k-subsets of {0,...,v-1} as possible
    such that every t-subset is contained in at least one block.
    
    Strategy:
    1. Bitset-based greedy construction with multiple random restarts
    2. Local search: remove redundant blocks
    3. Local search: replace blocks to improve coverage
    4. Time-limited with explicit deadline guard
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # All t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to its index
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point x, precompute the bitset of t-subsets containing x
    # Use numpy uint64 arrays for fast bitset operations
    n_bits = n_tsubsets
    n_words = (n_bits + 63) // 64
    
    # tsubsets_with_point_bits[x] = list of uint64 words representing bitset
    tsubsets_with_point_bits = [np.zeros(n_words, dtype=np.uint64) for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        word_idx = i // 64
        bit_idx = i % 64
        for x in ts:
            tsubsets_with_point_bits[x][word_idx] |= (1 << bit_idx)
    
    # For each pair (x, y), precompute bitset of t-subsets containing both
    pair_bits = {}
    for x in range(v):
        for y in range(x + 1, v):
            # t-subsets containing both x and y
            bits = np.zeros(n_words, dtype=np.uint64)
            for w in range(n_words):
                bits[w] = tsubsets_with_point_bits[x][w] & tsubsets_with_point_bits[y][w]
            pair_bits[(x, y)] = bits
    
    def count_bits(bitset):
        """Count set bits in a numpy uint64 array."""
        return int(np.sum(bitset.astype(np.uint64).view(np.uint64) & np.uint64(0x5555555555555555)))
    
    # Actually, let's use a simpler approach: just count using Python's bit_count
    def count_bits_list(bitset_list):
        """Count set bits in a list of uint64."""
        count = 0
        for w in bitset_list:
            count += bin(int(w)).count('1')
        return count
    
    def greedy_cover(seed_order=None):
        """Run greedy construction with optional randomization."""
        if seed_order is None:
            seed_order = list(range(v))
            random.shuffle(seed_order)
        
        # uncovered: bitset of uncovered t-subsets
        uncovered = np.ones(n_words, dtype=np.uint64)
        # Set all bits for indices < n_tsubsets
        for i in range(n_tsubsets):
            word_idx = i // 64
            bit_idx = i % 64
            uncovered[word_idx] |= (1 << bit_idx)
        # Clear bits beyond n_tsubsets in the last word
        if n_tsubsets % 64 != 0:
            last_word_mask = (1 << (n_tsubsets % 64)) - 1
            uncovered[-1] &= last_word_mask
        
        blocks = []
        remaining_points = set(range(v))
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Check time
            if time.time() > deadline - 30:
                break
            
            # Find the first uncovered t-subset
            min_idx = None
            for w in range(n_words):
                if uncovered[w] != 0:
                    # Find lowest set bit
                    word_val = int(uncovered[w])
                    bit_pos = (word_val & -word_val).bit_length() - 1
                    min_idx = w * 64 + bit_pos
                    break
            
            if min_idx is None:
                break
            
            # Start block with the elements of this t-subset
            start_ts = all_tsubsets[min_idx]
            block = list(start_ts)
            block_set = set(block)
            
            # Extend block to size k using greedy gain
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                for x in remaining_points:
                    if x in block_set:
                        continue
                    
                    # Compute gain: number of uncovered t-subsets that would be covered
                    # by adding x to current block
                    # A t-subset is covered if it contains all elements of (block + [x])
                    # We need to count uncovered t-subsets that contain all of block ∪ {x}
                    
                    if len(block) == 0:
                        # Count uncovered t-subsets containing x
                        gain = 0
                        for w in range(n_words):
                            gain += bin(int(tsubsets_with_point_bits[x][w] & uncovered[w])).count('1')
                    else:
                        # Count uncovered t-subsets containing all of block and x
                        # Start with t-subsets containing x
                        # Then intersect with t-subsets containing each element of block
                        bits = tsubsets_with_point_bits[x].copy()
                        for b in block:
                            bits = bits & tsubsets_with_point_bits[b]
                        gain = 0
                        for w in range(n_words):
                            gain += bin(int(bits[w] & uncovered[w])).count('1')
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            
            # Add block and update uncovered
            if block not in blocks:
                blocks.append(block)
                # Clear bits for all t-subsets contained in this block
                for ts in itertools.combinations(block, t):
                    idx = tsubset_index[tuple(ts)]
                    word_idx = idx // 64
                    bit_idx = idx % 64
                    uncovered[word_idx] &= ~(1 << bit_idx)
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove redundant blocks from the list."""
        if not blocks:
            return blocks
        
        # Build coverage counts
        # For each t-subset, count how many blocks cover it
        coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsubs = []
        
        for block in blocks:
            tsubs = set()
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[tuple(ts)]
                tsubs.add(idx)
                coverage_count[idx] += 1
            block_tsubs.append(tsubs)
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for bi in indices:
                # Check if any t-subset in this block has coverage == 1
                can_remove = True
                for idx in block_tsubs[bi]:
                    if coverage_count[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for idx in block_tsubs[bi]:
                        coverage_count[idx] -= 1
                    block_tsubs.pop(bi)
                    blocks.pop(bi)
                    changed = True
                    break
        
        return blocks
    
    def replace_block_search(blocks, max_iter=100):
        """Try to replace blocks with better ones."""
        if not blocks:
            return blocks
        
        # Build coverage
        coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            tsubs = set()
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[tuple(ts)]
                tsubs.add(idx)
                coverage_count[idx] += 1
            block_tsubs.append(tsubs)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline - 20:
            improved = False
            iterations += 1
            
            # Try replacing each block
            for bi in range(len(blocks)):
                # Find t-subsets uniquely covered by this block
                unique_tsubs = set()
                for idx in block_tsubs[bi]:
                    if coverage_count[idx] == 1:
                        unique_tsubs.add(idx)
                
                if not unique_tsubs:
                    continue
                
                # Try to find a new block that covers all unique_tsubs
                # and some additional t-subsets
                best_new_block = None
                best_gain = 0
                
                # Try several random blocks
                for _ in range(20):
                    # Create a random block that contains all elements from unique_tsubs
                    required_points = set()
                    for idx in unique_tsubs:
                        required_points.update(all_tsubsets[idx])
                    
                    if len(required_points) > k:
                        continue
                    
                    # Add random points to reach size k
                    new_block = list(required_points)
                    remaining = [x for x in range(v) if x not in required_points]
                    random.shuffle(remaining)
                    new_block.extend(remaining[:k - len(new_block)])
                    new_block.sort()
                    
                    # Calculate gain
                    new_tsubs = set()
                    for ts in itertools.combinations(new_block, t):
                        new_tsubs.add(tsubset_index[tuple(ts)])
                    
                    # Gain = number of t-subsets covered by new block but not by old block
                    gain = len(new_tsubs - block_tsubs[bi])
                    
                    # But we must ensure all unique_tsubs are still covered
                    if not unique_tsubs.issubset(new_tsubs):
                        continue
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_new_block = new_block
                
                if best_new_block is not None and best_gain > 0:
                    # Update coverage
                    for idx in block_tsubs[bi]:
                        coverage_count[idx] -= 1
                    for idx in [tsubset_index[tuple(ts)] for ts in itertools.combinations(best_new_block, t)]:
                        coverage_count[idx] += 1
                    
                    blocks[bi] = best_new_block
                    block_tsubs[bi] = set(tsubset_index[tuple(ts)] for ts in itertools.combinations(best_new_block, t))
                    improved = True
        
        return blocks
    
    # Main algorithm: multiple greedy restarts, keep best
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 40:
        # Run greedy with randomization
        blocks = greedy_cover()
        blocks = remove_redundant(blocks)
        
        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = blocks[:]
        
        num_restarts += 1
        
        # Try local search on the best solution
        if best_blocks is not None and time.time() < deadline - 25:
            candidate = best_blocks[:]
            candidate = remove_redundant(candidate)
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_blocks = candidate[:]
            
            candidate = replace_block_search(candidate, max_iter=50)
            candidate = remove_redundant(candidate)
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_blocks = candidate[:]
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_cover()
    
    # Final verification
    final_covered = np.zeros(n_tsubsets, dtype=np.bool_)
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            final_covered[tsubset_index[tuple(ts)]] = True
    
    if not np.all(final_covered):
        # Add blocks for uncovered t-subsets
        for i in range(n_tsubsets):
            if not final_covered[i]:
                ts = all_tsubsets[i]
                remaining = [x for x in range(v) if x not in ts]
                random.shuffle(remaining)
                block = list(ts) + remaining[:k - t]
                block.sort()
                best_blocks.append(block)
    
    return best_blocks
