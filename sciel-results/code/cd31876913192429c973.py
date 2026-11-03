"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict
from math import comb


def _all_blocks(v, k):
    """Yield all k-subsets of range(v) as tuples."""
    return itertools.combinations(range(v), k)


def _triple_index(triple, v):
    """Map a triple (a,b,c) with 0<=a<b<c<v to a unique integer index."""
    # We need to map C(v,3) triples to 0..C(v,3)-1
    # Use a simple approach: precompute or use a formula
    # For v=24, C(24,3)=2024, so we can use a lookup table
    return _triple_idx[triple]


# Precompute triple indexing for v=24 (the expected case)
# We'll build this lazily since v might vary
_triple_idx = {}
_triple_list = []
_v_precomputed = None


def _precompute_triples(v):
    global _triple_idx, _triple_list, _v_precomputed
    if _v_precomputed == v:
        return
    _triple_idx = {}
    _triple_list = []
    for i, triple in enumerate(itertools.combinations(range(v), t_global)):
        _triple_idx[triple] = i
        _triple_list.append(triple)
    _v_precomputed = v


t_global = 3  # will be set properly


def construct(v, k, t):
    global _v_precomputed, _triple_idx, _triple_list, t_global
    t_global = t
    
    # Deadline: 110 seconds to be safe
    deadline = time.time() + 110.0
    
    # Precompute all triples and their indices
    _precompute_triples(v)
    num_triples = len(_triple_list)
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, compute which triples it covers
    # block_triples[block_idx] = set of triple indices covered
    block_triples = []
    for block in all_blocks:
        covered = set()
        for triple in itertools.combinations(block, t):
            covered.add(_triple_idx[triple])
        block_triples.append(covered)
    
    # Greedy covering with multiple strategies
    def greedy_cover(uncovered_set, remaining_blocks, block_triples):
        """Greedy: repeatedly pick the block covering most uncovered triples."""
        blocks_chosen = []
        uncovered = set(uncovered_set)
        # For efficiency, track count of uncovered triples per block
        # block_cover_count[b] = number of uncovered triples in block b
        block_cover_count = [len(block_triples[b] & uncovered) for b in remaining_blocks]
        
        while uncovered:
            # Find block with max cover count
            best_b = max(remaining_blocks, key=lambda b: block_cover_count[b])
            if block_cover_count[best_b] == 0:
                break
            blocks_chosen.append(best_b)
            # Update: remove covered triples, update counts
            covered = block_triples[best_b]
            for tri in covered:
                if tri in uncovered:
                    uncovered.discard(tri)
            # Update block_cover_count for remaining blocks
            for b in remaining_blocks:
                if b == best_b:
                    continue
                # Count how many of b's triples are in covered
                overlap = len(block_triples[b] & covered)
                if overlap > 0:
                    block_cover_count[b] -= overlap
            remaining_blocks = [b for b in remaining_blocks if b != best_b]
            if not uncovered:
                break
        return blocks_chosen, uncovered
    
    # Strategy 1: Standard greedy
    uncovered_full = set(range(num_triples))
    all_block_indices = list(range(num_blocks))
    
    chosen, remaining_uncovered = greedy_cover(uncovered_full, all_block_indices, block_triples)
    
    # Convert to block lists
    def blocks_to_lists(block_indices):
        return [list(all_blocks[i]) for i in block_indices]
    
    result_blocks = blocks_to_lists(chosen)
    current_count = len(result_blocks)
    
    # If we haven't finished covering, we need a fallback
    if remaining_uncovered:
        # Add blocks to cover remaining triples (simple greedy on remaining)
        uncovered = set(remaining_uncovered)
        extra_blocks = []
        while uncovered and time.time() < deadline - 5:
            # Find block covering most uncovered
            best_b = None
            best_count = 0
            for b in range(num_blocks):
                if b in chosen or b in extra_blocks:
                    continue
                count = len(block_triples[b] & uncovered)
                if count > best_count:
                    best_count = count
                    best_b = b
            if best_b is None or best_count == 0:
                # Force add a block from uncovered triple
                tri = next(iter(uncovered))
                # Find any block containing this triple
                for b in range(num_blocks):
                    if tri in block_triples[b]:
                        best_b = b
                        break
                if best_b is None:
                    break
            extra_blocks.append(best_b)
            covered = block_triples[best_b]
            uncovered -= covered
        result_blocks = blocks_to_lists(chosen + extra_blocks)
        current_count = len(result_blocks)
    
    best_result = result_blocks
    best_count = current_count
    
    # Local search: try to replace blocks to reduce count
    # Simple approach: try removing each block and see if we can re-cover with fewer
    if time.time() < deadline - 10:
        # Try 1-for-1 swaps: remove block i, see if remaining still covers
        # Then try to add back a better block
        used_block_set = set(chosen + extra_blocks if remaining_uncovered else chosen)
        
        for attempt in range(50):  # Limited attempts
            if time.time() > deadline - 5:
                break
            
            # Try removing a random block and re-greedy the rest
            test_indices = list(used_block_set)
            if len(test_indices) <= 1:
                break
            remove_idx = random.choice(test_indices)
            new_indices = [b for b in test_indices if b != remove_idx]
            
            # Check if new_indices still covers all triples
            covered_triples = set()
            for b in new_indices:
                covered_triples |= block_triples[b]
            
            if len(covered_triples) == num_triples:
                # We can remove this block!
                used_block_set.discard(remove_idx)
                best_count -= 1
                best_result = blocks_to_lists(list(used_block_set))
            else:
                # Try to find a single block that covers the missing triples
                missing = set(range(num_triples)) - covered_triples
                if len(missing) == 0:
                    continue
                # Find block covering most missing
                best_new_b = None
                best_new_count = 0
                for b in range(num_blocks):
                    if b in new_indices:
                        continue
                    count = len(block_triples[b] & missing)
                    if count > best_new_count:
                        best_new_count = count
                        best_new_b = b
                if best_new_b is not None and best_new_count == len(missing):
                    # This single block covers all missing
                    new_set = set(new_indices) | {best_new_b}
                    if len(new_set) < len(used_block_set):
                        used_block_set = new_set
                        best_count = len(used_block_set)
                        best_result = blocks_to_lists(list(used_block_set))
    
    # Final local search: try to replace two blocks with one
    if time.time() < deadline - 3:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            if len(used_block_set) <= 2:
                break
            # Pick two random blocks
            test_list = list(used_block_set)
            if len(test_list) < 2:
                break
            i, j = random.sample(range(len(test_list)), 2)
            bi, bj = test_list[i], test_list[j]
            # Try replacing bi and bj with a single block
            covered_by_ij = block_triples[bi] | block_triples[bj]
            # Find a block that covers all of covered_by_ij
            found = False
            for b in range(num_blocks):
                if b in used_block_set and b != bi and b != bj:
                    continue
                if covered_by_ij <= block_triples[b]:
                    new_set = (used_block_set - {bi, bj}) | {b}
                    if len(new_set) < len(used_block_set):
                        used_block_set = new_set
                        best_count = len(used_block_set)
                        best_result = blocks_to_lists(list(used_block_set))
                    found = True
                    break
            if found:
                continue
    
    # Verify coverage
    final_covered = set()
    for b in used_block_set:
        final_covered |= block_triples[b]
    
    if len(final_covered) < num_triples:
        # Fallback to the greedy result which should be complete
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
