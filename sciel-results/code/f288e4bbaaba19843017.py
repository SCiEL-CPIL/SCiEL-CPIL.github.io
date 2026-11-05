"""
Covering design C(24,7,3) via randomized greedy + local search.
"""
import itertools
import random
import time
from collections import Counter

# Precompute all 3-subsets of 24 points
ALL_TRIPLES = list(itertools.combinations(range(24), 3))
NUM_TRIPLES = len(ALL_TRIPLES)  # 2024

# Precompute for each point, the list of triples containing it
POINT_TRIPLES = [[] for _ in range(24)]
for triple in ALL_TRIPLES:
    for p in triple:
        POINT_TRIPLES[p].append(triple)

# Precompute for each pair of points, the list of triples containing both
PAIR_TRIPLES = {}
for triple in ALL_TRIPLES:
    for i in range(3):
        for j in range(i+1, 3):
            pair = (triple[i], triple[j])
            if pair not in PAIR_TRIPLES:
                PAIR_TRIPLES[pair] = []
            PAIR_TRIPLES[pair].append(triple)


def _triple_in_block(triple, block_set):
    """Check if all elements of triple are in block_set."""
    return all(p in block_set for p in triple)


def _count_covered_triples(block_set):
    """Count how many of ALL_TRIPLES are covered by block_set."""
    count = 0
    for triple in ALL_TRIPLES:
        if all(p in block_set for p in triple):
            count += 1
    return count


def _greedy_cover(v, k, t, rng, deadline, max_blocks=None):
    """
    Randomized greedy: repeatedly pick an uncovered triple (randomly among the
    first few uncovered for diversity), then build a block of size k around it
    by greedily adding the point that covers the most currently-uncovered triples.
    
    Returns (blocks, covered_count) where covered_count is the number of distinct
    t-subsets covered.
    """
    # uncovered is a set of triples not yet covered
    uncovered = set(ALL_TRIPLES)
    blocks = []
    if max_blocks is None:
        max_blocks = 200  # safety
    
    while uncovered and len(blocks) < max_blocks:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered triple to anchor the block
        unc_list = list(uncovered)
        # For diversity, pick from a random subset of uncovered triples
        sample_size = min(len(unc_list), 50)
        chosen = rng.sample(unc_list, sample_size)
        anchor = chosen[0]
        
        block = list(anchor)
        block_set = set(block)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            if time.time() > deadline:
                break
            best_point = -1
            best_gain = -1
            candidates = [p for p in range(v) if p not in block_set]
            rng.shuffle(candidates)
            
            for p in candidates:
                # Compute gain: number of uncovered triples that would be covered
                # if we add p to the block
                gain = 0
                for triple in uncovered:
                    if p in triple:
                        # Check if the other t-1 elements are already in block_set
                        others = [x for x in triple if x != p]
                        if all(o in block_set for o in others):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
                    if gain == 0:
                        break  # No point gives positive gain, break early
            
            if best_point == -1 or best_gain == 0:
                # Pick a random candidate
                best_point = rng.choice(candidates)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in ALL_TRIPLES:
            if all(p in block_set for p in triple):
                uncovered.discard(triple)
    
    # Count covered triples
    covered = NUM_TRIPLES - len(uncovered)
    return blocks, covered


def _local_search(blocks, v, k, t, rng, deadline, max_iterations=500):
    """
    Local search: try to replace a block with a different block that covers
    at least as many triples, potentially freeing up another block for removal.
    
    Strategy: 
    1. Compute for each block, the set of triples it uniquely covers (not covered by others).
    2. Try to remove a block if its uniquely covered triples are also covered by others.
    3. Try to swap a block with a better one.
    """
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    if n == 0:
        return blocks
    
    # Precompute coverage: for each triple, which blocks cover it
    # covered_by[triple_idx] = set of block indices
    triple_to_blocks = {}
    for i, block in enumerate(blocks):
        block_set = set(block)
        for triple in ALL_TRIPLES:
            if all(p in block_set for p in triple):
                if triple not in triple_to_blocks:
                    triple_to_blocks[triple] = set()
                triple_to_blocks[triple].add(i)
    
    # Count how many triples are covered
    total_covered = len(triple_to_blocks)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        # Try to remove each block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if removing block i leaves all triples covered
            # A triple is uniquely covered by block i if no other block covers it
            can_remove = True
            for triple, covering_blocks in triple_to_blocks.items():
                if covering_blocks == {i}:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block i
                blocks.pop(i)
                # Update triple_to_blocks
                for triple in list(triple_to_blocks.keys()):
                    triple_to_blocks[triple].discard(i)
                    if not triple_to_blocks[triple]:
                        del triple_to_blocks[triple]
                total_covered = len(triple_to_blocks)
                improved = True
                break  # Restart the loop since indices changed
    
    return blocks


def _verify_covering(blocks, v, k, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        for triple in itertools.combinations(block, t):
            covered.add(triple)
    return len(covered) == NUM_TRIPLES


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    if v != 24 or k != 7 or t != 3:
        # Fallback: simple greedy
        return _simple_greedy(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized greedy attempts
    num_attempts = 8
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        rng = random.Random(attempt * 12345 + 67890)
        blocks, covered = _greedy_cover(v, k, t, rng, deadline)
        
        # Verify
        if not _verify_covering(blocks, v, k, t):
            # Try to fix by adding more blocks
            uncovered = set(ALL_TRIPLES)
            for block in blocks:
                block_set = set(block)
                for triple in ALL_TRIPLES:
                    if all(p in block_set for p in triple):
                        uncovered.discard(triple)
            
            while uncovered and time.time() < deadline:
                anchor = min(uncovered)
                block = list(anchor)
                block_set = set(block)
                while len(block) < k:
                    best_point = -1
                    best_gain = -1
                    for p in range(v):
                        if p in block_set:
                            continue
                        gain = 0
                        for triple in uncovered:
                            if p in triple:
                                others = [x for x in triple if x != p]
                                if all(o in block_set for o in others):
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point == -1:
                        best_point = next(p for p in range(v) if p not in block_set)
                    block.append(best_point)
                    block_set.add(best_point)
                block.sort()
                blocks.append(block)
                for triple in ALL_TRIPLES:
                    if all(p in block_set for p in triple):
                        uncovered.discard(triple)
        
        count = len(blocks)
        if count < best_count:
            best_count = count
            best_blocks = [list(b) for b in blocks]
        
        # Local search to try to improve
        if time.time() < deadline - 10:
            improved_blocks = _local_search(best_blocks, v, k, t, rng, deadline - 5, max_iterations=200)
            if len(improved_blocks) < best_count:
                if _verify_covering(improved_blocks, v, k, t):
                    best_count = len(improved_blocks)
                    best_blocks = improved_blocks
    
    if best_blocks is None:
        # Fallback
        best_blocks = _simple_greedy(v, k, t)
    
    return best_blocks


def _simple_greedy(v, k, t):
    """Simple deterministic greedy fallback."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
