"""Branch-and-bound exact cover for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from collections import defaultdict


def _greedy_cover(v, k, t, rng):
    """Greedy construction to get an initial valid covering."""
    all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_triples)
    blocks = []
    block_set = set()
    
    # Precompute for each point, which triples it's in
    point_triples = defaultdict(list)
    for i, tri in enumerate(all_triples):
        for p in tri:
            point_triples[p].append(i)
    
    while uncovered:
        # Pick the most constrained uncovered triple (one with fewest candidate blocks)
        best_triple = None
        best_count = float('inf')
        
        # Sample some uncovered triples to find a good one
        sample_size = min(50, len(uncovered))
        sample = rng.sample(list(uncovered), sample_size)
        
        for tri in sample:
            # Count how many k-blocks contain this triple
            count = 0
            for p in tri:
                for other_p in tri:
                    if other_p != p:
                        for tri2 in point_triples[p]:
                            t2 = all_triples[tri2]
                            if p in t2 and other_p in t2:
                                count += 1
                                break
                        break
            if count < best_count:
                best_count = count
                best_triple = tri
        
        if best_triple is None:
            best_triple = rng.choice(list(uncovered))
        
        # Find all k-blocks containing best_triple
        triple_set = set(best_triple)
        candidates = []
        for combo in itertools.combinations(range(v), k):
            if triple_set.issubset(set(combo)):
                candidates.append(combo)
        
        if not candidates:
            continue
        
        # Pick the candidate that covers the most uncovered triples
        best_block = None
        best_gain = -1
        for cand in candidates:
            cand_set = set(cand)
            gain = 0
            for tri in itertools.combinations(cand, t):
                if frozenset(tri) in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = cand
        
        if best_block is not None:
            best_block = tuple(sorted(best_block))
            if best_block not in block_set:
                block_set.add(best_block)
                blocks.append(best_block)
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(frozenset(tri))
    
    return blocks


def _exact_cover_branch_and_bound(v, k, t, initial_blocks, time_limit, start_time):
    """
    Branch-and-bound search for minimum covering.
    Uses the fact that we need to cover all C(v,t) triples.
    """
    all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute all k-blocks and which triples they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_triples = []
    for b in all_k_blocks:
        tris = [triple_to_idx[frozenset(c)] for c in itertools.combinations(b, t)]
        block_triples.append(set(tris))
    
    n_blocks = len(all_k_blocks)
    
    # For each triple, which blocks cover it
    triple_to_blocks = defaultdict(list)
    for i, btris in enumerate(block_triples):
        for tri_idx in btris:
            triple_to_blocks[tri_idx].append(i)
    
    # Initial solution
    best_solution = list(initial_blocks)
    best_count = len(initial_blocks)
    
    # State for search
    # covered[tri_idx] = number of blocks in current solution covering this triple
    covered = [0] * n_triples
    for b in initial_blocks:
        b_idx = all_k_blocks.index(b)
        for tri_idx in block_triples[b_idx]:
            covered[tri_idx] += 1
    
    # Track which blocks are in the solution
    in_solution = [False] * n_blocks
    for b in initial_blocks:
        b_idx = all_k_blocks.index(b)
        in_solution[b_idx] = True
    
    # Lower bound: at least ceil(uncovered_triples / max_triples_per_block) more blocks needed
    max_triples_per_block = len(block_triples[0]) if block_triples else 0
    
    # Branch and bound search
    # We'll try removing blocks and see if we can still cover everything
    # This is equivalent to finding a minimum subset of blocks that covers all triples
    
    # Sort blocks by how "critical" they are (blocks covering unique triples first)
    block_criticality = []
    for i, btris in enumerate(block_triples):
        # Count triples only covered by this block (in the full set)
        unique_count = 0
        for tri_idx in btris:
            if len(triple_to_blocks[tri_idx]) == 1:
                unique_count += 1
        block_criticality.append((unique_count, i))
    
    block_criticality.sort(reverse=True)
    
    # Try to remove blocks one by one, starting with least critical
    current_blocks = list(in_solution)
    current_count = sum(current_blocks)
    
    # Heuristic: try removing blocks that don't cover any unique triples
    removable = []
    for i in range(n_blocks):
        if current_blocks[i]:
            btris = block_triples[i]
            can_remove = True
            for tri_idx in btris:
                if covered[tri_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removable.append(i)
    
    # Remove all possible blocks
    for i in removable:
        current_blocks[i] = False
        for tri_idx in block_triples[i]:
            covered[tri_idx] -= 1
        current_count -= 1
    
    if current_count < best_count:
        best_count = current_count
        best_solution = [all_k_blocks[i] for i in range(n_blocks) if current_blocks[i]]
    
    # Now try adding back blocks strategically to see if we can do better
    # This is a simplified version - in practice we'd need more sophisticated search
    
    return best_solution


def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # leave 10s buffer for 150s total
    
    rng = random.Random(42)
    
    # Get initial greedy covering
    initial_blocks = _greedy_cover(v, k, t, rng)
    
    # Try branch and bound to improve
    if time.time() - start_time < time_limit - 5:
        improved = _exact_cover_branch_and_bound(v, k, t, initial_blocks, time_limit, start_time)
        if len(improved) < len(initial_blocks):
            initial_blocks = improved
    
    return [list(b) for b in initial_blocks]
# EVOLVE-BLOCK-END
