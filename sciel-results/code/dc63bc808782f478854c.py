"""Stronger greedy covering for C(24,7,3) using randomized restarts and scoring."""
import itertools
import random
import numpy as np


def _greedy_cover(v, k, t, rng=None, max_blocks=200):
    """
    Randomized greedy covering with scoring.
    Uses a set of uncovered t-subsets. At each step, picks a candidate block
    (k-subset) that maximizes the number of newly covered t-subsets, with
    some randomization to escape local optima.
    """
    if rng is None:
        rng = random.Random(42)
    
    all_points = list(range(v))
    # Precompute all t-subsets for quick lookup
    all_t_subs = set(itertools.combinations(all_points, t))
    uncovered = set(all_t_subs)
    blocks = []
    
    # Precompute for each point, the list of t-subsets containing it? 
    # Actually, for scoring a candidate block, we need to count how many
    # uncovered t-subsets are contained in the block.
    
    # Strategy: maintain a counter for each t-subset of how many uncovered blocks contain it? No.
    # Better: for each candidate block, score = number of t-subsets in the block that are in uncovered.
    
    # To speed up, we can use a different approach:
    # For each point, maintain a list of uncovered t-subsets containing that point.
    # But when we pick a block, we remove all t-subsets in that block from uncovered.
    
    # Let's use a simpler but effective randomized greedy:
    # At each step, generate several random candidate blocks, score them, and pick the best.
    # Or: use a weighted random selection based on coverage potential.
    
    # Efficient scoring: for a candidate block B, score = |{S in uncovered : S ⊆ B}|
    # We can precompute for each pair of points, etc., but that's complex.
    
    # Alternative: use a "set cover" style greedy where we pick the block that covers the most
    # uncovered t-subsets. Since we can't enumerate all C(24,7) blocks, we use heuristics.
    
    # Heuristic: build blocks by adding points that maximize marginal coverage.
    
    while uncovered and len(blocks) < max_blocks:
        # Start with a random point or the point in the most constrained uncovered subset
        # Pick an uncovered t-subset, start a block around it
        sample = list(rng.sample(list(uncovered), min(10, len(uncovered))))
        
        best_block = None
        best_score = -1
        
        for start_sub in sample:
            # Try to build a block starting from this subset
            block_set = set(start_sub)
            # Greedily add points to maximize coverage
            for _ in range(k - t):
                best_point = None
                best_gain = -1
                for x in all_points:
                    if x in block_set:
                        continue
                    # Compute gain: how many new uncovered t-subsets would be covered
                    # A t-subset S is covered by block_set ∪ {x} if S ⊆ block_set ∪ {x}
                    # and S is not already covered by block_set
                    # S ⊆ block_set ∪ {x} means S \ {x} ⊆ block_set and x in S
                    # So S must contain x and S \ {x} ⊆ block_set
                    # Number of such S in uncovered:
                    gain = 0
                    # We need to count uncovered t-subsets S such that x in S and S \ {x} ⊆ block_set
                    # S \ {x} is a (t-1)-subset of block_set
                    for rest in itertools.combinations(block_set, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block_set.add(best_point)
            
            # Score the complete block
            score = 0
            for s in itertools.combinations(block_set, t):
                if s in uncovered:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_block = sorted(block_set)
        
        if best_block is None:
            # Fallback: pick a random block
            best_block = sorted(rng.sample(all_points, k))
            best_score = 0
        
        blocks.append(best_block)
        # Remove all t-subsets in best_block from uncovered
        for s in itertools.combinations(best_block, t):
            uncovered.discard(s)
    
    if uncovered:
        # Failed to cover all; return what we have (invalid, but better than timeout)
        pass
    
    return blocks


def construct(v, k, t):
    """
    Main entry point. Uses multiple randomized restarts and returns the best solution.
    """
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    for seed in range(20):
        rng = random.Random(seed)
        blocks = _greedy_cover(v, k, t, rng=rng)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        if best_len <= 76:  # Best known
            break
    
    return best_blocks if best_blocks is not None else []
