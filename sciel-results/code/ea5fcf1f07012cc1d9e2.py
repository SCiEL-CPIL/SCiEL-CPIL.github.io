import itertools
import random
import time

# Precompute for the specific task parameters v=19, k=8, t=5
# But the function signature is construct(v, k, t), so we must handle generic v,k,t?
# The task says "The evaluator passes this task's parameters v, k and t."
# It implies v=19, k=8, t=5.
# However, to be safe and generic, we should check if v,k,t are small enough for precomputation.
# C(19,8) = 75582 blocks. C(19,5) = 11628 t-subsets.
# If v is larger, this might not work. But the task is specific to C(19,8,5).
# I will assume the parameters are fixed to the task description or small enough.
# The prompt says "construct(v, k, t) -> list of blocks; the evaluator passes this task's parameters".
# I will write code that works for v=19, k=8, t=5 efficiently, and falls back to a simpler method if parameters differ, 
# though the score is only for this specific task.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # If parameters are not the specific one, use a simple fallback
    if v != 19 or k != 8 or t != 5:
        # Fallback: Standard Greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick first uncovered
            target = min(uncovered)
            # Find best block containing target
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    # Specific optimization for v=19, k=8, t=5
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    # Precompute coverage for all k-subsets
    # block_to_covered: dict mapping block tuple to list of t-subset indices
    block_to_covered = {}
    all_blocks = list(itertools.combinations(points, k))
    for block in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(block, t):
            covered_indices.append(t_subset_to_idx[ts])
        block_to_covered[block] = covered_indices

    best_blocks = None
    best_len = float('inf')
    
    # Randomized Greedy
    # We will run multiple restarts.
    # In each restart, we maintain a set of uncovered t-subset indices.
    
    while time.time() < deadline:
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to anchor the next block
            # Using random choice ensures diversity
            u_idx = random.choice(list(uncovered))
            u_set = all_t_subsets[u_idx]
            
            # The next block must contain u_set.
            # Block = u_set + R, where R is a (k-t) subset of points not in u_set.
            points_in_u = set(u_set)
            available_points = [p for p in points if p not in points_in_u]
            need = k - t
            
            best_block = None
            best_gain = -1
            # To add randomness to tie-breaking, we can shuffle candidates or add noise
            # Here we just pick the max gain. To diversify, the random choice of u_idx is the main driver.
            
            for R in itertools.combinations(available_points, need):
                block = tuple(sorted(u_set + R))
                covered = block_to_covered[block]
                # Count how many of these are currently uncovered
                gain = 0
                for idx in covered:
                    if idx in uncovered:
                        gain += 1
                        # Early exit if gain exceeds best? 
                        # If gain > best_gain, we can't stop counting yet unless we know max possible.
                        # Max possible is len(covered).
                        if gain > best_gain:
                            # We can't break early easily without knowing if a later one is better,
                            # but we can update best.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break # Should not happen
            
            blocks.append(list(best_block))
            
            # Update uncovered
            covered_indices = block_to_covered[best_block]
            for idx in covered_indices:
                if idx in uncovered:
                    uncovered.remove(idx)
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # If we hit the lower bound, we can stop
        # Lower bound is ceil(n_t / C(k,t)) = ceil(11628 / 56) = 208
        if best_len <= 208:
            break

    if best_blocks is None:
        # Fallback if no valid solution found (should not happen)
        return []
        
    return best_blocks
