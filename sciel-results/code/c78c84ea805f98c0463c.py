import itertools
import random
import time
import sys
import numpy as np

_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]
    
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)
    
    # block_to_subs: for each block, list of indices of t-subsets it covers
    block_to_subs = []
    # sub_to_blocks: for each t-subset, list of indices of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        subs_in_block = []
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_map[t_sub]
            subs_in_block.append(s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        block_to_subs.append(subs_in_block)
        
    _cache[(v, k, t)] = (all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k)
    return _cache[(v, k, t)]

def _greedy_cover(v, k, t, all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k, 
                   seed=0, time_limit=5.0, start_time=None):
    """Fast greedy set cover with random tie-breaking."""
    if start_time is None:
        start_time = time.time()
    
    rng = random.Random(seed)
    
    uncovered = np.ones(n_t, dtype=np.int32)
    gain = np.full(n_k, len(block_to_subs[0]), dtype=np.int32)
    selected = []
    remaining = n_t
    
    while remaining > 0:
        if time.time() - start_time > time_limit:
            return None, None
        
        # Find max gain
        max_gain = gain.max()
        if max_gain == 0:
            break
        
        # Get all blocks with max gain
        candidates = np.where(gain == max_gain)[0]
        if len(candidates) == 0:
            break
        
        # Tie-breaking: prefer blocks that cover subsets with fewer alternatives
        # Compute a score for each candidate
        best_idx = -1
        best_score = -1
        for b in candidates:
            # Score = -sum of (number of blocks covering each subset in this block)
            # Prefer blocks covering "scarce" subsets
            score = 0
            for s in block_to_subs[b]:
                if uncovered[s]:
                    score -= len(sub_to_blocks[s])  # negative: prefer fewer alternatives
            if score > best_score:
                best_score = score
                best_idx = b
        
        if best_idx == -1:
            best_idx = rng.choice(candidates.tolist())
        
        selected.append(all_k_blocks[best_idx])
        
        # Update
        for s in block_to_subs[best_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                remaining -= 1
                for b_prime in sub_to_blocks[s]:
                    gain[b_prime] -= 1
    
    return selected, remaining

def _compress_solution(v, k, t, all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k,
                       solution, seed=0, time_limit=10.0, start_time=None):
    """Iterative compression: try to remove each block and repair."""
    if start_time is None:
        start_time = time.time()
    
    rng = random.Random(seed)
    n_sol = len(solution)
    if n_sol == 0:
        return solution
    
    # Create a set of block indices in the solution for fast lookup
    block_set = set(solution)
    solution_blocks = list(solution)
    
    # For each block in solution, compute which t-subsets it uniquely covers
    # (i.e., no other block in the solution covers them)
    
    # Count how many times each t-subset is covered by the solution
    sub_cover_count = np.zeros(n_t, dtype=np.int32)
    for block in solution_blocks:
        b_idx = all_k_blocks.index(block) if block in all_k_blocks else -1
        # This is slow; let's use a different approach
        pass
    
    # Better approach: track coverage
    # First, build a map from block tuple to index
    block_to_idx = {block: i for i, block in enumerate(all_k_blocks)}
    
    sub_cover_count = np.zeros(n_t, dtype=np.int32)
    for block in solution_blocks:
        b_idx = block_to_idx[block]
        for s in block_to_subs[b_idx]:
            sub_cover_count[s] += 1
    
    # Try removing each block
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Shuffle order for diversity
        order = list(range(len(solution_blocks)))
        rng.shuffle(order)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            block_to_remove = solution_blocks[i]
            b_idx = block_to_idx[block_to_remove]
            
            # Find t-subsets that would become uncovered
            uncovered_after_removal = []
            for s in block_to_subs[b_idx]:
                if sub_cover_count[s] == 1:
                    uncovered_after_removal.append(s)
            
            if not uncovered_after_removal:
                # Safe to remove
                del solution_blocks[i]
                for s in block_to_subs[b_idx]:
                    sub_cover_count[s] -= 1
                improved = True
                break  # Restart loop
            else:
                # Try to repair: find minimum number of blocks to cover uncovered_after_removal
                # Use greedy to find repair blocks
                # But we need to be careful: the repair blocks might overlap with existing solution
                # For now, just try adding one block that covers the most uncovered subsets
                
                # Find block that covers most of uncovered_after_removal
                best_block = None
                best_count = 0
                for b in range(n_k):
                    count = 0
                    for s in block_to_subs[b]:
                        if s in set(uncovered_after_removal):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_block = all_k_blocks[b]
                
                if best_count >= len(uncovered_after_removal):
                    # Perfect repair: one block covers all
                    # Add it
                    new_block = best_block
                    nb_idx = block_to_idx[new_block]
                    solution_blocks.append(new_block)
                    for s in block_to_subs[nb_idx]:
                        sub_cover_count[s] += 1
                    # Now remove the old block
                    del solution_blocks[i]
                    for s in block_to_subs[b_idx]:
                        sub_cover_count[s] -= 1
                    improved = True
                    break
                elif best_count > 0:
                    # Partial repair: add the block, then continue with remaining uncovered
                    # This gets complex; skip for now and try next block
                    pass
    
    return solution_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Run multiple greedy attempts with different seeds
    # Each greedy takes ~0.5-1s, so we can run ~100-200 attempts
    greedy_seed = 0
    while time.time() < deadline - 60.0:  # Reserve 60s for compression
        greedy_seed += 1
        solution, remaining = _greedy_cover(v, k, t, all_k_blocks, block_to_subs, 
                                            sub_to_blocks, n_t, n_k,
                                            seed=greedy_seed, 
                                            time_limit=3.0,
                                            start_time=start_time)
        if solution is not None and remaining == 0:
            if len(solution) < best_count:
                best_count = len(solution)
                best_blocks = solution
                # If we've found a very good solution, we can break early
                if best_count <= 300:
                    break
    
    if best_blocks is None:
        # Fallback: run one more greedy with longer time
        solution, remaining = _greedy_cover(v, k, t, all_k_blocks, block_to_subs, 
                                            sub_to_blocks, n_t, n_k,
                                            seed=1, time_limit=30.0,
                                            start_time=start_time)
        if solution is not None:
            best_blocks = solution
            best_count = len(solution)
    
    if best_blocks is None:
        return []
    
    # Phase 2: Iterative compression
    # Try to improve the best solution
    compress_seed = 0
    current_solution = list(best_blocks)
    current_count = len(current_solution)
    
    while time.time() < deadline - 5.0:
        compress_seed += 1
        compressed = _compress_solution(v, k, t, all_k_blocks, block_to_subs, 
                                        sub_to_blocks, n_t, n_k,
                                        current_solution, seed=compress_seed,
                                        time_limit=20.0,
                                        start_time=start_time)
        if compressed is not None and len(compressed) < current_count:
            current_count = len(compressed)
            current_solution = compressed
            best_count = current_count
            best_blocks = list(current_solution)
        else:
            # Try a different seed
            if time.time() < deadline - 30.0:
                # Occasionally try a fresh greedy + compress
                greedy_seed += 1
                solution, remaining = _greedy_cover(v, k, t, all_k_blocks, block_to_subs, 
                                                    sub_to_blocks, n_t, n_k,
                                                    seed=greedy_seed, 
                                                    time_limit=2.0,
                                                    start_time=start_time)
                if solution is not None and remaining == 0:
                    if len(solution) < current_count:
                        current_count = len(solution)
                        current_solution = list(solution)
                        best_count = current_count
                        best_blocks = list(current_solution)
    
    return best_blocks
