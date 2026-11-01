import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets, all k-blocks, and block-to-tset mappings."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
    
    # tset to blocks mapping
    tset_to_blocks = defaultdict(list)
    for b_idx in range(n_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_blocks[ts_idx].append(b_idx)
    
    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks

def _greedy_initial(all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, time_limit):
    """Fast greedy set cover to get a good initial solution."""
    deadline = time.time() + time_limit
    uncovered = np.ones(n_tsets, dtype=bool)
    blocks_chosen = []
    
    while np.any(uncovered) and time.time() < deadline:
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick random uncovered t-set
        rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
        
        # Find best block
        best_block = -1
        best_gain = -1
        for b_idx in tset_to_blocks[rand_idx]:
            gain = 0
            for ts_idx in block_tsets[b_idx]:
                if uncovered[ts_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks_chosen.append(best_block)
        for ts_idx in block_tsets[best_block]:
            uncovered[ts_idx] = False
            
    return blocks_chosen

def _compute_coverage(blocks_idx, n_tsets, block_tsets):
    """Compute coverage count for each t-set."""
    covered = np.zeros(n_tsets, dtype=np.int32)
    for b_idx in blocks_idx:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    return covered

def _try_reduce(blocks_idx, n_tsets, block_tsets, tset_to_blocks, rng, time_limit):
    """Try to reduce the number of blocks using local search."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    covered = _compute_coverage(current, n_tsets, block_tsets)
    best = current[:]
    best_len = len(best)
    
    # Build active block set for fast membership
    active_set = set(current)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            redundant = True
            for ts_idx in block_tsets[b_idx]:
                if covered[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                current.pop(i)
                active_set.discard(b_idx)
                for ts_idx in block_tsets[b_idx]:
                    covered[ts_idx] -= 1
                best = current[:]
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps to reduce size
        for i in range(len(current)):
            if time.time() > deadline:
                break
            old_b = current[i]
            
            # Find t-sets that would become uncovered
            newly_uncovered = []
            for ts_idx in block_tsets[old_b]:
                if covered[ts_idx] == 1:
                    newly_uncovered.append(ts_idx)
            
            if not newly_uncovered:
                # Just remove it
                current.pop(i)
                active_set.discard(old_b)
                for ts_idx in block_tsets[old_b]:
                    covered[ts_idx] -= 1
                best = current[:]
                best_len = len(best)
                improved = True
                break
            
            # Try to find replacement
            best_new_b = -1
            best_gain = -1
            for b_idx in tset_to_blocks[newly_uncovered[0]]:
                if b_idx in active_set:
                    continue
                # Check if it covers all newly_uncovered
                covers_all = True
                for ts_idx in newly_uncovered:
                    if ts_idx not in block_tsets[b_idx]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if covered[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new_b = b_idx
            
            if best_new_b != -1:
                active_set.discard(old_b)
                active_set.add(best_new_b)
                current[i] = best_new_b
                for ts_idx in block_tsets[old_b]:
                    covered[ts_idx] -= 1
                for ts_idx in block_tsets[best_new_b]:
                    covered[ts_idx] += 1
                best = current[:]
                best_len = len(best)
                improved = True
                break
        
        if not improved:
            # Phase 3: Random perturbation
            if len(current) < 2:
                break
            i, j = rng.sample(range(len(current)), 2)
            old_i, old_j = current[i], current[j]
            
            for ts_idx in block_tsets[old_i]:
                covered[ts_idx] -= 1
            for ts_idx in block_tsets[old_j]:
                covered[ts_idx] -= 1
            active_set.discard(old_i)
            active_set.discard(old_j)
            
            new_i, new_j = rng.sample(range(n_blocks), 2)
            current[i] = new_i
            current[j] = new_j
            for ts_idx in block_tsets[new_i]:
                covered[ts_idx] += 1
            for ts_idx in block_tsets[new_j]:
                covered[ts_idx] += 1
            active_set.add(new_i)
            active_set.add(new_j)
            
            if len(current) < best_len:
                best = current[:]
                best_len = len(current)
    
    return best

def _walksat_minimize(n_blocks, n_tsets, block_tsets, tset_to_blocks, initial_blocks, rng, time_limit):
    """WalkSAT-style minimization of block count."""
    deadline = time.time() + time_limit
    
    # Start from initial solution
    state = np.zeros(n_blocks, dtype=bool)
    for b_idx in initial_blocks:
        state[b_idx] = True
    
    covered = np.zeros(n_tsets, dtype=np.int32)
    for b_idx in initial_blocks:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    
    best_state = state.copy()
    best_count = state.sum()
    
    active_set = set(initial_blocks)
    
    while time.time() < deadline:
        # Try to flip a True variable to False (remove a block)
        true_indices = np.where(state)[0]
        if len(true_indices) == 0:
            break
        
        # Try removing blocks in random order
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            rand_idx = true_indices[rng.randint(0, len(true_indices)-1)]
            
            # Check if we can remove this block
            removable = True
            for ts_idx in block_tsets[rand_idx]:
                if covered[ts_idx] == 1:
                    removable = False
                    break
            
            if removable:
                state[rand_idx] = False
                active_set.discard(rand_idx)
                for ts_idx in block_tsets[rand_idx]:
                    covered[ts_idx] -= 1
                new_count = state.sum()
                if new_count < best_count:
                    best_count = new_count
                    best_state = state.copy()
                break
        
        # Try swapping: replace one block with another
        true_indices = np.where(state)[0]
        if len(true_indices) >= 1:
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                old_idx = true_indices[rng.randint(0, len(true_indices)-1)]
                
                # Find t-sets that become uncovered
                newly_uncovered = []
                for ts_idx in block_tsets[old_idx]:
                    if covered[ts_idx] == 1:
                        newly_uncovered.append(ts_idx)
                
                if not newly_uncovered:
                    continue
                
                # Try random replacement
                new_idx = rng.randint(0, n_blocks-1)
                if state[new_idx]:
                    continue
                
                # Check if new block covers all newly_uncovered
                covers_all = True
                for ts_idx in newly_uncovered:
                    if ts_idx not in block_tsets[new_idx]:
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Perform swap
                state[old_idx] = False
                active_set.discard(old_idx)
                for ts_idx in block_tsets[old_idx]:
                    covered[ts_idx] -= 1
                
                state[new_idx] = True
                active_set.add(new_idx)
                for ts_idx in block_tsets[new_idx]:
                    covered[ts_idx] += 1
                
                new_count = state.sum()
                if new_count < best_count:
                    best_count = new_count
                    best_state = state.copy()
                break
        
        # Random perturbation to escape local optima
        if len(true_indices) >= 2:
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                i, j = rng.sample(list(true_indices), 2)
                
                for ts_idx in block_tsets[i]:
                    covered[ts_idx] -= 1
                for ts_idx in block_tsets[j]:
                    covered[ts_idx] -= 1
                state[i] = False
                state[j] = False
                active_set.discard(i)
                active_set.discard(j)
                
                new_i = rng.randint(0, n_blocks-1)
                new_j = rng.randint(0, n_blocks-1)
                while new_j == new_i:
                    new_j = rng.randint(0, n_blocks-1)
                
                state[new_i] = True
                state[new_j] = True
                active_set.add(new_i)
                active_set.add(new_j)
                for ts_idx in block_tsets[new_i]:
                    covered[ts_idx] += 1
                for ts_idx in block_tsets[new_j]:
                    covered[ts_idx] += 1
                
                new_count = state.sum()
                if new_count < best_count:
                    best_count = new_count
                    best_state = state.copy()
    
    return best_state

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin
    
    # Precompute
    all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks = _precompute(v, k, t)
    
    # Multiple greedy runs
    best_blocks = None
    best_size = float('inf')
    
    for seed in range(10):
        if time.time() - start_time > total_budget - 30.0:
            break
        rng = random.Random(seed)
        blocks_idx = _greedy_initial(all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 5.0)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx[:]
    
    if best_blocks is None:
        rng = random.Random(42)
        best_blocks = _greedy_initial(all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 10.0)
        best_size = len(best_blocks)
    
    # Local search reduction
    if time.time() - start_time < total_budget - 5.0:
        rng = random.Random(123)
        remaining = min(60.0, total_budget - (time.time() - start_time) - 5.0)
        improved = _try_reduce(best_blocks, n_tsets, block_tsets, tset_to_blocks, rng, remaining)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # WalkSAT minimization
    if time.time() - start_time < total_budget - 5.0:
        rng = random.Random(456)
        remaining = min(30.0, total_budget - (time.time() - start_time) - 5.0)
        best_state = _walksat_minimize(n_blocks, n_tsets, block_tsets, tset_to_blocks, best_blocks, rng, remaining)
        result_indices = np.where(best_state)[0]
        if len(result_indices) < best_size:
            best_blocks = list(result_indices)
            best_size = len(best_blocks)
    
    # Verify
    covered = _compute_coverage(best_blocks, n_tsets, block_tsets)
    if not np.all(covered > 0):
        # Rebuild
        rng = random.Random(789)
        best_blocks = _greedy_initial(all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 10.0)
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result

# EVOLVE-BLOCK-END
