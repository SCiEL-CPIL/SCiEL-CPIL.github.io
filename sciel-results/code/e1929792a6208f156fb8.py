import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using:
    1. Fast greedy set cover with reverse indexing
    2. Local search with 1-for-1 and 2-for-2 block exchanges
    3. Redundant block removal
    Multiple restarts with different seeds to explore solution space.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_ok():
        return time.time() < deadline - 5.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_covered_t = [None] * num_blocks
    # Reverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
        block_covered_t[i] = np.array(block_covered_t[i], dtype=np.int32)
    
    # Convert t_to_blocks to numpy arrays for faster operations
    for i in range(num_t_subsets):
        t_to_blocks[i] = np.array(t_to_blocks[i], dtype=np.int32)
    
    # 2. Greedy Set Cover with reverse indexing
    def initial_greedy():
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        
        # gain[b] = number of currently-uncovered t-subsets covered by block b
        gain = np.zeros(num_blocks, dtype=np.int32)
        
        # Initialize gain: all t-subsets uncovered, so gain = C(k,t) for all blocks
        # But we only care about blocks that cover at least one uncovered t-subset
        # Initially all do, so gain[b] = len(block_covered_t[b]) = C(k,t)
        ckt = len(block_covered_t[0])
        gain[:] = ckt
        
        while np.any(uncovered) and time_ok():
            # Find block with max gain
            best_idx = int(np.argmax(gain))
            if gain[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update: mark t-subsets as covered, update gains
            for t_idx in block_covered_t[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t-subset
                    gain[t_to_blocks[t_idx]] -= 1
            
            # Zero out gain for selected block (it's already "used")
            gain[best_idx] = 0
            
        return selected
    
    # 3. Local search: 1-for-1 and 2-for-2 exchanges + removal
    def local_search(initial_blocks, rng, max_steps=None):
        if not initial_blocks:
            return []
            
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Coverage counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_covered_t[b_idx]] += 1
        
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        steps = 0
        max_steps = max_steps or 1000000
        
        while time_ok() and steps < max_steps:
            steps += 1
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to remove a redundant block
                if len(current_blocks) <= 1:
                    continue
                idx_remove = int(rng.integers(0, len(current_blocks)))
                b_remove = current_blocks[idx_remove]
                
                # Check if any t-subset is only covered by this block
                critical = False
                for t_idx in block_covered_t[b_remove]:
                    if cover_count[t_idx] == 1:
                        critical = True
                        break
                
                if not critical:
                    # Safe to remove
                    cover_count[block_covered_t[b_remove]] -= 1
                    current_blocks.pop(idx_remove)
                    current_set.discard(b_remove)
                    
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
            
            elif move_type < 0.7:
                # 1-for-1 exchange
                if len(current_blocks) == 0:
                    continue
                idx_remove = int(rng.integers(0, len(current_blocks)))
                b_remove = current_blocks[idx_remove]
                
                # Find critical t-subsets (only covered by b_remove)
                critical_list = []
                for t_idx in block_covered_t[b_remove]:
                    if cover_count[t_idx] == 1:
                        critical_list.append(t_idx)
                
                if not critical_list:
                    # Redundant, handled above; skip
                    continue
                
                # Find blocks that cover ALL critical t-subsets
                # Use intersection of t_to_blocks for each critical t-subset
                # Sort by length to intersect smallest first
                critical_list.sort(key=lambda x: len(t_to_blocks[x]))
                
                candidates = set(t_to_blocks[critical_list[0]].tolist())
                for t_idx in critical_list[1:]:
                    candidates &= set(t_to_blocks[t_idx].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Exclude current blocks
                candidates -= current_set
                if not candidates:
                    continue
                
                b_add = int(rng.choice(list(candidates)))
                
                # Perform swap
                cover_count[block_covered_t[b_remove]] -= 1
                cover_count[block_covered_t[b_add]] += 1
                current_blocks[idx_remove] = b_add
                current_set.discard(b_remove)
                current_set.add(b_add)
                
            else:
                # 2-for-2 exchange
                if len(current_blocks) < 2:
                    continue
                idx1 = int(rng.integers(0, len(current_blocks)))
                idx2 = int(rng.integers(0, len(current_blocks)))
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(current_blocks)
                
                b1 = current_blocks[idx1]
                b2 = current_blocks[idx2]
                
                # Find critical t-subsets for both blocks
                # A t-subset is critical if it's covered by b1 or b2 AND no other block covers it
                # i.e., cover_count[t] == 1 and t in b1, OR cover_count[t] == 2 and t in both b1 and b2
                critical_list = []
                
                # For t in b1: critical if cover_count[t] == 1
                for t_idx in block_covered_t[b1]:
                    if cover_count[t_idx] == 1:
                        critical_list.append(t_idx)
                
                # For t in b2: critical if cover_count[t] == 1, OR cover_count[t] == 2 and t also in b1
                # If t in both b1 and b2 and cover_count[t] == 2, removing both makes it 0
                # If t in b2 only and cover_count[t] == 1, removing b2 makes it 0
                b1_set = set(block_covered_t[b1].tolist())
                for t_idx in block_covered_t[b2]:
                    if cover_count[t_idx] == 1:
                        if t_idx not in b1_set:  # not already added
                            critical_list.append(t_idx)
                    elif cover_count[t_idx] == 2 and t_idx in b1_set:
                        # Covered by both, removing both makes it 0
                        if t_idx not in [x for x in critical_list]:
                            critical_list.append(t_idx)
                
                if not critical_list:
                    # Both blocks are redundant? Remove one
                    cover_count[block_covered_t[b1]] -= 1
                    current_blocks.pop(idx1)
                    current_set.discard(b1)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue
                
                # Find blocks that cover ALL critical t-subsets
                critical_list.sort(key=lambda x: len(t_to_blocks[x]))
                
                candidates = set(t_to_blocks[critical_list[0]].tolist())
                for t_idx in critical_list[1:]:
                    candidates &= set(t_to_blocks[t_idx].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                candidates -= current_set
                if not candidates:
                    continue
                
                # Pick 2 distinct blocks from candidates
                cand_list = list(candidates)
                if len(cand_list) < 2:
                    continue
                
                idx_a = int(rng.integers(0, len(cand_list)))
                idx_b = int(rng.integers(0, len(cand_list)))
                if idx_a == idx_b:
                    idx_b = (idx_b + 1) % len(cand_list)
                
                b_add1 = cand_list[idx_a]
                b_add2 = cand_list[idx_b]
                
                # Perform 2-for-2 swap
                cover_count[block_covered_t[b1]] -= 1
                cover_count[block_covered_t[b2]] -= 1
                cover_count[block_covered_t[b_add1]] += 1
                cover_count[block_covered_t[b_add2]] += 1
                
                current_blocks[idx1] = b_add1
                current_blocks[idx2] = b_add2
                current_set.discard(b1)
                current_set.discard(b2)
                current_set.add(b_add1)
                current_set.add(b_add2)
        
        # Final cleanup: remove redundant blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_covered_t[b_idx]] += 1
        
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    cover_count[block_covered_t[b_idx]] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break
        
        # Update best
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
        
        return best_blocks
    
    # 4. Main execution with multiple restarts
    best_result = None
    best_len = float('inf')
    
    # First, get a good initial solution with greedy
    init_blocks = initial_greedy()
    
    if not init_blocks:
        return []
    
    # Run local search on the initial solution
    rng = np.random.default_rng(42)
    result = local_search(init_blocks, rng)
    
    if result and len(result) < best_len:
        best_len = len(result)
        best_result = result
    
    # Try additional restarts with different initial solutions
    restart_count = 0
    while time_ok() and restart_count < 5:
        restart_count += 1
        # Generate a different initial solution with randomization
        rng_init = np.random.default_rng(1000 + restart_count)
        
        # Randomized greedy: at each step, pick from top-k blocks by gain
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        gain = np.full(num_blocks, len(block_covered_t[0]), dtype=np.int32)
        
        while np.any(uncovered) and time_ok():
            # Get top-k blocks
            k = min(10, num_blocks)
            top_k = np.argpartition(gain, -k)[-k:]
            best_idx = int(rng_init.choice(top_k))
            if gain[best_idx] == 0:
                break
            
            selected.append(best_idx)
            
            for t_idx in block_covered_t[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    gain[t_to_blocks[t_idx]] -= 1
            gain[best_idx] = 0
        
        if not selected:
            continue
        
        rng_ls = np.random.default_rng(2000 + restart_count)
        result = local_search(selected, rng_ls)
        
        if result and len(result) < best_len:
            best_len = len(result)
            best_result = result
    
    if best_result is None:
        best_result = result if result else []
    
    # Convert to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in best_result]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for b in result:
        b.sort()
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
    
    return unique_result

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
