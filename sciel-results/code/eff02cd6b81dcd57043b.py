"""
Strategy: Fast Greedy Set Cover with Bitmask Optimization + Local Search for C(19,8,5)

The key insight for speed is to use bitmasks to represent t-subsets within blocks.
For v=19, k=8, t=5:
- Universe size = C(19,5) = 11628
- Block universe = C(19,8) = 75582
- Each block covers C(8,5) = 56 t-subsets

Approach:
1. Precompute all 8-blocks and their covered 5-subset indices using bitmasks
2. Greedy: pick block covering most uncovered subsets
3. Local search: try replacing one block with another to reduce total count while maintaining coverage
4. Multiple restarts with randomization
"""

import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)  # 11628
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)  # 75582
    
    # For each block, compute which t-subsets it covers (as indices)
    # Also precompute the bitmask of covered t-subsets for each block
    # Since num_t = 11628, we can use Python ints as bitmasks
    block_covers = []  # list of frozensets of t-subset indices
    for block in all_k_blocks:
        covers = set()
        for sub in itertools.combinations(block, t):
            covers.add(t_sub_to_idx[sub])
        block_covers.append(frozenset(covers))
    
    # Also precompute for each t-subset, which blocks cover it (for local search)
    t_sub_to_blocks = [set() for _ in range(num_t)]
    for i, covers in enumerate(block_covers):
        for s_idx in covers:
            t_sub_to_blocks[s_idx].add(i)
    
    def greedy_solve(seed_val):
        random.seed(seed_val)
        # Start with all t-subsets uncovered
        uncovered_count = [1] * num_t  # how many times each t-subset needs to be covered
        covered_count = [0] * num_t
        selected_blocks = set()
        
        while any(c > 0 for c in uncovered_count):
            # Find the block that covers the most uncovered t-subsets
            best_gain = -1
            best_block_idx = -1
            best_candidates = []
            
            for b_idx in range(num_blocks):
                if b_idx in selected_blocks:
                    continue
                gain = sum(1 for s in block_covers[b_idx] if covered_count[s] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if best_gain <= 0:
                # Shouldn't happen if we have valid blocks
                break
            
            # Pick randomly among best candidates for diversity
            chosen = random.choice(best_candidates)
            selected_blocks.add(chosen)
            
            # Update covered counts
            for s in block_covers[chosen]:
                covered_count[s] += 1
        
        return list(selected_blocks)
    
    def local_search(block_indices, seed_val, time_limit=2.0):
        """Try to reduce the number of blocks via local search (swap-based)"""
        random.seed(seed_val)
        local_deadline = time.time() + time_limit
        current = set(block_indices)
        
        # Compute coverage counts
        covered_count = [0] * num_t
        for b in current:
            for s in block_covers[b]:
                covered_count[s] += 1
        
        best_solution = set(current)
        best_size = len(current)
        
        T = 5.0  # Temperature
        T_min = 0.1
        alpha = 0.95
        
        while time.time() < local_deadline and len(current) > best_size - 1:
            # Try to remove a block and add a better one
            # Randomly pick a block to consider removing
            if not current:
                break
            remove_b = random.choice(list(current))
            
            # Find a block that covers all the unique t-subsets that only remove_b covers
            # Compute which t-subsets are ONLY covered by remove_b
            unique_to_remove = []
            for s in block_covers[remove_b]:
                # Count how many other selected blocks cover s
                cnt = 0
                for b in current:
                    if b != remove_b and s in block_covers[b]:
                        cnt += 1
                if cnt == 0:
                    unique_to_remove.append(s)
            
            if not unique_to_remove:
                # This block is redundant, remove it
                current.discard(remove_b)
                for s in block_covers[remove_b]:
                    covered_count[s] -= 1
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = set(current)
                continue
            
            # Need to find a block that covers all of unique_to_remove
            # This is expensive, so limit candidates
            # A block that covers all unique_to_remove must contain the union of these subsets
            required_points = set()
            for s in unique_to_remove:
                required_points.update(all_t_subsets[s])
            
            if len(required_points) > k:
                continue  # No single block can cover all
            
            # Find candidate blocks that contain required_points
            candidates = []
            for b_idx in range(num_blocks):
                if b_idx in current:
                    continue
                # Check if b covers all required points
                block_set = set(all_k_blocks[b_idx])
                if required_points.issubset(block_set):
                    candidates.append(b_idx)
            
            if not candidates:
                continue
            
            # Pick a random candidate
            add_b = random.choice(candidates)
            
            # Calculate delta: how many new t-subsets are covered by add_b that aren't already covered
            delta = 0
            for s in block_covers[add_b]:
                if covered_count[s] == 0:
                    delta += 1
            
            # Accept if it improves or with some probability
            if delta > 0:
                current.discard(remove_b)
                current.add(add_b)
                for s in block_covers[remove_b]:
                    covered_count[s] -= 1
                for s in block_covers[add_b]:
                    covered_count[s] += 1
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = set(current)
            elif delta == 0 and random.random() < 0.3:
                # Neutral move for exploration
                current.discard(remove_b)
                current.add(add_b)
                for s in block_covers[remove_b]:
                    covered_count[s] -= 1
                for s in block_covers[add_b]:
                    covered_count[s] += 1
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Verify coverage of best_solution
        if best_solution:
            final_covered = [0] * num_t
            for b in best_solution:
                for s in block_covers[b]:
                    final_covered[s] += 1
            if all(c > 0 for c in final_covered):
                return list(best_solution)
        
        return list(current)
    
    best_blocks = None
    best_len = float('inf')
    
    # First, get a greedy solution
    initial_run_start = time.time()
    initial_blocks = greedy_solve(42)
    if initial_blocks:
        best_blocks = initial_blocks
        best_len = len(initial_blocks)
    
    # Try local search on the greedy solution
    if time.time() < deadline - 10.0:
        ls_start = time.time()
        minimized = local_search(best_blocks, 42, time_limit=min(5.0, deadline - time.time() - 5.0))
        if minimized and len(minimized) < best_len:
            # Verify
            final_covered = [0] * num_t
            for b in minimized:
                for s in block_covers[b]:
                    final_covered[s] += 1
            if all(c > 0 for c in final_covered):
                best_blocks = minimized
                best_len = len(minimized)
    
    # Multiple randomized greedy restarts with local search
    run_id = 0
    while time.time() < deadline - 15.0:
        seed_val = run_id * 1234 + 7
        greedy_blocks = greedy_solve(seed_val)
        
        if greedy_blocks:
            cur_len = len(greedy_blocks)
            if cur_len < best_len:
                best_blocks = greedy_blocks
                best_len = cur_len
            
            # Try local search if we have time
            if time.time() < deadline - 10.0 and cur_len <= best_len:
                ls_time = min(3.0, deadline - time.time() - 5.0)
                if ls_time > 1.0:
                    minimized = local_search(greedy_blocks, seed_val + 1000, time_limit=ls_time)
                    if minimized and len(minimized) < best_len:
                        final_covered = [0] * num_t
                        for b in minimized:
                            for s in block_covers[b]:
                                final_covered[s] += 1
                        if all(c > 0 for c in final_covered):
                            best_blocks = minimized
                            best_len = len(minimized)
        
        run_id += 1
    
    if best_blocks is None:
        # Fallback
        return greedy_solve(42)
    
    # Convert block indices back to actual blocks
    result = [list(all_k_blocks[i]) for i in best_blocks]
    # Sort each block for consistency
    for b in result:
        b.sort()
    return result
