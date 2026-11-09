import itertools
import time
import random
import sys
import math
from collections import defaultdict

def construct(v, k, t):
    """
    Solve C(v,k,t) using a Max-SAT-style stochastic local search (WalkSAT)
    where each block is a Boolean variable. Objective: minimize uncovered triples
    + lambda * number of active blocks. Neighborhood: flip 1 or 2 block variables.
    Final fill ensures validity.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]

    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices

    # Precompute block intersection sizes for faster gain computation
    # For each pair of blocks, how many t-subsets do they share?
    # This is expensive for C(24,7) = 346104 blocks, so we'll compute on the fly
    # but cache frequently used ones.

    def run_greedy(seed=None):
        """Greedy set cover to get an initial solution."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        uncovered_set = set(range(num_t_subsets))
        selected = []
        
        while uncovered_set:
            if time.time() > deadline:
                break
            
            target_t_idx = rng.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]
            
            best_block = -1
            best_gain = -1
            best_blocks_with_gain = []
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_covered_t[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_blocks_with_gain = [b_idx]
                elif gain == best_gain:
                    best_blocks_with_gain.append(b_idx)
            
            if best_block == -1:
                break
            
            best_block = rng.choice(best_blocks_with_gain)
            selected.append(best_block)
            
            for t_idx in block_covered_t[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.discard(t_idx)
        
        return selected

    def try_remove_blocks(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
        
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks

    def walksat_search(initial_blocks, time_budget=15.0, seed=None):
        """
        WalkSAT-style local search on the Max-SAT formulation.
        Variables: x[i] = 1 if block i is selected.
        Clauses: for each t-subset j, clause is (OR of x[i] for i in t_to_blocks[j]).
        Objective: minimize sum(unsatisfied_clauses) + lambda * sum(x[i])
        
        Neighborhood: flip one variable, or flip two variables (swap).
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        # Start from initial solution
        x = [0] * num_blocks
        for b_idx in initial_blocks:
            x[b_idx] = 1
        
        # Compute current coverage
        cover_count = [0] * num_t_subsets
        num_active = 0
        for b_idx in range(num_blocks):
            if x[b_idx]:
                num_active += 1
                for t_idx in block_covered_t[b_idx]:
                    cover_count[t_idx] += 1
        
        uncovered = sum(1 for c in cover_count if c == 0)
        
        # Lambda: weight for block count penalty
        # We want to encourage fewer blocks but prioritize coverage
        # Start with lambda that makes 1 uncovered triple cost more than 1 extra block
        # but not so much that we can't reduce blocks
        lam = 0.5  # Tuned parameter
        
        def compute_objective(cover_count, num_active):
            unc = sum(1 for c in cover_count if c == 0)
            return unc + lam * num_active
        
        current_obj = compute_objective(cover_count, num_active)
        best_obj = current_obj
        best_x = x[:]
        best_cover = cover_count[:]
        best_num_active = num_active
        
        # Simulated annealing parameters
        temp = 2.0
        cooling = 0.995
        min_temp = 0.01
        
        elapsed = 0.0
        t0 = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while elapsed < time_budget and temp > min_temp:
            if time.time() > deadline - 3.0:
                break
            
            # Choose neighborhood: 1-flip or 2-flip
            if rng.random() < 0.4:
                # 1-flip: add or remove one block
                # Pick a block to flip
                # Bias towards blocks that cover uncovered triples or are redundant
                flip_candidates = []
                
                # Try to add a block that covers uncovered triples
                if uncovered > 0:
                    # Pick a random uncovered triple
                    unc_triples = [i for i, c in enumerate(cover_count) if c == 0]
                    target = rng.choice(unc_triples)
                    for b_idx in t_to_blocks[target]:
                        if not x[b_idx]:
                            flip_candidates.append((b_idx, 1))  # add
                    # Also consider removing a redundant block
                    for b_idx in range(num_blocks):
                        if x[b_idx]:
                            redundant = True
                            for t_idx in block_covered_t[b_idx]:
                                if cover_count[t_idx] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                flip_candidates.append((b_idx, -1))  # remove
                else:
                    # All covered, try to remove redundant blocks
                    for b_idx in range(num_blocks):
                        if x[b_idx]:
                            redundant = True
                            for t_idx in block_covered_t[b_idx]:
                                if cover_count[t_idx] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                flip_candidates.append((b_idx, -1))
                    # Also try random swaps
                    active = [i for i in range(num_blocks) if x[i]]
                    if active:
                        b_idx = rng.choice(active)
                        flip_candidates.append((b_idx, -1))
                
                if not flip_candidates:
                    # Random flip
                    b_idx = rng.randint(0, num_blocks - 1)
                    if x[b_idx]:
                        flip_candidates.append((b_idx, -1))
                    else:
                        flip_candidates.append((b_idx, 1))
                
                b_idx, direction = rng.choice(flip_candidates)
                
                # Compute delta objective
                if direction == 1:
                    # Adding block b_idx
                    gain = 0
                    for t_idx in block_covered_t[b_idx]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    delta_obj = -gain + lam
                else:
                    # Removing block b_idx
                    loss = 0
                    for t_idx in block_covered_t[b_idx]:
                        if cover_count[t_idx] == 1:
                            loss += 1
                    delta_obj = loss - lam
                
                # Accept with probability
                if delta_obj <= 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta_obj / temp)
                
                if accept:
                    # Apply flip
                    if direction == 1:
                        x[b_idx] = 1
                        num_active += 1
                        for t_idx in block_covered_t[b_idx]:
                            cover_count[t_idx] += 1
                            if cover_count[t_idx] == 1:
                                uncovered -= 1
                    else:
                        x[b_idx] = 0
                        num_active -= 1
                        for t_idx in block_covered_t[b_idx]:
                            cover_count[t_idx] -= 1
                            if cover_count[t_idx] == 0:
                                uncovered += 1
                    
                    current_obj = compute_objective(cover_count, num_active)
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_x = x[:]
                        best_cover = cover_count[:]
                        best_num_active = num_active
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count > max_no_improve:
                        # Restart from best
                        x = best_x[:]
                        cover_count = best_cover[:]
                        num_active = best_num_active
                        uncovered = sum(1 for c in cover_count if c == 0)
                        current_obj = best_obj
                        no_improve_count = 0
                        # Reheat
                        temp = min(2.0, temp * 2.0)
                
            else:
                # 2-flip: swap one block for another
                active = [i for i in range(num_blocks) if x[i]]
                if len(active) < 2:
                    temp *= cooling
                    continue
                
                b_remove = rng.choice(active)
                
                # Find candidate blocks to add that cover the essential triples of b_remove
                essential = []
                for t_idx in block_covered_t[b_remove]:
                    if cover_count[t_idx] == 1:
                        essential.append(t_idx)
                
                if not essential:
                    # b_remove is redundant, just remove it
                    x[b_remove] = 0
                    num_active -= 1
                    for t_idx in block_covered_t[b_remove]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            uncovered += 1
                    current_obj = compute_objective(cover_count, num_active)
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_x = x[:]
                        best_cover = cover_count[:]
                        best_num_active = num_active
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        x = best_x[:]
                        cover_count = best_cover[:]
                        num_active = best_num_active
                        uncovered = sum(1 for c in cover_count if c == 0)
                        current_obj = best_obj
                        no_improve_count = 0
                        temp = min(2.0, temp * 2.0)
                    temp *= cooling
                    continue
                
                # Sample candidate blocks to add
                # Prefer blocks that cover many essential triples
                best_add = -1
                best_delta = float('inf')
                
                # Get candidates from essential triples
                candidate_set = set()
                for t_idx in essential:
                    for b_idx in t_to_blocks[t_idx]:
                        if b_idx != b_remove:
                            candidate_set.add(b_idx)
                
                if not candidate_set:
                    temp *= cooling
                    continue
                
                # Sample up to 200 candidates
                candidates = list(candidate_set)
                if len(candidates) > 200:
                    candidates = rng.sample(candidates, 200)
                
                for b_add in candidates:
                    if x[b_add]:
                        continue
                    
                    # Check if b_add covers all essential triples
                    covers_all = True
                    for t_idx in essential:
                        if t_idx not in block_covered_t[b_add]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Compute delta: remove b_remove, add b_add
                    # Loss from removing b_remove: triples that become uncovered
                    # Gain from adding b_add: triples that become covered
                    loss = 0
                    for t_idx in block_covered_t[b_remove]:
                        if cover_count[t_idx] == 1:
                            # This triple would be uncovered, but check if b_add covers it
                            if t_idx in block_covered_t[b_add]:
                                pass  # Still covered by b_add
                            else:
                                loss += 1
                    
                    gain = 0
                    for t_idx in block_covered_t[b_add]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                        elif t_idx in block_covered_t[b_remove] and cover_count[t_idx] == 1:
                            # Already counted in loss check
                            pass
                    
                    # Net change in uncovered: loss - gain
                    # Change in num_active: 0 (swap)
                    delta_obj = loss - gain
                    
                    if delta_obj < best_delta:
                        best_delta = delta_obj
                        best_add = b_add
                
                if best_add == -1:
                    temp *= cooling
                    continue
                
                # Accept with probability
                if best_delta <= 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-best_delta / temp)
                
                if accept:
                    # Perform swap
                    x[b_remove] = 0
                    for t_idx in block_covered_t[b_remove]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            uncovered += 1
                    
                    x[best_add] = 1
                    for t_idx in block_covered_t[best_add]:
                        cover_count[t_idx] += 1
                        if cover_count[t_idx] == 1:
                            uncovered -= 1
                    
                    # num_active unchanged
                    
                    current_obj = compute_objective(cover_count, num_active)
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_x = x[:]
                        best_cover = cover_count[:]
                        best_num_active = num_active
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count > max_no_improve:
                        x = best_x[:]
                        cover_count = best_cover[:]
                        num_active = best_num_active
                        uncovered = sum(1 for c in cover_count if c == 0)
                        current_obj = best_obj
                        no_improve_count = 0
                        temp = min(2.0, temp * 2.0)
                
                temp *= cooling
            
            elapsed = time.time() - t0
        
        # Extract blocks from best solution
        result = [i for i in range(num_blocks) if best_x[i]]
        
        # Ensure validity: fill in any uncovered triples
        cover_count = [0] * num_t_subsets
        for b_idx in result:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        uncovered_triples = [i for i, c in enumerate(cover_count) if c == 0]
        for t_idx in uncovered_triples:
            # Find a block covering this triple
            for b_idx in t_to_blocks[t_idx]:
                if b_idx not in result:
                    result.append(b_idx)
                    for t2 in block_covered_t[b_idx]:
                        cover_count[t2] += 1
                    break
        
        return result

    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
        
        processed_res = try_remove_blocks(greedy_res)
        
        # Run WalkSAT search
        ws_res = walksat_search(processed_res, time_budget=12.0, seed=seed)
        
        if len(ws_res) < best_len:
            best_len = len(ws_res)
            best_blocks = ws_res
        
        if best_len <= 69:
            break
    
    # If we have time, try more random seeds
    if time.time() < deadline - 15.0:
        while time.time() < deadline - 5.0:
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            ws_res = walksat_search(processed_res, time_budget=10.0, seed=seed)
            
            if len(ws_res) < best_len:
                best_len = len(ws_res)
                best_blocks = ws_res
            
            if best_len <= 69:
                break
    
    if best_blocks is None:
        return []
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    for b in final_blocks:
        b.sort()
    
    return final_blocks
