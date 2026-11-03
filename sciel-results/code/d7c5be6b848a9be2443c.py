import itertools
import numpy as np
import time
import random
import heapq
import sys

def construct(v, k, t):
    deadline = time.time() + 145
    
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(points, k))
    n_k = len(k_subsets)
    
    subsets_of_block = []
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        sub_indices = []
        for S in itertools.combinations(B, t):
            i = t_index[S]
            sub_indices.append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block.append(sub_indices)
        
    best_solution = None
    best_size = float('inf')
    
    max_restarts = 2000
    lower_bound = 131
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        if best_size <= lower_bound:
            break
            
        # Randomize initial gains to break symmetry
        # Base gain is 35 for all.
        # We add small noise.
        noise = np.random.uniform(0, 1.0, n_k)
        current_gain = np.array([len(subs) + noise[i] for i, subs in enumerate(subsets_of_block)], dtype=np.float64)
        
        heap = [(-g, j) for j, g in enumerate(current_gain)]
        heapq.heapify(heap)
        
        uncovered_mask = np.ones(n_t, dtype=bool)
        num_uncovered = n_t
        
        solution_blocks = []
        
        while num_uncovered > 0:
            if time.time() > deadline - 5:
                break
                
            best_idx = -1
            best_gain = -1
            while heap:
                neg_g, j = heapq.heappop(heap)
                g = -neg_g
                if g == current_gain[j]:
                    best_idx = j
                    best_gain = g
                    break
            
            if best_idx == -1 or best_gain <= 0:
                break
                
            solution_blocks.append(best_idx)
            
            subs_b = subsets_of_block[best_idx]
            newly_covered = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered:
                continue
            
            num_uncovered -= len(newly_covered)
            
            for s in newly_covered:
                uncovered_mask[s] = False
                for b_prime in blocks_of_subset[s]:
                    if b_prime != best_idx:
                        current_gain[b_prime] -= 1.0
                        heapq.heappush(heap, (-current_gain[b_prime], b_prime))
                        
        if solution_blocks:
            solution_blocks = _deletion_pass(solution_blocks, subsets_of_block, n_t)
            
            if len(solution_blocks) < best_size:
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
                
                if best_size <= lower_bound:
                    break

    if best_solution is None:
        # Fallback
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks

    return [list(k_subsets[idx]) for idx in best_solution]

def _deletion_pass(solution_blocks, subsets_of_block, n_t):
    if not solution_blocks:
        return []
        
    n_blocks = len(solution_blocks)
    incidence = np.zeros((n_t, n_blocks), dtype=np.int8)
    for col, b_idx in enumerate(solution_blocks):
        subs_b = subsets_of_block[b_idx]
        incidence[subs_b, col] = 1
        
    cover_count = incidence.sum(axis=1)
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for col in range(len(solution_blocks)):
            b_idx = solution_blocks[col]
            subs_b = subsets_of_block[b_idx]
            if np.any(cover_count[subs_b] == 1):
                new_blocks.append(b_idx)
            else:
                cover_count[subs_b] -= 1
                changed = True
        solution_blocks = new_blocks
        
    return solution_blocks
