import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets (indices)
    block_to_subsets = [None] * num_blocks
    # subset_to_blocks[j] is a list of block indices containing subset j
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        subs = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_to_subsets[b_idx] = np.array(subs, dtype=np.int32)
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
            
    def run_greedy(initial_blocks=None, random_seed=0):
        """
        Greedy set cover.
        If initial_blocks is provided, it starts with them and only adds blocks to cover gaps.
        """
        rng = np.random.default_rng(random_seed)
        
        # Determine current coverage
        if initial_blocks:
            covered_mask = np.zeros(num_subsets, dtype=bool)
            for b_idx in initial_blocks:
                covered_mask[block_to_subsets[b_idx]] = True
            uncovered = ~covered_mask
            gains = np.zeros(num_blocks, dtype=np.int32)
            selected_set = set(initial_blocks)
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                gains[b_idx] = np.sum(uncovered[block_to_subsets[b_idx]])
            selected = list(initial_blocks)
        else:
            uncovered = np.ones(num_subsets, dtype=bool)
            gains = np.array([56] * num_blocks, dtype=np.int32)
            selected = []
            selected_set = set()
            
        while True:
            if not np.any(uncovered):
                break
            
            best_idx = int(np.argmax(gains))
            
            if gains[best_idx] <= 0:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            gains[best_idx] = -1
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            
            for s_idx in newly_covered:
                for b_idx in subset_to_blocks[s_idx]:
                    if b_idx in selected_set:
                        continue
                    gains[b_idx] -= 1
                    
        return selected

    def minimize(blocks, random_seed=0, deadline=None):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return blocks
        
        rng = random.Random(random_seed)
        current_blocks = list(blocks)
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            redundant_indices = []
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    redundant_indices.append(i)
                    
            if redundant_indices:
                remove_pos = redundant_indices[rng.randint(0, len(redundant_indices)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                cover_count[block_to_subsets[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    while time.time() < deadline:
        greedy_res = run_greedy(None, trial)
        min_res = minimize(greedy_res, trial, deadline)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        if best_blocks is not None and len(best_blocks) > 5 and time.time() < deadline - 5.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                remaining = [b for b in perturbed if b not in blocks_to_remove]
                repair_res = run_greedy(remaining, trial + 1000)
                min_res2 = minimize(repair_res, trial + 2000, deadline)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy(0, 0)
        best_blocks = minimize(greedy_res, 0, deadline)
        
    return [list(all_blocks[i]) for i in best_blocks]
