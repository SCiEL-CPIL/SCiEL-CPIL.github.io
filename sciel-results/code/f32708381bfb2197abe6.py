import itertools
import time
import random
import sys
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # For t=3, precompute: for each pair (a,b), the list of 3-subset indices {a,b,z} for z != a,b
    # Also precompute for each point x, which 3-subsets contain x
    if t == 3:
        pair_to_subsets = defaultdict(list)
        point_to_subsets = defaultdict(list)
        for idx, sub in enumerate(all_subsets):
            point_to_subsets[sub[0]].append(idx)
            point_to_subsets[sub[1]].append(idx)
            point_to_subsets[sub[2]].append(idx)
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = sub[i], sub[j]
                    pair = (a, b) if a < b else (b, a)
                    pair_to_subsets[pair].append(idx)
    else:
        pair_to_subsets = None
        point_to_subsets = None
    
    # Precompute for each subset index, the tuple of elements (for fast membership)
    subset_elements = all_subsets  # already tuples
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_rng):
        """Greedy construction with fast gain calculation using NumPy."""
        blocks = []
        uncovered = np.ones(n_subsets, dtype=np.int8)
        num_uncovered = n_subsets
        
        if t == 3:
            # For each pair, precompute the subset indices as a numpy array
            pair_to_subsets_arr = {}
            for pair, lst in pair_to_subsets.items():
                pair_to_subsets_arr[pair] = np.array(lst, dtype=np.int32)
            
            # Precompute for each point x: boolean mask over all subsets where x appears
            # shape: (v, n_subsets), each row is bool mask
            point_mask = np.zeros((v, n_subsets), dtype=np.int8)
            for x in range(v):
                for idx in point_to_subsets[x]:
                    point_mask[x, idx] = 1
            
            def get_gain_fast(block, candidate_x):
                """Calculate how many new subsets are covered by adding x to block."""
                if len(block) < 2:
                    return 0
                gain = 0
                pmask = point_mask[candidate_x]
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        a, b = block[i], block[j]
                        pair = (a, b) if a < b else (b, a)
                        arr = pair_to_subsets_arr.get(pair)
                        if arr is not None and len(arr) > 0:
                            gain += int(np.sum(uncovered[arr] & pmask[arr]))
                return gain
        else:
            def get_gain_fast(block, candidate_x):
                if len(block) < 2:
                    return 0
                gain = 0
                for combo in itertools.combinations(block, t-1):
                    combo_set = set(combo)
                    for idx, sub in enumerate(all_subsets):
                        if x in sub and combo_set.issubset(sub):
                            if uncovered[idx]:
                                gain += 1
                            break
                return gain
        
        while num_uncovered > 0:
            if time.time() > deadline:
                return None
            
            # Find a random uncovered subset
            seed_idx = -1
            for _ in range(50):
                candidate = seed_rng.randint(0, n_subsets - 1)
                if uncovered[candidate]:
                    seed_idx = candidate
                    break
            if seed_idx == -1:
                for i in range(n_subsets):
                    if uncovered[i]:
                        seed_idx = i
                        break
            if seed_idx == -1:
                break
            
            seed_sub = all_subsets[seed_idx]
            block = list(seed_sub)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1
                tied_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = get_gain_fast(block, x)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tied_points = [x]
                    elif gain == best_gain and gain > 0:
                        tied_points.append(x)
                
                if best_point is None or best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                elif len(tied_points) > 1:
                    best_point = seed_rng.choice(tied_points)
                
                block.append(best_point)
            
            # Fill to size k if needed
            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if uncovered[idx]:
                    uncovered[idx] = 0
                    num_uncovered -= 1
                
        return blocks
    
    def compute_coverage(blocks_list):
        cov_count = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks_list:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                cov_count[idx] += 1
        return cov_count
    
    def is_valid_covering(blocks_list):
        cov = compute_coverage(blocks_list)
        return bool(np.all(cov > 0))
    
    def local_search(blocks):
        """Try to improve by removing redundant blocks and swapping."""
        blocks = list(blocks)
        
        # Remove redundant blocks iteratively - compute coverage once per pass
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            cov_count = compute_coverage(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                block_i = blocks[i]
                redundant = True
                for sub in itertools.combinations(block_i, t):
                    idx = subset_to_idx[sub]
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try block replacement: replace a block with a better one
        max_replacement_attempts = 50
        for _ in range(max_replacement_attempts):
            if time.time() > deadline - 15:
                break
            if len(blocks) == 0:
                break
            
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Compute coverage once for this iteration
            cov_count = compute_coverage(blocks)
            unique_subs = []
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                if cov_count[idx] == 1:
                    unique_subs.append(sub)
            
            if not unique_subs:
                blocks.pop(i)
                continue
            
            # Build new block from union of unique subsets
            candidates = set()
            for sub in unique_subs:
                candidates.update(sub)
            
            if len(candidates) > k:
                continue
            
            # Try multiple random completions
            best_new_block = None
            best_new_unique = 0
            for _ in range(3):
                cand = set(candidates)
                while len(cand) < k:
                    x = random.randint(0, v - 1)
                    if x not in cand:
                        cand.add(x)
                
                new_block = tuple(sorted(cand))
                
                # Verify it covers all unique_subs
                valid = True
                for sub in unique_subs:
                    if not all(x in new_block for x in sub):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count how many new unique subsets this block would cover
                # Instead of recomputing full coverage, just check which subsets
                # in new_block are not covered by other blocks
                cov_without_i = cov_count.copy()
                for sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[sub]
                    cov_without_i[idx] -= 1
                
                new_unique = 0
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    if cov_without_i[idx] == 0:
                        new_unique += 1
                
                if new_unique > best_new_unique:
                    best_new_unique = new_unique
                    best_new_block = new_block
            
            if best_new_block is not None and best_new_unique > len(unique_subs):
                blocks[i] = list(best_new_block)
                blocks[i].sort()
        
        return blocks
    
    def simulated_annealing(blocks, seed_rng):
        """SA to escape local minima by randomly swapping blocks."""
        if time.time() > deadline - 20:
            return blocks
        
        blocks = list(blocks)
        n = len(blocks)
        if n < 2:
            return blocks
        
        # Initial temperature
        T_start = 5.0
        T_end = 0.1
        n_iter = 2000
        cooling = (T_end / T_start) ** (1.0 / max(n_iter - 1, 1))
        
        current_blocks = list(blocks)
        best_blocks_sa = list(blocks)
        best_len = len(blocks)
        
        # Precompute coverage for current state
        current_cov = compute_coverage(current_blocks)
        
        for it in range(n_iter):
            if time.time() > deadline - 20:
                break
            
            T = T_start * (cooling ** it)
            
            # Choose a random block to modify
            i = seed_rng.randint(0, n - 1)
            old_block = current_blocks[i]
            
            # Generate a new random block that covers the same unique subsets
            # Find subsets uniquely covered by old_block
            unique_subs = []
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                if current_cov[idx] == 1:
                    unique_subs.append(sub)
            
            if unique_subs:
                candidates = set()
                for sub in unique_subs:
                    candidates.update(sub)
                
                if len(candidates) > k:
                    continue
                
                # Random completion
                new_block_set = set(candidates)
                while len(new_block_set) < k:
                    x = seed_rng.randint(0, v - 1)
                    if x not in new_block_set:
                        new_block_set.add(x)
            else:
                # No unique subsets - can try a completely random block
                new_block_set = set(seed_rng.sample(range(v), k))
            
            new_block = tuple(sorted(new_block_set))
            
            # Compute cost change: how many subsets become uncovered?
            lost = 0
            gained = 0
            new_block_set_final = set(new_block)
            
            for sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[sub]
                if current_cov[idx] == 1:
                    if not all(x in new_block_set_final for x in sub):
                        lost += 1
            
            for sub in itertools.combinations(new_block, t):
                idx = subset_to_idx[sub]
                if current_cov[idx] == 0:
                    gained += 1
            
            delta_cost = lost - gained  # positive means worse
            
            if delta_cost <= 0:
                current_blocks[i] = list(new_block)
                for sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[sub]
                    current_cov[idx] -= 1
                for sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[sub]
                    current_cov[idx] += 1
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks_sa = list(current_blocks)
            else:
                if seed_rng.random() < math.exp(-delta_cost / max(T, 1e-10)):
                    current_blocks[i] = list(new_block)
                    for sub in itertools.combinations(old_block, t):
                        idx = subset_to_idx[sub]
                        current_cov[idx] -= 1
                    for sub in itertools.combinations(new_block, t):
                        idx = subset_to_idx[sub]
                        current_cov[idx] += 1
            
            # Occasionally try removing a block if it's redundant
            if it % 15 == 0 and len(current_blocks) > 1:
                j = seed_rng.randint(0, len(current_blocks) - 1)
                test_block = current_blocks[j]
                redundant = True
                for sub in itertools.combinations(test_block, t):
                    idx = subset_to_idx[sub]
                    if current_cov[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(test_block, t):
                        idx = subset_to_idx[sub]
                        current_cov[idx] -= 1
                    current_blocks.pop(j)
                    n = len(current_blocks)
                    if n < len(best_blocks_sa):
                        best_len = len(current_blocks)
                        best_blocks_sa = list(current_blocks)
        
        return best_blocks_sa
    
    def fast_redundant_delete(blocks):
        """Quick pass: remove blocks whose every t-subset is covered by others."""
        if not blocks:
            return blocks
        cov = np.zeros(n_subsets, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[subset_to_idx[sub]] += 1
        to_remove = set()
        for i, block in enumerate(blocks):
            redundant = True
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if cov[idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.add(i)
        if to_remove:
            blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        return blocks

    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 45:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
        
        if len(blocks) < best_count:
            blocks = fast_redundant_delete(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            improved = local_search(blocks)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
        
        num_restarts += 1
    
    # Use remaining time for SA on the best solution found
    if best_blocks is not None and time.time() < deadline - 10:
        sa_rng = random.Random(42)
        sa_result = simulated_annealing(best_blocks, sa_rng)
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_blocks = sa_result
            # Try local search again on SA result
            if time.time() < deadline - 5:
                final = local_search(best_blocks)
                if len(final) < best_count:
                    best_count = len(final)
                    best_blocks = final
        # Final redundant deletion pass
        if best_blocks is not None:
            final = fast_redundant_delete(best_blocks)
            if len(final) < best_count:
                best_count = len(final)
                best_blocks = final
    
    if best_blocks is None:
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
    
    return best_blocks
