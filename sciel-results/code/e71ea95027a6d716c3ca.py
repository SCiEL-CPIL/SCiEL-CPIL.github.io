import time
import sys
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights for all 2^N words
    weights = np.array([0] * max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
    
    def build_linear_code(seed_order=None):
        """Build a linear code using greedy basis selection."""
        basis = []
        current_code = [0]
        
        if seed_order is None:
            candidates = list(range(1, max_val))
        else:
            candidates = seed_order
        
        for g in candidates:
            if time.time() > deadline - 20:
                break
            if g == 0:
                continue
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                old_code = current_code
                current_code = []
                for c in old_code:
                    current_code.append(c)
                    current_code.append(c ^ g)
                if len(basis) >= 8:
                    break
        
        return current_code if len(current_code) >= 2 else None
    
    def nonlinear_expand(base_code, time_budget):
        """Expand a base code using multi-restart greedy nonlinear addition."""
        base_set = set(base_code)
        base_arr = np.array(base_code, dtype=np.int32)
        
        # Compute scores for all candidates against the base code
        candidates_by_score = defaultdict(list)
        
        # Vectorized: compute min distance from each candidate to base code
        # Process in chunks to manage memory
        chunk_size = 10000
        all_candidates = []
        all_scores = []
        
        for start in range(0, max_val, chunk_size):
            end = min(start + chunk_size, max_val)
            chunk = np.arange(start, end, dtype=np.int32)
            
            # Filter out base words
            mask = np.array([x not in base_set for x in chunk], dtype=bool)
            chunk = chunk[mask]
            
            if len(chunk) == 0:
                continue
            
            # Compute distances to all base words (vectorized)
            # chunk[:, None] ^ base_arr[None, :] gives XOR matrix
            # This may be memory intensive; use chunked approach
            min_dists = np.full(len(chunk), max_val, dtype=np.int32)
            
            # Process base words in sub-chunks to limit memory
            base_chunk_size = 500
            for bs in range(0, len(base_arr), base_chunk_size):
                be = min(bs + base_chunk_size, len(base_arr))
                base_sub = base_arr[bs:be]
                # XOR matrix: len(chunk) x (be-bs)
                xor_matrix = chunk[:, None] ^ base_sub[None, :]
                dist_matrix = weights[xor_matrix]
                chunk_min = dist_matrix.min(axis=1)
                min_dists = np.minimum(min_dists, chunk_min)
            
            # Filter valid candidates (min_dist >= MIN_D)
            valid_mask = min_dists >= MIN_D
            valid_chunk = chunk[valid_mask]
            valid_scores = min_dists[valid_mask]
            
            for x, s in zip(valid_chunk, valid_scores):
                candidates_by_score[int(s)].append(int(x))
        
        if not candidates_by_score:
            return base_code[:]
        
        sorted_scores = sorted(candidates_by_score.keys(), reverse=True)
        
        best_result = base_code[:]
        best_size = len(base_code)
        
        # Multi-restart greedy with improved local search
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            added_in_phase2 = []
            existing_set = set(base_code)
            existing_arr = np.array(base_code, dtype=np.int32)
            
            for s in sorted_scores:
                group = candidates_by_score[s][:]
                random.shuffle(group)
                
                for x in group:
                    if time.time() > deadline - 5:
                        break
                    
                    # Check against added_in_phase2 using vectorized distance
                    if len(added_in_phase2) > 0:
                        added_arr = np.array(added_in_phase2, dtype=np.int32)
                        dists = weights[x ^ added_arr]
                        if dists.min() < MIN_D:
                            continue
                    
                    added_in_phase2.append(x)
                    existing_set.add(x)
                    existing_arr = np.append(existing_arr, x)
            
            current_size = len(base_code) + len(added_in_phase2)
            if current_size > best_size:
                best_size = current_size
                best_result = base_code + added_in_phase2
            
            # Local improvement: try to add more words or swap
            if time.time() < deadline - 8:
                current_code = best_result[:]
                current_set = set(current_code)
                
                # Try adding high-score candidates that weren't added
                for s in sorted_scores:
                    if time.time() > deadline - 8:
                        break
                    group = candidates_by_score[s]
                    random.shuffle(group)
                    for x in group:
                        if time.time() > deadline - 8:
                            break
                        if x in current_set:
                            continue
                        # Check against all current codewords
                        cur_arr = np.array(current_code, dtype=np.int32)
                        dists = weights[x ^ cur_arr]
                        if dists.min() >= MIN_D:
                            current_code.append(x)
                            current_set.add(x)
                
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_result = current_code
        
        return best_result
    
    # Phase 1: Try multiple linear code constructions
    base_codes = []
    
    # First, deterministic scan
    base_code = build_linear_code()
    if base_code and len(base_code) >= 256:
        base_codes.append(base_code)
    
    # Try a few random orderings if time permits
    if time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            random_order = list(range(1, max_val))
            random.shuffle(random_order)
            bc = build_linear_code(random_order)
            if bc and len(bc) >= 256:
                base_codes.append(bc)
    
    if not base_codes:
        # Fallback: return what we have
        return [0]
    
    # Phase 2: Nonlinear expansion for each base code
    best_overall = []
    best_overall_size = 0
    
    for i, bc in enumerate(base_codes):
        if time.time() > deadline - 10:
            break
        # Allocate remaining time
        remaining = deadline - 8 - time.time()
        if remaining < 2:
            break
        result = nonlinear_expand(bc, remaining)
        if len(result) > best_overall_size:
            best_overall_size = len(result)
            best_overall = result
    
    if not best_overall:
        best_overall = base_codes[0]
    
    return best_overall
