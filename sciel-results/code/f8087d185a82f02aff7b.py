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
    
    # Precompute weights for all 2^N words using numpy for speed
    all_words = np.arange(max_val, dtype=np.int32)
    weights = np.zeros(max_val, dtype=np.int32)
    for bit in range(N):
        mask = 1 << bit
        weights[all_words & mask] += 1
    
    def build_linear_code(seed_order=None):
        """Build a linear code using greedy basis selection with vectorized checks."""
        basis = []
        current_code = [0]
        current_code_arr = np.array(current_code, dtype=np.int32)
        
        if seed_order is None:
            candidates = list(range(1, max_val))
        else:
            candidates = seed_order
        
        for g in candidates:
            if time.time() > deadline - 20:
                break
            if g == 0:
                continue
            
            # Vectorized check: min weight of g ^ c for all c in current_code
            xor_vals = g ^ current_code_arr
            min_w = np.min(weights[xor_vals])
            
            if min_w >= MIN_D:
                basis.append(g)
                # Expand code
                old_code = current_code
                current_code = []
                for c in old_code:
                    current_code.append(c)
                    current_code.append(c ^ g)
                current_code_arr = np.array(current_code, dtype=np.int32)
                
                if len(basis) >= 8:
                    break
        
        return current_code if len(current_code) >= 2 else None
    
    def nonlinear_expand(base_code, time_budget):
        """Expand a base code using bulk compatibility check and greedy addition."""
        base_arr = np.array(base_code, dtype=np.int32)
        base_size = len(base_arr)
        
        # Compute min distance from all 2^N words to the base code
        # Shape: (2^N, base_size)
        # This might use ~131072 * 256 * 4 bytes = 134 MB. Acceptable.
        
        # To save memory/time if base is large, we could chunk, but 256 is small.
        xor_matrix = all_words[:, None] ^ base_arr[None, :]
        
        # Compute weights for the xor matrix
        # w[i, j] = weight(xor_matrix[i, j])
        w = np.zeros((max_val, base_size), dtype=np.int32)
        for bit in range(N):
            m = 1 << bit
            w[xor_matrix & m] += 1
        
        min_dists = np.min(w, axis=1)
        
        # Find candidates with min distance >= MIN_D
        valid_mask = min_dists >= MIN_D
        candidates = all_words[valid_mask]
        cand_dists = min_dists[valid_mask]
        
        if len(candidates) == 0:
            return base_code[:]
        
        # Sort by distance descending (prefer those far from base)
        sort_idx = np.argsort(-cand_dists)
        candidates = candidates[sort_idx]
        
        best_result = base_code[:]
        best_size = len(base_code)
        
        # Multi-restart greedy addition
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Shuffle candidates for diversity in restarts
            # We only shuffle the order of consideration
            shuffled_idx = np.random.permutation(len(candidates))
            shuffled_candidates = candidates[shuffled_idx]
            
            added_in_phase2 = []
            added_set = set()
            added_arr = np.array(added_in_phase2, dtype=np.int32) # empty initially
            
            for x in shuffled_candidates:
                if time.time() > deadline - 5:
                    break
                
                x_int = int(x)
                if x_int in added_set:
                    continue
                
                # Check against added words
                valid = True
                if len(added_in_phase2) > 0:
                    # Vectorized check against added
                    xor_added = x_int ^ added_arr
                    if np.min(weights[xor_added]) < MIN_D:
                        valid = False
                
                if valid:
                    added_in_phase2.append(x_int)
                    added_set.add(x_int)
                    # Update added_arr for next vectorized check
                    # This is O(M) per addition, which is fine if M is small
                    added_arr = np.array(added_in_phase2, dtype=np.int32)
            
            current_size = len(base_code) + len(added_in_phase2)
            if current_size > best_size:
                best_size = current_size
                best_result = base_code + added_in_phase2
        
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
