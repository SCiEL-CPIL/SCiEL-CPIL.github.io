import time
import sys
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights for all 2^N words
    weights = np.array([0] * max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # Precompute offsets with weight < MIN_D
    # These are the "conflict" offsets
    offsets_lt_d = [i for i in range(max_val) if weights[i] < MIN_D]
    
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

    def get_valid_candidates(base_code, time_limit):
        """Find all valid candidates against base code using vectorized distance computation."""
        base_set = set(base_code)
        base_arr = np.array(base_code, dtype=np.int32)
        
        valid_candidates = []
        chunk_size = 4096
        
        for start in range(0, max_val, chunk_size):
            if time.time() > time_limit:
                break
            end = min(start + chunk_size, max_val)
            chunk = np.arange(start, end, dtype=np.int32)
            
            # Compute min distance to base code
            min_dists = np.full(len(chunk), max_val, dtype=np.int32)
            for bs in range(0, len(base_arr), 256):
                be = min(bs + 256, len(base_arr))
                base_sub = base_arr[bs:be]
                xor_mat = chunk[:, None] ^ base_sub[None, :]
                dist_mat = weights[xor_mat]
                min_dists = np.minimum(min_dists, dist_mat.min(axis=1))
            
            valid_mask = min_dists >= MIN_D
            valid_chunk = chunk[valid_mask]
            for x in valid_chunk:
                x_int = int(x)
                if x_int not in base_set:
                    valid_candidates.append(x_int)
        
        return valid_candidates

    def min_degree_greedy(base_code, time_limit):
        """Expand using Minimum Degree Heuristic on the conflict graph."""
        valid_candidates = get_valid_candidates(base_code, time_limit)
        valid_set = set(valid_candidates)
        code = list(base_code)
        
        # Greedy Loop
        while valid_set:
            if time.time() > time_limit:
                break
            
            # Calculate degree for each candidate in valid_set
            best_v = None
            min_deg = float('inf')
            
            for v in valid_set:
                # Count neighbors in valid_set
                deg = 0
                for e in offsets_lt_d:
                    u = v ^ e
                    if u in valid_set:
                        deg += 1
                        if deg >= min_deg:
                            break
                
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            # Add best_v to code
            code.append(best_v)
            
            # Remove best_v and its neighbors from valid_set
            valid_set.discard(best_v)
            neighbors_to_remove = set()
            for e in offsets_lt_d:
                neighbors_to_remove.add(best_v ^ e)
            valid_set.difference_update(neighbors_to_remove)
            
        return code

    def local_search_improve(code, time_limit):
        """Try to improve the code using local search: remove a word, then greedily add back."""
        if time.time() > time_limit:
            return code
        
        code_set = set(code)
        best_code = list(code)
        
        # Try removing each word and then greedily adding back
        # But this is expensive. Let's try a simpler approach:
        # For each word in the code, try to find a replacement that allows more words.
        
        # Actually, let's just try multiple random restarts of the greedy phase
        # with different orderings of valid candidates.
        
        return best_code

    # Phase 1: Linear Code
    base_code = build_linear_code()
    if not base_code or len(base_code) < 256:
        return [0]
        
    # Phase 2: Min Degree Greedy with time limit
    # Leave 10 seconds for potential local search
    result = min_degree_greedy(base_code, deadline - 10)
    
    # Phase 3: If we have time, try local search or alternative strategies
    if time.time() < deadline - 5:
        # Try a different strategy: start with a subset of the base code
        # and see if we can get a larger code by being more selective
        
        # Strategy: Try removing some words from the base code and then expanding
        # This might allow more non-linear words to be added
        
        # For now, just return the result from Phase 2
        pass
    
    return result
