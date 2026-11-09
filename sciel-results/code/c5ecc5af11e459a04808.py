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

    def min_degree_greedy(base_code):
        """Expand using Minimum Degree Heuristic on the conflict graph."""
        base_set = set(base_code)
        # Find all valid candidates against base code
        # We can do this vectorized or loop
        # Vectorized:
        base_arr = np.array(base_code, dtype=np.int32)
        
        # We need to find all x such that min(dist(x, c) for c in base) >= MIN_D
        # This is equivalent to x not in union of spheres of radius MIN_D-1 around base words.
        # Since N=17, we can just iterate all x and check.
        # 131k * 256 checks is ~33M ops. Might be slow in Python loop.
        # Let's use the offset trick.
        # x is invalid if (x ^ c) in offsets_lt_d for any c in base.
        # So x is valid if for all c in base, (x ^ c) NOT in offsets_lt_d_set.
        
        offsets_set = set(offsets_lt_d)
        
        valid_candidates = []
        # Optimization: Check x against base words.
        # If we precompute for each x, the min dist to base?
        # Let's just loop. 131k iterations. Inside, loop over 256 base words? 33M.
        # 33M Python ops is ~1-2 seconds. Acceptable.
        
        # Better: Use NumPy to compute distances to base code in chunks.
        chunk_size = 4096
        for start in range(0, max_val, chunk_size):
            if time.time() > deadline - 10:
                break
            end = min(start + chunk_size, max_val)
            chunk = np.arange(start, end, dtype=np.int32)
            
            # Mask out base words
            # We can't easily mask with set in numpy, but we can filter later.
            
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
                if x_int not in base_set: # Ensure not in base
                    valid_candidates.append(x_int)
        
        valid_set = set(valid_candidates)
        code = list(base_code)
        
        # Greedy Loop
        while valid_set:
            if time.time() > deadline - 5:
                break
            
            # Calculate degree for each candidate in valid_set
            # degree(v) = count of u in valid_set such that (v^u) in offsets_set
            # We iterate over valid_set.
            # If |valid_set| is small, this is fast.
            # Initially |valid_set| might be ~300-500.
            # 500 * 9402 lookups = 4.7M lookups. ~0.2s.
            
            best_v = None
            min_deg = float('inf')
            
            # To speed up, we can sample if valid_set is huge, but it shrinks fast.
            # We iterate all.
            for v in valid_set:
                # Count neighbors in valid_set
                deg = 0
                # Optimization: If deg exceeds current min_deg, break early
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
            # Remove neighbors
            # neighbors = {best_v ^ e for e in offsets_lt_d}
            # valid_set.difference_update(neighbors)
            # This creates a set of 9402 elements.
            # difference_update is efficient.
            neighbors_to_remove = set()
            for e in offsets_lt_d:
                neighbors_to_remove.add(best_v ^ e)
            valid_set.difference_update(neighbors_to_remove)
            
        return code

    # Phase 1: Linear Code
    base_code = build_linear_code()
    if not base_code or len(base_code) < 256:
        return [0]
        
    # Phase 2: Min Degree Greedy
    result = min_degree_greedy(base_code)
    
    # Phase 3: Try a few random restarts of the linear code construction 
    # to see if a different base helps? 
    # Or just run min_degree_greedy on the best base?
    # The base is likely the same optimal linear code.
    
    return result
