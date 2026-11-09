import numpy as np
import time
import random
import math

N = 18

def _shell_sort_network(N):
    """Generate a Shell sort network."""
    comps = []
    gaps = []
    gap = N // 2
    while gap > 0:
        gaps.append(gap)
        gap //= 2
        
    for gap in gaps:
        for i in range(N - gap):
            comps.append((i, i + gap))
    return comps

def _depth_of_network(comps, N):
    """Calculate parallel depth of the network."""
    depth = [0] * N
    for i, j in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Pre-compute inputs for verification
    # 2^18 = 262144
    # inputs: (262144, 18) uint8
    inputs = np.arange(2**N, dtype=np.uint32)
    bits = 1 << np.arange(N, dtype=np.uint32)
    # matrix is (262144, 18) bool
    # We will copy this for each verification.
    
    def _is_valid(comps):
        if not comps:
            return True
        # Copy inputs
        # Using .copy() on a bool array is fast
        m = (inputs[:, None] & bits[None, :]) > 0
        # m is a fresh array each time? 
        # No, `inputs[:, None] & bits[None, :]` creates a new array.
        # This is the expensive part.
        # Can we optimize?
        # We can pre-compute the base matrix once.
        pass

    # Better: Pre-compute the base matrix once outside the loop.
    base_matrix = (inputs[:, None] & bits[None, :]) > 0
    
    def _is_valid(comps):
        if not comps:
            return True
        m = base_matrix.copy()
        for i, j in comps:
            min_val = np.minimum(m[:, i], m[:, j])
            max_val = np.maximum(m[:, i], m[:, j])
            m[:, i] = min_val
            m[:, j] = max_val
        if N > 1:
            diff = np.diff(m, axis=1)
            return np.all(diff >= 0)
        return True

    # 1. Generate initial network
    comps = _shell_sort_network(N)
    if not _is_valid(comps):
        # Fallback
        comps = []
        for k in range(N):
            if k % 2 == 0:
                for i in range(0, N-1, 2):
                    comps.append((i, i+1))
            else:
                for i in range(1, N-1, 2):
                    comps.append((i, i+1))
    
    best_comps = list(comps)
    best_count = len(comps)
    best_depth = _depth_of_network(best_comps, N)
    
    # 2. Phase 1: Greedy deletion
    i = len(comps) - 1
    while i >= 0 and time.time() < deadline:
        test_comps = comps[:i] + comps[i+1:]
        if _is_valid(test_comps):
            comps = test_comps
            if len(comps) < best_count:
                best_comps = list(comps)
                best_count = len(comps)
                best_depth = _depth_of_network(best_comps, N)
        else:
            i -= 1
            
    # 3. Phase 2: Local Search
    current_comps = list(best_comps)
    current_count = best_count
    
    steps = 0
    max_steps = 10000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        if not current_comps:
            break
            
        # Random move: Move a comparator
        idx = random.randint(0, len(current_comps) - 1)
        comp = current_comps[idx]
        
        new_comps = current_comps[:idx] + current_comps[idx+1:]
        new_idx = random.randint(0, len(new_comps))
        new_comps.insert(new_idx, comp)
        
        if _is_valid(new_comps):
            # Try to delete
            if len(new_comps) > 0:
                del_idx = random.randint(0, len(new_comps) - 1)
                test_comps = new_comps[:del_idx] + new_comps[del_idx+1:]
                if _is_valid(test_comps):
                    current_comps = test_comps
                    current_count = len(current_comps)
                    if current_count < best_count:
                        best_comps = list(current_comps)
                        best_count = current_count
                        best_depth = _depth_of_network(best_comps, N)
                else:
                    current_comps = new_comps
            else:
                current_comps = new_comps
                
    return best_comps
