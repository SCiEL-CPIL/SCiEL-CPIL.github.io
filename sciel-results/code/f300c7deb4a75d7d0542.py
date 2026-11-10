"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Returns the start index and the corresponding smallest positive square roots.
"""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    p = 1009
    m = 5
    
    # Boolean array to check if a number is a quadratic residue
    is_res = np.zeros(p, dtype=bool)
    # Array to store the smallest positive root for each residue
    root_map = np.zeros(p, dtype=np.int32)
    
    # Precompute residues and roots
    # Iterating from 1 to p-1 ensures the first root found is the smallest positive one
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_res[r]:
                is_res[r] = True
                root_map[r] = x
    
    # Find the smallest s such that s, s+1, ..., s+m-1 are all quadratic residues
    # Use NumPy for efficient consecutive-run detection
    # Create a sliding window sum to find positions where all m values are residues
    # is_res[1:p] gives us indices 1 to p-1
    res_slice = is_res[1:p]  # length p-1, indices 0..p-2 correspond to values 1..p-1
    
    # Use convolution or cumulative sum approach for fixed window
    # For window of size m, compute prefix sums
    prefix = np.cumsum(res_slice.astype(np.int32))
    # prefix[i] = sum of res_slice[0..i-1], so prefix[i+m] - prefix[i] = sum of res_slice[i..i+m-1]
    # We need all m values to be True, so sum should equal m
    # Valid starting positions in res_slice: index 0 to len(res_slice)-m
    # which corresponds to value s = index + 1, where s ranges from 1 to p-m
    
    window_sums = prefix[m:] - prefix[:-m]
    # window_sums[j] corresponds to res_slice[j..j+m-1], i.e., values j+1 to j+m
    # So s = j + 1
    
    # Find the first position where window sum equals m
    valid_positions = np.where(window_sums == m)[0]
    
    if len(valid_positions) > 0:
        s = int(valid_positions[0]) + 1
        roots = [int(root_map[s + i]) for i in range(m)]
        return {'start': s, 'roots': roots}
    
    # Fallback: linear search (should not be reached for p=1009, m=5)
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            raise RuntimeError("Deadline exceeded")
        all_res = True
        for i in range(m):
            if not is_res[s + i]:
                all_res = False
                break
        if all_res:
            roots = [int(root_map[s + i]) for i in range(m)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError("No consecutive run of quadratic residues found")
