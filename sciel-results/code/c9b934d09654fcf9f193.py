import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    N = 17
    MIN_D = 6
    max_val = 1 << N

    # Precompute weights
    weights = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)

    # 1. Construct the 256-word linear code
    # The [18,9,6] QR code shortened.
    # We can generate a basis for the linear code.
    # A known generator matrix for the [17,8,6] code (shortened [18,9,6])?
    # Actually, the seed program constructs it greedily. Let's stick to that but optimize it.
    # Or, use a known generator matrix.
    # The shortened extended QR code of length 17 is a [17,8,6] code?
    # Wait, [18,9,6] has dimension 9. Shortening one coordinate reduces dimension by at most 1.
    # If the shortened coordinate is not in the support of any basis vector, dim stays 9?
    # No, shortening a linear code of dimension k usually results in dimension k or k-1.
    # The problem statement says "The 256-word linear reference... shortened by one coordinate".
    # $2^8 = 256$. So the shortened code has dimension 8.
    
    # Let's use the greedy method from the seed, it worked.
    def build_linear_code():
        basis = []
        current_code = [0]
        # To ensure we get the specific 256-word code, we should use a deterministic order
        # or just any order that yields a [17,8,6] code.
        # The greedy method with natural order 1, 2, ... usually finds a good basis.
        for g in range(1, max_val):
            if time.time() > deadline - 30:
                break
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_code = []
                for c in current_code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                current_code = new_code
                if len(basis) == 8: # 2^8 = 256 words
                    break
        return current_code

    base_code = build_linear_code()
    if not base_code or len(base_code) < 256:
        # Fallback: return what we have
        return base_code if base_code else [0]

    base_set = set(base_code)
    
    # 2. Find all valid candidates V
    # x is valid if min(dist(x, c) for c in base_code) >= MIN_D
    # We can check this by ensuring (x ^ c) is not in offsets_lt_d for any c in base_code.
    # offsets_lt_d = {i : weight(i) < 6}
    offsets_lt_d_set = set(i for i in range(max_val) if weights[i] < MIN_D)
    
    valid_candidates = []
    # Iterate through all x
    # Optimization: Vectorize the check?
    # For each x, we need to check if (x ^ c) is in offsets_lt_d_set for any c in base_code.
    # This is equivalent to: x is NOT in (base_code + offsets_lt_d) (Minkowski sum / XOR sum).
    # Forbidden set F = { c ^ o | c in base_code, o in offsets_lt_d }
    # Valid candidates are those NOT in F and NOT in base_code.
    
    # Constructing F explicitly might be memory heavy if we use a set of 131k elements, but it's fine.
    # |base_code| = 256. |offsets_lt_d| = sum_{i=0}^{5} C(17,i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402.
    # Size of F <= 256 * 9402 ~ 2.4M. But max_val is 131k. So F will likely cover most of the space?
    # If F covers most of the space, valid_candidates will be small.
    
    # Let's compute F as a boolean array or set.
    forbidden = np.zeros(max_val, dtype=bool)
    base_arr = np.array(base_code, dtype=np.int32)
    offsets_arr = np.array(list(offsets_lt_d_set), dtype=np.int32)
    
    # Vectorized XOR and mark forbidden
    # We can do this in chunks to avoid large temporary arrays
    chunk_size = 64
    for i in range(0, len(base_arr), chunk_size):
        b_chunk = base_arr[i:i+chunk_size]
        # b_chunk shape (k, 1), offsets_arr shape (1, m)
        xor_mat = b_chunk[:, None] ^ offsets_arr[None, :]
        # Flatten and mark
        forbidden[xor_mat.flatten()] = True
        
    # Mark base_code as forbidden too (we don't want to add them again)
    forbidden[base_arr] = True
    
    valid_candidates = np.where(~forbidden)[0]
    
    if len(valid_candidates) == 0:
        return base_code

    # 3. Greedy selection from valid_candidates
    # We want to select a subset S of valid_candidates such that d(x, y) >= 6 for all x, y in S.
    # This is the independence number of the conflict graph on valid_candidates.
    
    # Precompute conflicts among valid_candidates?
    # |valid_candidates| is likely small (around 100-300).
    # Let's check the size.
    # If it's small, we can build the adjacency list.
    
    vc_list = valid_candidates.tolist()
    vc_set = set(vc_list)
    
    # Build conflict graph
    # conflict[u] = set of v in vc_list such that d(u, v) < 6
    
    # To speed up, we can just do a greedy loop with random restarts.
    best_size = 0
    best_subset = []
    
    # Number of restarts depends on time
    # Each restart is O(|V|^2) roughly.
    # If |V| = 200, 40k ops. Very fast.
    # We can do many restarts.
    
    restarts = 100
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
            
        # Random order
        candidates = vc_list.copy()
        random.shuffle(candidates)
        
        selected = []
        selected_set = set()
        
        for v in candidates:
            # Check if v is compatible with all selected
            ok = True
            for s in selected:
                if weights[v ^ s] < MIN_D:
                    ok = False
                    break
            if ok:
                selected.append(v)
                selected_set.add(v)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_subset = selected
            
        # Early stop if we found a large set?
        # We know we need at least 2 to beat 256 (256+2=258).
        # If best_size >= 3, we are likely good.
        # Actually, we want to maximize.
        
    # Combine base_code and best_subset
    final_code = base_code + best_subset
    
    return final_code

# Run a quick self-test? No, just return.
