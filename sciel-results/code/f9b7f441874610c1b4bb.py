"""
Strategy: Simulated Annealing for Kissing Number in R^11.
Instead of greedy filling, we treat the set of vectors as a state in a Markov Chain.
Moves:
1. Add a random vector (if valid).
2. Remove a random vector (if it allows adding 2-3 others later, or just to shuffle).
3. Swap: Remove one, Add one (if valid).
4. 2-for-1: Remove two, Add three (if valid).
We use a dynamic pool that starts with low-norm vectors and expands to higher norms 
if the low-norm space is exhausted, ensuring we explore the full geometric landscape.
"""

import time
import numpy as np
from itertools import combinations, product

D = 11

# Pre-generate candidate pools for different shells (weights)
# We will use a lazy generation strategy or pre-generate up to weight 6.
# Weight 1: 11 vectors
# Weight 2: 110 vectors (e_i - e_j)
# Weight 3: 11*10*9 * 2 / 6 * 8? No.
# Number of vectors with weight k (nonzero entries in {1, -1}):
# C(D, k) * 2^k.
# k=1: 11 * 2 = 22 (but we can restrict to positive for symmetry? No, signs matter for distance)
# Actually, for kissing number, we usually look at codes.
# Let's generate all vectors with weight up to 6.
# k=1: 22
# k=2: 55 * 4 = 220
# k=3: 165 * 8 = 1320
# k=4: 330 * 16 = 5280
# k=5: 462 * 32 = 14784
# k=6: 462 * 64 = 29568
# Total ~ 50k vectors. This is manageable for precomputation of norms, 
# but dot products are O(N^2 * D). 
# We will not precompute the full Gram matrix (50k x 50k is too big).
# We will compute dots on the fly.

def _generate_pool(max_weight=6):
    """Generate all integer vectors with entries in {-1, 0, 1} and weight <= max_weight."""
    vecs = []
    # We can use a recursive generator or itertools.
    # For efficiency, we iterate over subsets of indices.
    for k in range(1, max_weight + 1):
        for comb in combinations(range(D), k):
            for signs in product([1, -1], repeat=k):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(comb, signs):
                    v[idx] = s
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)

# Pre-compute the pool once. 
# 50k vectors * 11 dims * 8 bytes = ~4.4 MB. Very fast.
POOL = _generate_pool(max_weight=6)
N_CAND = len(POOL)
NORMS_SQ = np.sum(POOL * POOL, axis=1)

# Sort candidates by norm squared to prioritize denser packing (lower norms)
SORT_ORDER = np.argsort(NORMS_SQ)
SORTED_POOL = POOL[SORT_ORDER]
SORTED_NORMS_SQ = NORMS_SQ[SORT_ORDER]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    # State: Set of indices into SORTED_POOL
    # We use a list for the current set to allow fast iteration, 
    # and a set for O(1) membership checks.
    
    # Initialize with a greedy seed to get a good starting point
    current_indices = []
    current_set = set()
    
    # Greedy initialization
    # We iterate through sorted candidates and add if compatible.
    # To speed this up, we maintain the current set as a matrix.
    # However, for the first few hundred, a simple list check is fine.
    # Let's do a fast greedy pass.
    
    C = [] # List of vectors in current set
    C_norms = [] # List of norms squared
    C_idx = [] # List of indices in SORTED_POOL
    
    # Optimization: Use numpy array for C if it gets large
    # But for initialization, let's just loop.
    # We can vectorize the check against the current set.
    
    # Since N_CAND is ~50k, and we want to be fast, we will do a 
    # "batched" greedy: 
    # 1. Take first 1000 candidates.
    # 2. Greedy select from them.
    # 3. Repeat with next batch? 
    # No, greedy order matters. 
    # Let's just loop through SORTED_POOL and check compatibility.
    # To make it fast, we keep C as a numpy array and update it.
    
    # Start with empty
    C_arr = np.empty((0, D), dtype=np.int64)
    C_norms_arr = np.empty(0, dtype=np.int64)
    
    # Greedy Phase
    for i in range(N_CAND):
        if time.time() > deadline - 60: break # Leave 60s for annealing
        
        v = SORTED_POOL[i]
        r_sq = SORTED_NORMS_SQ[i]
        
        if len(C_arr) == 0:
            C_arr = np.vstack([C_arr, v])
            C_norms_arr = np.append(C_norms_arr, r_sq)
            current_indices.append(i)
            current_set.add(i)
            continue
            
        # Check compatibility
        # dots = C_arr @ v
        # dists_sq = r_sq + C_norms_arr - 2 * dots
        # reqs = np.maximum(C_norms_arr, r_sq)
        # if np.all(dists_sq >= reqs):
        
        # Optimization: If we have many vectors, this is O(N_sel * D).
        # With N_sel ~ 500, this is fast.
        
        dots = C_arr @ v
        dists_sq = r_sq + C_norms_arr - 2 * dots
        reqs = np.maximum(C_norms_arr, r_sq)
        
        if np.all(dists_sq >= reqs):
            C_arr = np.vstack([C_arr, v])
            C_norms_arr = np.append(C_norms_arr, r_sq)
            current_indices.append(i)
            current_set.add(i)
            
    # Annealing Phase
    # We want to maximize len(current_indices).
    # We will perform random moves.
    # Temperature T starts high and decays.
    
    n_sel = len(C_arr)
    best_k = n_sel
    best_C_arr = C_arr.copy()
    best_C_norms_arr = C_norms_arr.copy()
    best_indices = current_indices.copy()
    best_set = current_set.copy()
    
    # For efficiency in annealing, we need to quickly check if a vector can be added.
    # We will maintain C_arr and C_norms_arr.
    
    # We need a way to remove a vector from C_arr efficiently.
    # Since N_sel is small (< 1000), rebuilding the array on removal is O(N_sel).
    # This is acceptable if removals are not too frequent.
    # However, we can also just mark indices as "active" and use a list of active indices.
    # Let's stick to the list-of-indices approach for the set, and rebuild the matrix 
    # only when necessary? No, matrix multiplication is the bottleneck for checking.
    # Let's keep C_arr as the source of truth for geometry, and a list for indices.
    
    # To remove index i from C_arr:
    # C_arr = np.delete(C_arr, i, axis=0)
    # This is O(N_sel * D).
    
    # Moves:
    # 1. Try to add a random candidate.
    # 2. If fails, try to remove a random vector from current set and add the candidate?
    #    This is a swap.
    # 3. Try to remove 2, add 3?
    
    # To make this effective, we need to identify "conflict" vectors.
    # If candidate v conflicts with vector u in the set, removing u might allow v to be added.
    
    # Let's implement a "conflict-guided" swap.
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 5:
        
        # 1. Try to add a random candidate
        # Pick a random index from the pool
        v_idx = np.random.randint(0, N_CAND)
        
        if v_idx not in current_set:
            v = SORTED_POOL[v_idx]
            r_sq = SORTED_NORMS_SQ[v_idx]
            
            if n_sel > 0:
                dots = C_arr @ v
                dists_sq = r_sq + C_norms_arr - 2 * dots
                reqs = np.maximum(C_norms_arr, r_sq)
                conflict_mask = dists_sq < reqs
                n_conflicts = np.sum(conflict_mask)
            else:
                conflict_mask = np.array([])
                n_conflicts = 0
                
            if n_conflicts == 0:
                # Add it
                C_arr = np.vstack([C_arr, v])
                C_norms_arr = np.append(C_norms_arr, r_sq)
                current_indices.append(v_idx)
                current_set.add(v_idx)
                n_sel += 1
                
                if n_sel > best_k:
                    best_k = n_sel
                    best_C_arr = C_arr.copy()
                    best_C_norms_arr = C_norms_arr.copy()
                    best_indices = current_indices.copy()
                    best_set = current_set.copy()
                    
                T = max(min_T, T * cooling)
                continue
                
            else:
                # We have conflicts.
                # Try a swap: Remove one conflict, Add v_idx.
                # This is valid if v_idx is compatible with (Set - {conflict}).
                # We know it conflicts with at least one.
                # If we remove exactly one conflict, we must ensure it doesn't conflict with others.
                # So, we can only do this if n_conflicts == 1.
                
                if n_conflicts == 1:
                    # Find the index of the conflict in C_arr
                    # conflict_mask is a boolean array of length n_sel
                    # The position in C_arr is the index in the mask.
                    pos = np.where(conflict_mask)[0][0]
                    rem_idx = current_indices[pos]
                    
                    # Perform swap
                    # Remove rem_idx from set and arrays
                    current_set.remove(rem_idx)
                    current_indices.pop(pos)
                    
                    # Rebuild C_arr and C_norms_arr? 
                    # Or just update in place?
                    # np.delete is cleaner but slower.
                    # Let's do in-place update if possible?
                    # C_arr is a numpy array.
                    # We can overwrite C_arr[pos] with v?
                    # But we also need to update norms.
                    # And we need to ensure the rest of the set is compatible with v.
                    # We already checked that v is compatible with ALL except the one at 'pos'.
                    # So yes, overwriting is valid.
                    
                    C_arr[pos] = v
                    C_norms_arr[pos] = r_sq
                    current_indices[pos] = v_idx
                    current_set.add(v_idx)
                    
                    # n_sel remains same
                    # This move changes the geometry.
                    # Accept it with probability 1 (it's a neutral move in terms of k, 
                    # but might open up future additions).
                    # Actually, to avoid cycling, we should only accept if it improves "margin" 
                    # or randomly. Let's just accept it to shuffle.
                    
                else:
                    # Multiple conflicts.
                    # Try a 2-for-1 move? Remove 2 conflicts, add v_idx?
                    # This decreases k by 1. We should only do this if T is high.
                    # Or try to remove 1 conflict and add v_idx, even if it conflicts with others?
                    # No, that would be invalid.
                    
                    # Let's try to remove a random vector from the set (not necessarily a conflict)
                    # and see if we can add v_idx?
                    # This is a random swap.
                    
                    # Or, just skip.
                    # To prevent stagnation, let's try a random removal of a non-conflict vector?
                    # No, that's bad.
                    
                    # Let's try to find a vector in the set that has "low margin" 
                    # and swap it with v_idx?
                    # Too complex.
                    
                    # Simple strategy: Do nothing for this candidate.
                    pass
                    
        # 2. Random Removal (to escape local minima)
        # With small probability, remove a random vector to allow future additions.
        # This is risky but can help if we are stuck.
        if n_sel > 10 and np.random.random() < 0.01:
            pos = np.random.randint(0, n_sel)
            rem_idx = current_indices[pos]
            
            current_set.remove(rem_idx)
            current_indices.pop(pos)
            
            # Rebuild arrays
            # This is O(N_sel)
            # We can do this by slicing
            if pos == n_sel - 1:
                C_arr = C_arr[:-1]
                C_norms_arr = C_norms_arr[:-1]
            else:
                C_arr = np.concatenate([C_arr[:pos], C_arr[pos+1:]])
                C_norms_arr = np.concatenate([C_norms_arr[:pos], C_norms_arr[pos+1:]])
                
            n_sel -= 1
            
            # Don't update best_k here, as we lost a vector.
            # But this might allow adding 2-3 later.
            
        T = max(min_T, T * cooling)
        
    # Return the best set found
    if best_indices:
        return SORTED_POOL[best_indices]
    else:
        # Fallback
        return np.array([np.zeros(D, dtype=np.int64) + 1], dtype=np.int64)

# EVOLVE-BLOCK-END
