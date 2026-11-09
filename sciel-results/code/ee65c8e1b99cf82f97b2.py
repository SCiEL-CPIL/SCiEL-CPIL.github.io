# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 590.0  # safety margin before 600s limit
    
    best_points = None
    best_k = 2
    best_set = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    # Try different weights (number of non-zero entries)
    # For weight w vectors with entries in {-1,0,1}, norm^2 = w
    # Need pairwise dot products <= w/2
    
    for w in range(2, 12):
        if time.time() > deadline - 30:
            break
        
        max_dot_allowed = w // 2  # floor(w/2), since dot products are integers
        
        # Generate all weight-w vectors
        vecs = _generate_weight_w_vectors(d, w)
        if not vecs:
            continue
        
        n = len(vecs)
        if n == 0:
            continue
        
        vecs_arr = np.array(vecs, dtype=np.int64)
        
        # Compute all pairwise dot products
        # dot[i,j] = vecs[i] . vecs[j]
        # We need dot[i,j] <= max_dot_allowed for i != j
        
        # Build conflict: conflict[i,j] = True if dot[i,j] > max_dot_allowed
        # Use bitsets for efficiency
        
        # Precompute dot products
        dots = vecs_arr @ vecs_arr.T  # n x n matrix
        
        # Conflict matrix: True where dot > max_dot_allowed (and i != j)
        conflict = (dots > max_dot_allowed)
        np.fill_diagonal(conflict, False)
        
        # Convert to bitsets for fast independent set search
        # conflict_bits[i] is a set of indices j that conflict with i
        conflict_sets = []
        for i in range(n):
            conflict_sets.append(np.where(conflict[i])[0])
        
        # Greedy with multiple restarts
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            if attempt == 0:
                order = np.arange(n)
            else:
                order = np.random.permutation(n)
            
            selected = set()
            
            for idx in order:
                idx = int(idx)
                # Check if idx conflicts with any selected
                compatible = True
                for sel in selected:
                    if conflict[idx, sel]:
                        compatible = False
                        break
                if compatible:
                    selected.add(idx)
            
            k = len(selected)
            if k > best_k:
                best_k = k
                best_set = [vecs_arr[i] for i in sorted(selected)]
        
        # Local search: try to improve by swapping
        if best_k >= 2:
            if time.time() > deadline - 30:
                break
            
            # Get current best set
            if best_set and len(best_set) >= 2:
                current = np.array(best_set, dtype=np.int64)
                current_set = set(range(len(current)))
                
                # Try to add vectors not in current set
                # Check which vectors are compatible with all in current set
                # For each candidate vector, check dot with all current vectors
                
                # This is complex with mixed norms, so let's do local search within same weight
                # Actually, let's just do 1-swap local search
                # For each pair (in_set, not_in_set), try swapping if it improves
                
                # This is getting complex. Let's try a simpler approach:
                # Try to add more vectors by checking compatibility
                for idx in range(n):
                    if time.time() > deadline - 30:
                        break
                    if idx in current_set:
                        continue
                    # Check if vecs_arr[idx] is compatible with all in current set
                    compatible = True
                    for sel_idx in current_set:
                        if conflict[idx, sel_idx]:
                            compatible = False
                            break
                    if compatible:
                        # Add it
                        current = np.vstack([current, vecs_arr[idx]])
                        current_set.add(idx)
                        best_k = len(current)
                        best_set = list(current)
    
    # Also try a different construction: use the standard basis and negatives
    # e_i and -e_i for i=1..11, plus some combinations
    # Norm^2 = 1 for basis vectors, need dot <= 0.5, so dot <= 0
    # e_i . e_j = 0 for i != j, e_i . (-e_j) = 0 for i != j, e_i . (-e_i) = -1
    # So all 22 vectors e_i, -e_i are mutually compatible!
    # Wait: e_i . e_j = 0 <= 0.5 for i != j. Yes!
    # And ||e_i - e_j||^2 = 2 >= 1 = max norm^2. Yes!
    
    basis_vectors = []
    for i in range(d):
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        v2 = np.zeros(d, dtype=np.int64)
        v2[i] = -1
        basis_vectors.append(v1)
        basis_vectors.append(v2)
    
    if len(basis_vectors) > best_k:
        best_k = len(basis_vectors)
        best_set = basis_vectors
    
    # Try to add more vectors to the basis set
    # The basis set has 2d = 22 vectors, all with norm^2 = 1
    # We need vectors v with ||v||^2 >= 1 (since max norm is 1) and
    # ||v - e_i||^2 >= 1 and ||v + e_i||^2 >= 1 for all i
    # ||v - e_i||^2 = ||v||^2 - 2*v_i + 1 >= 1 => ||v||^2 >= 2*v_i
    # ||v + e_i||^2 = ||v||^2 + 2*v_i + 1 >= 1 => ||v||^2 >= -2*v_i
    # So ||v||^2 >= 2*|v_i| for all i
    # Also need ||v - v_j||^2 >= 1 for all j in basis, which is covered above
    
    # Also need ||v - v'||^2 >= 1 for other added vectors
    # And ||v||^2 can be > 1, but then max norm increases
    
    # Let's try to add vectors of norm^2 = 2
    # v with ||v||^2 = 2, so v has two entries of ±1 or one entry of ±2
    # If v has entries in {-1,0,1} with weight 2: ||v||^2 = 2
    # Need 2 >= 2*|v_i| for all i, so |v_i| <= 1, which is satisfied
    # Need ||v - e_i||^2 >= 1: 2 - 2*v_i + 1 >= 1 => 2 >= 2*v_i => v_i <= 1, OK
    # Need ||v + e_i||^2 >= 1: 2 + 2*v_i + 1 >= 1 => 2 >= -2*v_i => v_i >= -1, OK
    
    # So any weight-2 vector with entries in {-1,0,1} is compatible with the basis!
    # But we also need pairwise compatibility among the added vectors
    # Two weight-2 vectors u, v: ||u - v||^2 = 4 - 2*u.v >= 1 => u.v <= 1.5 => u.v <= 1
    # u.v = sum u_i*v_i. If they share one position with same sign: u.v = 1, OK
    # If they share one position with opposite sign: u.v = -1, OK
    # If they share no positions: u.v = 0, OK
    # If they share both positions with same signs: u.v = 2, NOT OK
    # If they share both positions with one same, one opposite: u.v = 0, OK
    
    # So we need weight-2 vectors that don't share both positions with same signs
    
    # Let's generate all weight-2 vectors and find a large compatible set
    weight2_vecs = _generate_weight_w_vectors(d, 2)
    if weight2_vecs:
        n2 = len(weight2_vecs)
        w2_arr = np.array(weight2_vecs, dtype=np.int64)
        
        # Check compatibility with basis vectors (already verified above, all compatible)
        # Check pairwise compatibility among weight-2 vectors
        # u.v <= 1
        
        dots2 = w2_arr @ w2_arr.T
        conflict2 = (dots2 > 1)
        np.fill_diagonal(conflict2, False)
        
        # Greedy to find max independent set
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            if attempt == 0:
                order = np.arange(n2)
            else:
                order = np.random.permutation(n2)
            
            selected2 = set()
            for idx in order:
                idx = int(idx)
                compatible = True
                for sel in selected2:
                    if conflict2[idx, sel]:
                        compatible = False
                        break
                if compatible:
                    selected2.add(idx)
            
            # Total points = 22 (basis) + len(selected2)
            total_k = 22 + len(selected2)
            if total_k > best_k:
                best_k = total_k
                best_set = basis_vectors + [w2_arr[i] for i in sorted(selected2)]
    
    # Validate final result
    if best_set and len(best_set) >= 2:
        result = np.array(best_set, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        
        k = len(result)
        if k >= 2 and result.shape[1] == d:
            norms_sq = np.sum(result ** 2, axis=1)
            if np.all(norms_sq > 0):
                max_norm_sq = np.max(norms_sq)
                
                min_dist_sq = np.inf
                for i in range(k):
                    for j in range(i + 1, k):
                        diff = result[i] - result[j]
                        dist_sq = np.sum(diff ** 2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                
                if min_dist_sq >= max_norm_sq:
                    return result
    
    # Fallback
    return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)


def _generate_weight_w_vectors(d, w):
    """Generate all integer vectors in R^d with exactly w non-zero entries, each ±1."""
    vecs = []
    if w > d:
        return vecs
    
    # Choose w positions out of d
    for positions in itertools.combinations(range(d), w):
        # For each choice of positions, assign ±1
        for signs in itertools.product([-1, 1], repeat=w):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vecs.append(v)
    
    return vecs


# EVOLVE-BLOCK-END
