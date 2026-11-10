# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points maximizing k
    subject to: min pairwise squared distance >= max squared norm.
    
    Strategy:
    1. Use vectors with exactly 4 non-zero entries (each +/-1), so norm_sq = 4.
       Condition: for any two selected u, v with norm_sq 4:
         ||u-v||^2 = 4+4-2*u.v = 8-2*u.v >= 4  =>  u.v <= 2
       So we need a set of 4-weight binary vectors with pairwise dot <= 2.
    
    2. Greedily build such a set from all C(11,4)*2^4 = 5280 candidates.
       Use a smart ordering to maximize the set size.
    
    3. After the 4-norm set is built, try to add 2-norm vectors (two +/-1s).
       For 2-norm u and 4-norm v: ||u-v||^2 = 6-2*u.v >= 4 => u.v <= 1
       For two 2-norm u, w: ||u-w||^2 = 4-2*u.w >= 4 => u.w <= 0
       Note: max norm becomes max(4, 2) = 4, so all distances must be >= 4.
       But wait: if we add 2-norm vectors, the max norm is still 4 (from 4-norm vectors).
       So the condition remains: all pairwise distances >= 4.
       For two 2-norm vectors: ||u-w||^2 = 4-2*u.w. Need >= 4 => u.w <= 0.
       For 2-norm u and 4-norm v: ||u-v||^2 = 6-2*u.v. Need >= 4 => u.v <= 1.
    
    4. We can also try adding vectors with other norms, but 4-norm seems optimal
       because it gives a good balance.
    
    Key insight for maximizing: The problem is equivalent to finding a large code
    in {0, +1, -1}^11 with weight 4, where pairwise dot product <= 2.
    
    To maximize, we should try multiple random orderings or use a smarter greedy.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin: 50s buffer
    
    # Generate all 4-norm vectors (exactly 4 non-zero entries, each +/-1)
    # Pool size: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_4.append(v)
    
    # Convert to numpy array for fast operations
    cand_arr = np.array(candidates_4, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Greedy selection with multiple random restarts to find a larger set
    best_selected = []
    best_size = 0
    
    # Number of restarts based on available time
    # Each restart: try greedy on a random permutation of candidates
    # But 5280 candidates * ~1000 selected * 11 dims = ~58M ops per restart in Python
    # With numpy, it's faster. Let's do a few restarts.
    
    np.random.seed(42)
    
    # First, try a deterministic greedy: sort by some heuristic
    # Then try random restarts
    
    num_restarts = 5  # Start with 5, can adjust
    
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Create a random permutation of candidate indices
        order = np.random.permutation(n_cand)
        
        selected = []
        selected_indices = set()
        
        # Maintain a numpy array of selected for fast dot product checks
        # We'll use a list and convert periodically, or just loop for small sets
        
        for idx in order:
            if time.time() > deadline - 30:
                break
            
            v = cand_arr[idx]
            
            # Check if v can be added
            # For 4-norm vectors, condition is u.v <= 2 for all selected u
            valid = True
            
            if selected:
                # Use numpy for efficiency when set is large
                if len(selected) < 50:
                    # Small set: Python loop
                    for u in selected:
                        if np.dot(u, v) > 2:
                            valid = False
                            break
                else:
                    # Large set: numpy
                    sel_arr = np.array(selected, dtype=np.int64)
                    dots = sel_arr @ v
                    if np.max(dots) > 2:
                        valid = False
            
            if valid:
                selected.append(v)
                selected_indices.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    # Now try to add 2-norm vectors to the best set
    if best_selected:
        # Generate all 2-norm vectors
        candidates_2 = []
        for combo in itertools.combinations(range(d), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = np.zeros(d, dtype=np.int64)
                for idx, val in zip(combo, signs):
                    v[idx] = val
                candidates_2.append(v)
        
        cand2_arr = np.array(candidates_2, dtype=np.int64)
        
        # For each 2-norm candidate, check if it can be added
        # Conditions:
        #   - For each 4-norm selected u: u.v <= 1
        #   - For each 2-norm selected w: w.v <= 0
        
        # Separate selected into 4-norm and 2-norm
        # All current selected are 4-norm, so initially:
        sel_4 = best_selected[:]
        sel_2 = []
        
        for v in cand2_arr:
            if time.time() > deadline - 10:
                break
            
            valid = True
            
            # Check against 4-norm selected
            if sel_4:
                sel4_arr = np.array(sel_4, dtype=np.int64)
                dots4 = sel4_arr @ v
                if np.max(dots4) > 1:
                    valid = False
            
            if valid and sel_2:
                sel2_arr = np.array(sel_2, dtype=np.int64)
                dots2 = sel2_arr @ v
                if np.max(dots2) > 0:
                    valid = False
            
            if valid:
                sel_2.append(v)
        
        # Combine
        best_selected = sel_4 + sel_2
    
    # If we didn't get a good set, fallback to the 220-point construction
    if len(best_selected) < 220:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        # This gives 55 * 4 = 220 points, all with norm_sq = 2
        # Check: for any two, ||u-v||^2 = 4-2*u.v
        # u.v can be: if they share no indices: 0, dist^2 = 4 >= 2 OK
        # if they share one index: u.v = s1*t1 or s1*t2, can be -1,0,1
        #   if u.v = 1: dist^2 = 4-2 = 2 >= 2 OK
        #   if u.v = 0: dist^2 = 4 >= 2 OK  
        #   if u.v = -1: dist^2 = 4+2 = 6 >= 2 OK
        # if they share both indices: u.v = s1*t1 + s2*t2, can be -2,0,2
        #   if u.v = 2: dist^2 = 4-4 = 0 < 2 INVALID!
        # So this construction is INVALID when two vectors share both indices and have same signs.
        # We need to fix this.
        
        # Better fallback: use only vectors where for each pair of indices, we pick
        # a subset of sign combinations that are pairwise compatible.
        # Actually, let's just use the 4-norm greedy result even if small,
        # or construct a valid set differently.
        
        # Simple valid fallback: all vectors with exactly 2 non-zeros, but only
        # one sign combination per index pair (e.g., always +1, +1)
        # That gives 55 points. Not great.
        
        # Let's just return what we have if it's valid, otherwise use a minimal valid set.
        if not best_selected:
            # Minimal valid set: two orthogonal vectors
            p1 = np.zeros(d, dtype=np.int64)
            p1[0] = 1
            p1[1] = 1
            p2 = np.zeros(d, dtype=np.int64)
            p2[2] = 1
            p2[3] = 1
            best_selected = [p1, p2]
    
    # Final validation and conversion
    if not best_selected:
        p1 = np.zeros(d, dtype=np.int64)
        p1[0] = 1
        p1[1] = 1
        p2 = np.zeros(d, dtype=np.int64)
        p2[2] = 1
        p2[3] = 1
        return np.array([p1, p2], dtype=np.int64)
    
    result = np.array(best_selected, dtype=np.int64)
    
    # Round and cast as required
    result = np.around(result).astype(np.int64)
    
    return result

# EVOLVE-BLOCK-END
