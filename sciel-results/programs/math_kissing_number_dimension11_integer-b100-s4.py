# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    
    Strategy:
    1. Try to build a large set using vectors with norm squared 4 (four +/-1s).
       Condition for R=4: min_dist_sq >= 4.
       For two 4-norm vectors u, v: ||u-v||^2 = 4+4-2u.v = 8-2u.v.
       Need 8-2u.v >= 4 => 2u.v <= 4 => u.v <= 2.
       
    2. Greedily add vectors from the pool of all 4-norm integer vectors.
       Pool size: C(11,4) * 2^4 = 330 * 16 = 5280.
       
    3. If time permits, try to add 2-norm vectors (two +/-1s) to the set.
       Condition for 2-norm u and 4-norm v: ||u-v||^2 = 2+4-2u.v = 6-2u.v >= 4 => 2u.v <= 2 => u.v <= 1.
       Condition for two 2-norm vectors u, w: ||u-w||^2 = 2+2-2u.w = 4-2u.w >= 4 => 2u.w <= 0 => u.w <= 0.
       
    4. Return the largest valid set found.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Generate all vectors with exactly four non-zero entries, each +1 or -1
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_4.append(v)
    
    # Greedy selection for 4-norm vectors
    selected = []
    for v in candidates_4:
        if time.time() > deadline - 60: # Reserve 60s for rest
            break
            
        valid = True
        # Check against all selected points
        # Using numpy for speed
        if selected:
            # selected is list of arrays, convert to array for dot product
            # To avoid conversion overhead every time, we can maintain a numpy array
            # But appending to numpy array is expensive. 
            # List of arrays is fine for dot product if we convert once per check?
            # Better: maintain a numpy array `sel_arr` and update it.
            # But we don't know final size.
            # Let's just use a list and convert if size is large, or use loop for small size.
            # Given 5280 candidates, and potentially ~1000 selected, 
            # a pure python loop over selected for each candidate is 5280 * 1000 = 5M iterations.
            # Each iteration 11 ops. 55M ops. Might be slow in Python.
            # Let's use numpy.
            
            if len(selected) < 100:
                # Small set, python loop is fine
                for u in selected:
                    if np.dot(u, v) > 2:
                        valid = False
                        break
            else:
                # Large set, use numpy
                sel_arr = np.array(selected, dtype=np.int64)
                dots = sel_arr @ v
                if np.max(dots) > 2:
                    valid = False
        
        if valid:
            selected.append(v)
            
    # Try to add 2-norm vectors
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_2.append(v)
    
    # Shuffle candidates_2 to potentially find a better set?
    # Or just iterate.
    # We need to check against all selected (4-norm and 2-norm).
    # Condition for 2-norm u vs 4-norm v: u.v <= 1
    # Condition for 2-norm u vs 2-norm w: u.w <= 0
    
    # We can add 2-norm vectors if they satisfy conditions.
    # Since 2-norm vectors have stricter constraints with each other (dot <= 0),
    # we might not add many.
    
    for v in candidates_2:
        if time.time() > deadline - 10:
            break
            
        valid = True
        if selected:
            # Check against all selected
            # We need to distinguish between 4-norm and 2-norm in selected?
            # We can just check dot product and infer norm?
            # Or store norms.
            # Let's just compute dot product and check based on norm of selected point.
            # To avoid recomputing norms, we can store them.
            # But for simplicity, let's just check:
            # If selected point is 4-norm: dot <= 1
            # If selected point is 2-norm: dot <= 0
            
            # We can identify norm by checking sum of squares.
            # Or we can just check both conditions? No, they are different.
            
            # Optimization: Most selected are 4-norm.
            # Let's assume selected are mostly 4-norm.
            # We can check dot <= 1 for all, and then refine for 2-norms?
            # No, if dot is 1 with a 2-norm point, it's invalid (needs <= 0).
            # If dot is 1 with a 4-norm point, it's valid.
            
            # Let's just loop.
            for u in selected:
                dot = np.dot(u, v)
                # Determine norm of u
                # u is int64 array.
                # If u has 4 non-zeros, norm 4.
                # If u has 2 non-zeros, norm 2.
                # We can check np.count_nonzero(u)
                # But that's expensive.
                # We can store norms in a parallel list.
                pass
        
        # This loop is getting complex and potentially slow.
        # Given that 4-norms alone might yield > 220, and 2-norms add complexity,
        # let's skip adding 2-norms for now to ensure we finish and return a good 4-norm set.
        # If the 4-norm set is small, we can revisit.
        
    # Convert to array
    if not selected:
        # Fallback to 220 points
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64)
                p1[i] = 1
                p1[j] = 1
                points.append(p1)
                p2 = np.zeros(d, dtype=np.int64)
                p2[i] = -1
                p2[j] = -1
                points.append(p2)
                p3 = np.zeros(d, dtype=np.int64)
                p3[i] = 1
                p3[j] = -1
                points.append(p3)
                p4 = np.zeros(d, dtype=np.int64)
                p4[i] = -1
                p4[j] = 1
                points.append(p4)
        return np.array(points, dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
