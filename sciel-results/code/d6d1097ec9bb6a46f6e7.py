import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Precompute 5-subsets for each block
    block_5subs = [list(combinations(b, 5)) for b in all_blocks]
    
    # Map 5-subset to a unique ID
    all_5subs = set()
    for subs in block_5subs:
        for s in subs:
            all_5subs.add(s)
    all_5subs = list(all_5subs)
    sub_to_id = {s: i for i, s in enumerate(all_5subs)}
    n_5subs = len(all_5subs)
    
    # Build the incidence matrix A where A[i, j] = 1 if block i contains 5-subset j
    # We want to maximize sum(x_i) subject to A @ x <= 1
    # Using sparse representation for efficiency if possible, but scipy.milp takes dense or sparse
    # Let's try to build it efficiently.
    
    # Initialize A as a numpy array of zeros
    # n_blocks = 924, n_5subs = 792. 924*792 is ~730k floats. This is small enough for dense.
    A = np.zeros((n_5subs, n_blocks), dtype=np.float64)
    
    for i, b in enumerate(all_blocks):
        for s in block_5subs[i]:
            j = sub_to_id[s]
            A[j, i] = 1.0
            
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -np.ones(n_blocks)
    
    # Constraints: A @ x <= 1
    # x_i in {0, 1}
    
    # Define bounds
    bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
    
    # Define linear constraint
    # A @ x <= 1
    # scipy.milp expects LinearConstraint(A, lb, ub)
    # Here A is the constraint matrix. Note: scipy.milp takes A as (m, n) where m is constraints, n is variables.
    # Our A is (n_5subs, n_blocks), which fits (m, n).
    lincons = LinearConstraint(A, lb=np.zeros(n_5subs), ub=np.ones(n_5subs))
    
    # Integrality: all variables are integer
    integrality = np.ones(n_blocks)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=[lincons],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100.0}  # Leave some time for post-processing
        )
        
        if res.status == 0:  # Optimal
            # res.x contains the optimal solution
            x = res.x
            selected_indices = [i for i in range(n_blocks) if x[i] > 0.5]
            selected_blocks = [list(all_blocks[i]) for i in selected_indices]
            
            # Verify validity just in case
            used_5subs = set()
            valid = True
            for b in selected_blocks:
                for s in combinations(b, 5):
                    if s in used_5subs:
                        valid = False
                        break
                    used_5subs.add(s)
                if not valid:
                    break
            
            if valid:
                return selected_blocks
            else:
                # Fallback if solver returned invalid (should not happen)
                pass
                
    except Exception as e:
        pass # Fallback to heuristic

    # Fallback Heuristic: Randomized Greedy with Restart
    # If ILP fails or times out, we use a fast heuristic to get a good score.
    # We want to maximize the number of blocks.
    
    best_blocks = []
    best_count = 0
    
    # Precompute conflict neighbors for faster heuristic
    # Two blocks conflict if they share a 5-subset.
    # This is equivalent to saying they are not disjoint in a specific way?
    # Actually, two 6-subsets of 12 points can share 0, 1, ..., 6 points.
    # They share a 5-subset if they share exactly 5 points.
    # If they share 6, they are identical.
    # So conflict if intersection size is 5.
    
    # To speed up heuristic, we can just use the 5-subset set check.
    
    rng = random.Random(42)
    
    # Try a few random orderings
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        # Shuffle blocks
        shuffled_indices = list(range(n_blocks))
        rng.shuffle(shuffled_indices)
        
        used_5subs = set()
        current_blocks = []
        
        for idx in shuffled_indices:
            b = all_blocks[idx]
            # Check if any 5-subset of b is in used_5subs
            conflict = False
            for s in block_5subs[idx]:
                if s in used_5subs:
                    conflict = True
                    break
                    
            if not conflict:
                current_blocks.append(b)
                for s in block_5subs[idx]:
                    used_5subs.add(s)
                
                if len(current_blocks) == 132:
                    return [list(b) for b in current_blocks]
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
    return best_blocks

# Note: scipy.milp might be slow on the first call or if the problem is hard.
# However, S(5,6,12) is a well-known structure. The ILP relaxation might be tight or close.
# If it takes too long, the time_limit in options will trigger, and it will return the best found or status 2/3.
# We handle that by falling back to the heuristic.
