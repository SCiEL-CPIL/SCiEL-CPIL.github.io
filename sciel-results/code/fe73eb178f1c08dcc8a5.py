import time
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    all_5subs = list(combinations(points, 5))
    
    n_blocks = len(all_blocks)
    n_5subs = len(all_5subs)
    
    # Map 5-subsets to indices for quick lookup
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Build the incidence matrix A where A[i, j] = 1 if all_5subs[i] is a subset of all_blocks[j]
    # Since blocks are size 6 and 5subs are size 5, we can just check subset relationship.
    # For efficiency, we can iterate over blocks and their 5-subsets.
    A = np.zeros((n_5subs, n_blocks), dtype=np.int8)
    
    for j, block in enumerate(all_blocks):
        for sub in combinations(block, 5):
            i = sub_to_idx[sub]
            A[i, j] = 1
            
    # Objective: maximize sum(x)
    c = -np.ones(n_blocks)  # scipy minimizes, so we negate for maximization
    
    # Constraints: A @ x <= 1
    constraint = LinearConstraint(A, ub=1)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrate variables
    integrality = np.ones(n_blocks)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"presolve": True, "time_limit": 100})
        
        if res.success and res.x is not None:
            # Round the solution
            x = res.x.round().astype(int)
            selected_blocks = [list(all_blocks[i]) for i in range(n_blocks) if x[i] == 1]
            
            # Verify the solution just in case
            # Check if any 5-subset is repeated
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
                # Fallback if ILP returned an invalid solution (should not happen)
                pass
                
    except Exception as e:
        pass

    # Fallback: Deterministic construction of S(5,6,12)
    # If ILP fails, we can use a known explicit construction.
    # One standard way is to use the blocks from the Mathieu group M12 action.
    # However, generating M12 orbits is complex. 
    # A simpler fallback is the greedy method from the previous code, but optimized.
    # Or, since we know the answer is 132, we can try a few random restarts quickly.
    
    # Let's implement a fast greedy with randomization as a fallback.
    # Precompute 5-subsets for each block
    block_5subs = [list(combinations(b, 5)) for b in all_blocks]
    
    best_blocks = []
    rng = __import__('random').Random(42)
    
    # Try a few random orderings
    for _ in range(10):
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        used_5subs = set()
        current_blocks = []
        
        for idx in indices:
            b = all_blocks[idx]
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
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            
    return [list(b) for b in best_blocks]
