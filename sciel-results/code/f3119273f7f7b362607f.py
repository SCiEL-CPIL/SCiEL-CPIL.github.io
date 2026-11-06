import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Generate all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    triples_set = set()
    for a in range(1, N + 1):
        a_sq = a * a
        # Iterate u from 1 to a
        for u in range(1, a + 1):
            if a_sq % u == 0:
                v = a_sq // u
                # We need b = a + u and c = a + v
                b = a + u
                c = a + v
                # Check bounds
                if b > N or c > N:
                    continue
                # Check distinctness: b != c => u != v
                if u == v:
                    continue
                # a is always < b and < c
                # Add to set to avoid duplicates (u, v) and (v, u) might generate same set?
                # If we iterate u up to a, and v = a^2/u, then u <= a <= v.
                # So u != v is handled.
                # The set {a, b, c} is unique for this pair.
                triples_set.add(tuple(sorted([a, b, c])))
                
    triples_list = list(triples_set)
    num_triples = len(triples_list)
    
    # 2. Formulate and solve MILP
    # Variables: x_i for i in 1..N (index i-1)
    # Maximize sum(x_i)
    # Subject to: for each triple (a,b,c), x_a + x_b + x_c <= 2
    #             x_i in {0, 1}
    
    best_set = set(numbers) # Fallback: start with all, then remove? No, MILP is better.
    
    try:
        c_obj = np.ones(N)
        
        # Build constraint matrix A
        # Shape: (num_triples, N)
        A = np.zeros((num_triples, N), dtype=np.float64)
        for idx, t in enumerate(triples_list):
            a, b, c_val = t
            A[idx, a - 1] = 1.0
            A[idx, b - 1] = 1.0
            A[idx, c_val - 1] = 1.0
        
        integrality = np.ones(N)
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Constraint: A @ x <= 2
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=2.0)
        
        # Run MILP
        # Note: scipy.optimize.milp uses HiGHS
        milp_result = milp(
            c=c_obj,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )
        
        if milp_result.success and milp_result.x is not None:
            x = milp_result.x
            # Round to nearest integer
            x_rounded = np.round(x).astype(int)
            
            # Ensure binary
            x_rounded = (x_rounded > 0.5).astype(int)
            
            current_set = set()
            for i in range(N):
                if x_rounded[i] == 1:
                    current_set.add(i + 1)
            
            # 3. Greedy Addition Polish
            # Try to add elements from the complement that don't create conflicts
            complement = set(numbers) - current_set
            changed = True
            while changed:
                changed = False
                # Shuffle for randomization? Not strictly necessary, but helps break ties
                # We can just iterate.
                for x_val in list(complement):
                    # Check if adding x_val creates a forbidden triple
                    # A triple is forbidden if all 3 elements are in the set.
                    # If we add x_val, a triple becomes forbidden only if the other 2 are already in current_set.
                    conflict = False
                    for t in triples_list:
                        if x_val in t:
                            others = [v for v in t if v != x_val]
                            if others[0] in current_set and others[1] in current_set:
                                conflict = True
                                break
                    if not conflict:
                        current_set.add(x_val)
                        complement.remove(x_val)
                        changed = True
                        if time.time() > deadline - 5:
                            break
                
                if time.time() > deadline - 5:
                    break
            
            if len(current_set) > 0:
                best_set = current_set
        
    except Exception as e:
        # Fallback: Greedy Removal
        # Start with all numbers, remove elements involved in triples
        # This is a heuristic, but guarantees validity.
        current_set = set(numbers)
        # Sort triples by "density" or just iterate
        for t in triples_list:
            if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                # Remove the one with highest "degree" or just the first one?
                # To maximize size, we should remove the one that is in the most other triples.
                # Precompute degrees
                # For simplicity in fallback, just remove t[0]
                current_set.remove(t[0])
        
        # Try to add back
        removed = set(numbers) - current_set
        for x_val in list(removed):
            conflict = False
            for t in triples_list:
                if x_val in t:
                    others = [v for v in t if v != x_val]
                    if others[0] in current_set and others[1] in current_set:
                        conflict = True
                        break
            if not conflict:
                current_set.add(x_val)
        
        best_set = current_set

    # Final verification
    for t in triples_list:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            # Remove one element to fix
            best_set.remove(t[0])
            # Re-verify? The loop will catch others.
            # This is a safety net.
            pass

    if not best_set:
        best_set = {1}
        
    return list(best_set)
