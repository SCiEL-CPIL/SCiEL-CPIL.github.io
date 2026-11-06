"""Combined program: Exact MILP with Bitset-Optimized Conflict Graph and Robust Heuristic Fallback."""
# EVOLVE-BLOCK-START
import time
from math import lcm
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit (5s buffer)
    
    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute incompatibility (conflict) pairs using Bitsets for speed
    # conflict[i] is an integer bitmask where bit j is set if candidates[i] and candidates[j] conflict
    # i.e., lcm(candidates[i], candidates[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # 3. Try to solve exactly using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Weights: 1/x
        weights = np.array([1.0 / x for x in candidates])
        
        # Variables: x_i in {0, 1}
        integrality = np.ones(n)
        
        # Objective: Maximize sum(w_i * x_i) -> Minimize -sum(w_i * x_i)
        c = -weights
        
        # Build constraint matrix from bitsets
        # We need to convert bitsets to a list of (i, j) conflicts for the sparse matrix
        # Or we can iterate through the bitsets to build the COO matrix directly
        
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            mask = conflict[i]
            # Iterate through set bits in mask
            # Only consider j > i to avoid duplicates, or just all j?
            # The constraint is x_i + x_j <= 1.
            # If we include both (i,j) and (j,i), it's redundant but valid.
            # To save memory, we can just generate unique pairs.
            # Since we built conflict symmetrically, we can just iterate i and check bits j > i.
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    rows.append(len(rows)) # Wait, we need row index for each constraint
                    # Actually, let's just count conflicts first or build lists properly.
                    pass
        
        # Let's rebuild the lists properly
        rows = []
        cols = []
        data = []
        constraint_idx = 0
        for i in range(n):
            mask = conflict[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    # Constraint: x_i + x_j <= 1
                    rows.extend([constraint_idx, constraint_idx])
                    cols.extend([i, j])
                    data.extend([1, 1])
                    constraint_idx += 1
        
        if constraint_idx > 0:
            A = sp.coo_matrix((data, (rows, cols)), shape=(constraint_idx, n)).tocsr()
            b_ub = np.ones(constraint_idx)
            constraints = [
                LinearConstraint(A, ub=b_ub)
            ]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        
        # Check time before solving
        if time.time() > deadline - 10:
            raise TimeoutError("Time limit approaching before MILP")
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:  # Optimal solution found
            # Extract solution
            x_sol = res.x
            selected_indices = np.where(x_sol > 0.5)[0]
            selected_values = [candidates[i] for i in selected_indices]
            
            # Verify validity
            valid = True
            sel_set = set(selected_values)
            for i in range(len(selected_values)):
                a = selected_values[i]
                for j in range(i + 1, len(selected_values)):
                    b = selected_values[j]
                    if lcm(a, b) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and selected_values:
                return selected_values
            else:
                # Fallback if verification fails
                pass
                
    except Exception as e:
        # Fallback to heuristic if scipy fails or is not available
        pass

    # 4. Fallback: Greedy + Local Search using Bitsets
    # This ensures we return *something* valid if MILP fails.
    
    # Greedy: Sort by weight descending (smallest x first)
    # To maximize sum(1/x), we want small x.
    # Simple greedy: pick smallest x that doesn't conflict with selected.
    
    # Order candidates by value ascending (which corresponds to weight descending)
    # candidates is already sorted ascending [2, 3, ..., 256]
    
    selected_bits = 0
    selected_indices = []
    
    for i in range(n):
        # Check if i conflicts with any selected
        # conflict[i] has bits set for all j that conflict with i
        # If (conflict[i] & selected_bits) == 0, then i is compatible
        if not (conflict[i] & selected_bits):
            selected_indices.append(i)
            selected_bits |= (1 << i)
    
    # Local Search: Try to swap or add
    # We maintain selected_bits and a list of selected indices for fast removal
    # For simplicity and speed, we'll do a few iterations of "remove one, add best"
    
    current_sum = sum(1.0 / candidates[i] for i in selected_indices)
    sel_set = set(selected_indices)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing one and adding one
        # To speed up, we only try removing a few candidates
        for i in list(sel_set):
            if time.time() > deadline - 5:
                break
            trial_bits = selected_bits & ~(1 << i)
            # Find best candidate to add
            best_gain = 0
            best_candidate = -1
            for j in range(n):
                if (trial_bits >> j) & 1:
                    continue
                # Check if j conflicts with trial_set
                if not (conflict[j] & trial_bits):
                    gain = 1.0 / candidates[j] - 1.0 / candidates[i]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_candidate = j
            if best_candidate != -1:
                # Perform swap
                selected_bits &= ~(1 << i)
                selected_bits |= (1 << best_candidate)
                sel_set.remove(i)
                sel_set.add(best_candidate)
                improved = True
                break # Restart loop with new set
                
    return [candidates[i] for i in sel_set]

# EVOLVE-BLOCK-END
