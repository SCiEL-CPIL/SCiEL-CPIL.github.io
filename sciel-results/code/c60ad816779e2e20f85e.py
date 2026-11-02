"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We focus on weight-4 vectors (4 non-zero coordinates, values +/-1).
    # Total number of weight-4 vectors in R^11 is C(11,4) * 2^4 = 330 * 16 = 5280.
    
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute Gram matrix G where G[i,j] = candidates[i] @ candidates[j].
    G = candidates @ candidates.T
    
    # Compatibility condition: G[i,j] <= 2 for i != j.
    # Conflict if G[i,j] > 2.
    # We want the Maximum Independent Set in the conflict graph.
    
    # Build the conflict matrix (boolean)
    conflict_mat = G > 2
    np.fill_diagonal(conflict_mat, False) # Self is not a conflict for the graph definition, but we exclude i=j in constraints anyway.
    
    # Formulate as MILP:
    # Maximize sum(x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i,j) where conflict_mat[i,j] is True
    #   x_i in {0, 1}
    
    # Number of conflicts can be large. 
    # Let's estimate. 
    # If the graph is dense, this is huge. 
    # If the graph is sparse, it's manageable.
    # For weight-4 vectors, how many conflicts?
    # Two weight-4 vectors conflict if their dot product > 2.
    # Dot product is sum of products.
    # Max dot is 4. 
    # Dot 3: overlap 3 same, 1 diff? No, weights are 4.
    # If they share 4 coords: dot = 4 - 2*k where k is number of sign flips.
    # If they share 3 coords: dot = 3 - 2*k. Max 3.
    # If they share 2 coords: dot = 2 - 2*k. Max 2.
    # If they share 1 coord: dot = 1 - 2*k. Max 1.
    # If they share 0 coords: dot = 0.
    # So conflicts exist only if they share 3 or 4 coordinates.
    # 
    # Case 1: Share 4 coordinates (same support).
    # For a fixed support (4 indices), there are 16 sign vectors.
    # Two sign vectors conflict if dot > 2.
    # Dot = 4 - 2*(Hamming distance).
    # Dot > 2 => 4 - 2d > 2 => 2d < 2 => d < 1 => d=0.
    # So only the vector itself conflicts with itself? 
    # Wait. 
    # If d=0, dot=4. Conflict.
    # If d=1, dot=2. Not conflict (<=2).
    # If d=2, dot=0. Not conflict.
    # If d=3, dot=-2. Not conflict.
    # If d=4, dot=-4. Not conflict.
    # So within a support, only the vector conflicts with itself.
    # 
    # Case 2: Share 3 coordinates.
    # Let support A and B share 3 indices.
    # There are C(11,3) * C(8,1) pairs of supports? 
    # Let's count conflicts.
    # For a fixed vector v, how many vectors u conflict with v?
    # u must share 3 or 4 indices with v.
    # 
    # Subcase 2a: u shares 4 indices with v.
    # Only v itself (since any other sign pattern has distance >=1, dot <=2).
    # So 0 other vectors in same support conflict.
    # 
    # Subcase 2b: u shares 3 indices with v.
    # Let v have support S (size 4).
    # u has support T (size 4) such that |S \cap T| = 3.
    # There are 4 choices for which index in S is not in T.
    # There are 11-3=8 choices for the new index in T \ S.
    # So for each of the 4 "missing" indices in S, there are 8 possible new indices.
    # Total supports T sharing 3 with S: 4 * 8 = 32.
    # For each such support T, how many sign vectors conflict?
    # Let S = {1,2,3,4}, T = {1,2,3,5}.
    # v has signs s1,s2,s3,s4.
    # u has signs t1,t2,t3,t5.
    # Dot(v,u) = s1*t1 + s2*t2 + s3*t3. (s4 and t5 don't overlap)
    # We need Dot > 2.
    # Max dot is 3 (if s1=t1, s2=t2, s3=t3).
    # Min dot is -3.
    # Dot = 3 - 2*k where k is number of sign mismatches in the 3 shared coords.
    # Dot > 2 => 3 - 2k > 2 => 2k < 1 => k=0.
    # So only the sign pattern where t1=s1, t2=s2, t3=s3 conflicts.
    # There is exactly 1 such sign pattern for the 3 shared coords.
    # The 4th coord (t5) can be +1 or -1.
    # So for each support T, there are 2 vectors u that conflict with v.
    # Total conflicts from shared-3 supports: 32 supports * 2 vectors/support = 64.
    # 
    # So each vector v conflicts with:
    # - 0 other vectors in its own support.
    # - 64 vectors in supports sharing 3 indices.
    # 
    # Total conflicts per node ~ 64.
    # Total edges in conflict graph ~ N * 64 / 2 = 5280 * 32 = 168,960.
    # This is a VERY sparse graph!
    # 
    # A MILP with ~170k constraints and 5280 binary variables is very solvable.
    
    # Extract indices of conflicts
    # conflict_mat is (5280, 5280) boolean.
    # We need rows i, cols j such that conflict_mat[i,j] is True and i < j.
    
    # Using np.nonzero on the upper triangle
    # To save memory/time, we can just get all True indices and filter i<j.
    # 5280^2 = 28M. np.nonzero is fast.
    
    true_indices = np.nonzero(conflict_mat)
    i_idx = true_indices[0]
    j_idx = true_indices[1]
    
    # Filter i < j
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    n_constraints = len(i_idx)
    
    # Build the constraint matrix A_ub for x_i + x_j <= 1
    # Shape (n_constraints, n_cand)
    # This is a sparse matrix. scipy.optimize.milp accepts sparse matrices.
    
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((n_constraints, n_cand), dtype=np.float64)
    for k in range(n_constraints):
        A[k, i_idx[k]] = 1.0
        A[k, j_idx[k]] = 1.0
        
    A = A.tocsr()
    
    # Objective: maximize sum(x_i)
    c = -np.ones(n_cand) # Minimize -sum(x_i)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality
    integrality = np.ones(n_cand)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, ub=np.ones(n_constraints))
    
    # Time limit for solver
    # We have ~145s. Let's give it 140s.
    # scipy.optimize.milp uses HiGHS by default.
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 140, "mip_rel_gap": 0.0})
        
        if res.success or res.status == 2: # 2 is "Optimal solution found" or similar? 
            # Actually, check res.x
            if res.x is not None:
                # res.x contains the solution.
                # We need to round to integers? 
                # HiGHS should return integers for binary variables if integrality is set.
                # But sometimes it returns 0.999999.
                x_sol = np.round(res.x).astype(np.int64)
                
                # Verify solution validity
                selected_indices = np.where(x_sol == 1)[0]
                
                # Check if valid
                # For all i, j in selected_indices, G[i,j] <= 2
                if len(selected_indices) > 0:
                    # Check pairwise
                    # This is O(K^2). K ~ 500-1000. 1M ops. Fast.
                    valid = True
                    for i in range(len(selected_indices)):
                        for j in range(i+1, len(selected_indices)):
                            if G[selected_indices[i], selected_indices[j]] > 2:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        final_vecs = candidates[selected_indices]
                        return final_vecs
                    else:
                        # Fallback to greedy if MILP returned invalid (shouldn't happen)
                        pass
                        
    except Exception as e:
        print(f"MILP failed: {e}")
        
    # Fallback: Greedy if MILP fails or times out without a feasible solution
    # Simple greedy on the sparse conflict graph
    # We can use the conflict structure to do a fast greedy.
    # Or just return a small valid set to ensure score > 0.
    
    # Let's do a quick greedy just in case.
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_cand)
    selected = []
    for i in perm:
        if not selected:
            selected.append(i)
        else:
            # Check against selected
            # We can use the precomputed conflict info? 
            # For speed, just use G
            if np.all(G[i, selected] <= 2):
                selected.append(i)
                
    if selected:
        return candidates[selected]
    else:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)

# EVOLVE-BLOCK-END
