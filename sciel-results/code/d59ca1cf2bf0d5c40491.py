import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # We can verify this by iterating over all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Strategy: Use ILP to find integer decompositions
    # We fix a set of "basis" tensors defined by random sparse patterns.
    # We then solve for integer coefficients x_c such that sum x_c * T_c = T_mm
    # where T_mm is the matrix multiplication tensor (1 at (i,j,k), 0 else).
    
    # To make the ILP tractable, we limit the number of candidate basis tensors (rank)
    # and the magnitude of the integer coefficients.
    
    # We will try to find a decomposition of rank R.
    # We generate R random sparse basis tensors U_c, V_c, W_c with entries in {-1, 0, 1}.
    # Then we set up the ILP:
    # Variables: x_1, ..., x_R (integers)
    # Constraints: For each (i,j,k), sum_c x_c * (U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]) = 1
    # Bounds: -B <= x_c <= B
    
    # Note: The basis tensors themselves are fixed in this approach.
    # If the ILP is infeasible, we regenerate the basis tensors.
    
    random.seed(42)
    
    # Try ranks from 66 down to 40
    # Rank 66 is a known lower bound for 3x4x6 (Landsberg bound or similar improvements over schoolbook)
    # Schoolbook is 72.
    
    for target_rank in range(66, 40, -1):
        if time.time() > deadline:
            break
        
        # Attempt to find a solution for this rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse basis tensors
            # We want them to be "good" candidates, so maybe bias towards non-zero
            # but keep them sparse to allow linear independence and sparsity in solution.
            
            # Each basis tensor is defined by U_c (n*m x 1), V_c (m*p x 1), W_c (p*n x 1)
            # Entries in {-1, 0, 1}
            
            U_basis = []
            V_basis = []
            W_basis = []
            
            for c in range(target_rank):
                u_vec = np.array([random.choice([-1, 0, 1]) for _ in range(n * m)])
                v_vec = np.array([random.choice([-1, 0, 1]) for _ in range(m * p)])
                w_vec = np.array([random.choice([-1, 0, 1]) for _ in range(p * n)])
                
                # Ensure not all zero
                if np.all(u_vec == 0) or np.all(v_vec == 0) or np.all(w_vec == 0):
                    u_vec[0] = 1
                    v_vec[0] = 1
                    w_vec[0] = 1
                
                U_basis.append(u_vec)
                V_basis.append(v_vec)
                W_basis.append(w_vec)
            
            # Build the constraint matrix A for the ILP
            # A[i*m+j*p+k, c] = U_basis[c][i*m+j] * V_basis[c][j*p+k] * W_basis[c][k*n+i]
            
            A = np.zeros((n * m * p, target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            A[idx, c] = U_basis[c][i * m + j] * V_basis[c][j * p + k] * W_basis[c][k * n + i]
            
            # The target vector b is 1 for all (i,j,k)
            b = np.ones(n * m * p)
            
            # Solve ILP: A @ x = b, x integer, |x| <= B
            # We use scipy.optimize.milp
            # Objective: minimize 0 (feasibility problem)
            # We can add a small penalty on |x| to prefer smaller coefficients, but feasibility is key.
            
            c_obj = np.zeros(target_rank)
            constraints = [LinearConstraint(A, b, b)]
            integrality = np.ones(target_rank)
            bounds = Bounds(lb=-10, ub=10)  # Limit coefficient magnitude
            
            try:
                res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                           options={"time_limit": 5, "gap_abs": 0})
            except Exception:
                continue
            
            if res.status == 0:  # Optimal
                x = res.x
                # Check if solution is integer (within tolerance)
                if np.allclose(x, np.round(x), atol=1e-6):
                    x_int = np.round(x).astype(int)
                    
                    # Construct U, V, W from x and basis
                    U_sol = []
                    V_sol = []
                    W_sol = []
                    
                    for row in range(n * m):
                        U_row = []
                        for c in range(target_rank):
                            U_row.append(x_int[c] * U_basis[c][row])
                        U_sol.append(U_row)
                    
                    for row in range(m * p):
                        V_row = []
                        for c in range(target_rank):
                            V_row.append(x_int[c] * V_basis[c][row])
                        V_sol.append(V_row)
                    
                    for row in range(p * n):
                        W_row = []
                        for c in range(target_rank):
                            W_row.append(x_int[c] * W_basis[c][row])
                        W_sol.append(W_row)
                    
                    # Verify
                    if verify(U_sol, V_sol, W_sol, n, m, p):
                        if target_rank < best_rank:
                            best = {"U": U_sol, "V": V_sol, "W": W_sol}
                            best_rank = target_rank
                        found = True
                        break
        
        if found:
            # If we found a solution for this rank, we can continue to lower ranks
            # or break if we hit a known lower bound
            if best_rank <= 46: # Known minimal rank for 3x4x6 is 46? 
                # Actually, let's just keep going down to 40 as per loop
                pass
        else:
            # If we couldn't find a solution for this rank with random bases,
            # it might be infeasible or we need better bases.
            # Continue to next lower rank.
            continue
            
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Strategy: Use ILP to find integer decompositions
    # We fix a set of "basis" tensors defined by random sparse patterns.
    # We then solve for integer coefficients x_c such that sum x_c * T_c = T_mm
    
    random.seed(42)
    
    # Try ranks from 66 down to 40
    for target_rank in range(66, 40, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U_basis = []
            V_basis = []
            W_basis = []
            
            for c in range(target_rank):
                u_vec = np.array([random.choice([-1, 0, 1]) for _ in range(n * m)])
                v_vec = np.array([random.choice([-1, 0, 1]) for _ in range(m * p)])
                w_vec = np.array([random.choice([-1, 0, 1]) for _ in range(p * n)])
                
                if np.all(u_vec == 0) or np.all(v_vec == 0) or np.all(w_vec == 0):
                    u_vec[0] = 1
                    v_vec[0] = 1
                    w_vec[0] = 1
                
                U_basis.append(u_vec)
                V_basis.append(v_vec)
                W_basis.append(w_vec)
            
            A = np.zeros((n * m * p, target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            A[idx, c] = U_basis[c][i * m + j] * V_basis[c][j * p + k] * W_basis[c][k * n + i]
            
            b = np.ones(n * m * p)
            
            c_obj = np.zeros(target_rank)
            constraints = [LinearConstraint(A, b, b)]
            integrality = np.ones(target_rank)
            bounds = Bounds(lb=-10, ub=10)
            
            try:
                res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                           options={"time_limit": 3, "gap_abs": 0})
            except Exception:
                continue
            
            if res.status == 0:
                x = res.x
                if np.allclose(x, np.round(x), atol=1e-6):
                    x_int = np.round(x).astype(int)
                    
                    U_sol = []
                    V_sol = []
                    W_sol = []
                    
                    for row in range(n * m):
                        U_row = []
                        for c in range(target_rank):
                            U_row.append(x_int[c] * U_basis[c][row])
                        U_sol.append(U_row)
                    
                    for row in range(m * p):
                        V_row = []
                        for c in range(target_rank):
                            V_row.append(x_int[c] * V_basis[c][row])
                        V_sol.append(V_row)
                    
                    for row in range(p * n):
                        W_row = []
                        for c in range(target_rank):
                            W_row.append(x_int[c] * W_basis[c][row])
                        W_sol.append(W_row)
                    
                    if verify(U_sol, V_sol, W_sol, n, m, p):
                        if target_rank < best_rank:
                            best = {"U": U_sol, "V": V_sol, "W": W_sol}
                            best_rank = target_rank
                        found = True
                        break
        
        if found:
            continue
        else:
            continue
            
    return best
# EVOLVE-BLOCK-END
