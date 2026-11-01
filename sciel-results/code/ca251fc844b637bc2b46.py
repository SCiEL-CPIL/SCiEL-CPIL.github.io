"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The tensor T has indices (i, j, k) with i in [0, n), j in [0, m), k in [0, p).
    T[i,j,k] = 1 for all valid i,j,k.
    The decomposition is T = sum_r U_r * V_r * W_r where:
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    The entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    We use a structural approach: 
    1. Start with schoolbook (rank 48)
    2. Use iterative refinement / local search to find lower rank decompositions
    3. For each rank target, initialize with structured patterns (not pure random)
       and use coordinate descent / gradient-free local search to satisfy constraints
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = r_sb
    
    def verify_decomposition(U, V, W, r, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def compute_residuals(U, V, W, r):
        """Compute residuals for all (i,j,k) entries."""
        residuals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    residuals[(i,j,k)] = val - 1
        return residuals
    
    def max_residual(residuals):
        return max(abs(v) for v in residuals.values())
    
    # Local search: perturb one entry at a time to reduce max residual
    def local_search(U, V, W, r, max_iter=500, tol=1e-9):
        """
        Coordinate descent style local search.
        Try to reduce the maximum residual by adjusting individual entries.
        """
        residuals = compute_residuals(U, W, V, r)
        best_max_res = max_residual(residuals)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if best_max_res < tol:
                break
            
            # Find the worst residual
            worst_key = max(residuals, key=lambda k: abs(residuals[k]))
            i, j, k = worst_key
            current_val = residuals[worst_key]
            
            # Try adjusting U[i*m+j, r'] for each r'
            improved = False
            for rr in range(r):
                old_u = U[i*m+j][rr]
                # Try small perturbations
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25, 0.1, -0.1]:
                    U[i*m+j][rr] = old_u + delta
                    res = compute_residuals(U, V, W, r)
                    new_max = max_residual(res)
                    if new_max < best_max_res:
                        best_max_res = new_max
                        residuals = res
                        improved = True
                        break
                    else:
                        U[i*m+j][rr] = old_u
            
            if improved:
                continue
            
            # Try adjusting V[j*p+k, r']
            for rr in range(r):
                old_v = V[j*p+k][rr]
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25, 0.1, -0.1]:
                    V[j*p+k][rr] = old_v + delta
                    res = compute_residuals(U, V, W, r)
                    new_max = max_residual(res)
                    if new_max < best_max_res:
                        best_max_res = new_max
                        residuals = res
                        improved = True
                        break
                    else:
                        V[j*p+k][rr] = old_v
            
            if improved:
                continue
            
            # Try adjusting W[k*n+i, r']
            for rr in range(r):
                old_w = W[k*n+i][rr]
                for delta in [0.5, -0.5, 1, -1, 0.25, -0.25, 0.1, -0.1]:
                    W[k*n+i][rr] = old_w + delta
                    res = compute_residuals(U, V, W, r)
                    new_max = max_residual(res)
                    if new_max < best_max_res:
                        best_max_res = new_max
                        residuals = res
                        improved = True
                        break
                    else:
                        W[k*n+i][rr] = old_w
            
            if improved:
                continue
            
            # If no improvement, try random perturbation to escape local minimum
            if random.random() < 0.1:
                # Randomly perturb a few entries
                for _ in range(3):
                    choice = random.choice(['U', 'V', 'W'])
                    if choice == 'U':
                        idx = random.randint(0, n*m-1)
                        rr = random.randint(0, r-1)
                        U[idx][rr] += random.choice([0.5, -0.5, 1, -1])
                    elif choice == 'V':
                        idx = random.randint(0, m*p-1)
                        rr = random.randint(0, r-1)
                        V[idx][rr] += random.choice([0.5, -0.5, 1, -1])
                    else:
                        idx = random.randint(0, p*n-1)
                        rr = random.randint(0, r-1)
                        W[idx][rr] += random.choice([0.5, -0.5, 1, -1])
                residuals = compute_residuals(U, V, W, r)
                best_max_res = max_residual(residuals)
        
        return best_max_res < tol, U, V, W
    
    # Structured initialization: use patterns that partially satisfy constraints
    def structured_init(target_rank):
        """
        Create a structured initialization that has some built-in symmetry.
        For 2x4x6, we can exploit the small n=2 dimension.
        """
        U = [[0] * target_rank for _ in range(n*m)]
        V = [[0] * target_rank for _ in range(m*p)]
        W = [[0] * target_rank for _ in range(p*n)]
        
        # Initialize with random values from a small set
        vals = [-1, 0, 1, 0.5, -0.5]
        
        # For each column r, assign random values
        for r_idx in range(target_rank):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r_idx] = random.choice(vals)
            for j in range(m):
                for k in range(p):
                    V[j*p+k][r_idx] = random.choice(vals)
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r_idx] = random.choice(vals)
        
        return U, V, W
    
    # Try to find lower rank solutions
    # Use local search with structured initialization
    for target_rank in range(47, 25, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple structured initializations
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            U, V, W = structured_init(target_rank)
            success, U, V, W = local_search(U, V, W, target_rank, max_iter=200)
            
            if success:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
        else:
            # If we can't find a solution at this rank, try the next lower rank
            # But don't break immediately - maybe a different rank works
            pass
    
    # Final verification
    if best_rank < r_sb:
        if verify_decomposition(best_result["U"], best_result["V"], best_result["W"], best_rank):
            return best_result
    
    return best_result
# EVOLVE-BLOCK-END
