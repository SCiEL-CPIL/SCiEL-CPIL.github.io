import numpy as np
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Construct the target tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # 2. Try to find a rank-10 decomposition with integer entries in {-2, -1, 1, 2}
    # We model this as: Minimize sum(x) subject to T(x) = T_target
    # where x_i are binary variables indicating if the i-th "atom" (support pattern) is used.
    # Actually, simpler: We fix the support patterns to be "simple" (one non-zero per vector) 
    # or "Strassen-like". 
    # However, for 2x2x3, rank 10 is the target.
    # A known result for 2x2x3 is rank 10.
    # Let's try to construct it explicitly using a known algorithm or search.
    
    # Strategy: 
    # 1. Generate a set of candidate "simple" tensors (rank 1) that are likely parts of the decomposition.
    #    A simple tensor is u ⊗ v ⊗ w where u, v, w have sparse supports.
    # 2. Use an integer program to select a subset of these that sums to T_target.
    
    # Generate candidate rank-1 tensors
    # For 2x2x3, the indices are:
    # U: 2x2 (4 entries)
    # V: 2x3 (6 entries)
    # W: 3x2 (6 entries) -- Wait, W is p*n = 3*2 = 6 entries.
    # The problem states W[k*n+i, r]. So W is 6xR.
    
    # Let's define the set of possible "atoms".
    # An atom is defined by (i, j, k) where:
    # U has 1 at (i, j)
    # V has 1 at (j, k)
    # W has 1 at (k, i)
    # This is the schoolbook atom. There are 2*2*3 = 12 such atoms.
    # Rank 12 is schoolbook. We want rank 10.
    # This implies some atoms must be combined or replaced by non-simple atoms?
    # Or, the decomposition uses atoms where U, V, W have multiple non-zeros?
    
    # Actually, for small matrix multiplication, the minimal rank decompositions often use
    # "Strassen-like" atoms where the vectors have 2 or 3 non-zeros.
    
    # Let's try a direct search for a rank-10 solution using a randomized approach
    # with exact rational arithmetic, but guided by the structure.
    
    # Alternative: Use the fact that 2x2x3 can be embedded in 2x2x4 or related.
    # Or, use the known rank 10 decomposition if available in literature.
    # Since I don't have the explicit coefficients memorized, I will use a 
    # "Rounding + Local Search" strategy which is more robust than pure ILP on huge spaces.
    
    # 1. Find a float solution for Rank 10 using Differential Evolution.
    # 2. Round to nearest integer/half-integer.
    # 3. Correct the error using a sparse linear system solver.
    
    from scipy.optimize import differential_evolution
    import sympy as sp

    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def objective(params, target_rank):
        U_flat = params[:nm*target_rank]
        V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
        W_flat = params[nm*target_rank + mp*target_rank:]
        
        U = U_flat.reshape(nm, target_rank)
        V = V_flat.reshape(mp, target_rank)
        W = W_flat.reshape(pn, target_rank)
        
        T_prime = compute_tensor(U, V, W)
        diff = T_prime - T_target
        return np.sum(diff**2)

    best_U, best_V, best_W = None, None, None
    best_rank = nm * mp * pn # Schoolbook fallback
    
    # Fallback: Schoolbook (Rank 12)
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()

    # Try Rank 10
    target_rank = 10
    total_params = (nm + mp + pn) * target_rank
    bounds = [(-3, 3)] * total_params
    
    if time.time() < deadline - 30.0:
        try:
            # Use DE to find a close float solution
            res = differential_evolution(
                lambda p: objective(p, target_rank), 
                bounds, 
                maxiter=200, 
                tol=1e-9, 
                popsize=20,
                seed=42,
                polish=False
            )
            
            if res.fun < 1e-6:
                U_flat = res.x[:nm*target_rank]
                V_flat = res.x[nm*target_rank:nm*target_rank + mp*target_rank]
                W_flat = res.x[nm*target_rank + mp*target_rank:]
                
                U_num = U_flat.reshape(nm, target_rank)
                V_num = V_flat.reshape(mp, target_rank)
                W_num = W_flat.reshape(pn, target_rank)
                
                # Round to nearest integer (or half-integer if we suspect that)
                # For 2x2x3, integers are likely.
                U_int = np.round(U_num).astype(int)
                V_int = np.round(V_num).astype(int)
                W_int = np.round(W_num).astype(int)
                
                # Verify
                T_int = compute_tensor(U_int, V_int, W_int)
                if np.allclose(T_int, T_target, atol=1e-9):
                    best_U, best_V, best_W = U_int.tolist(), V_int.tolist(), W_int.tolist()
                    best_rank = target_rank
                else:
                    # Try half-integers
                    U_half = np.round(U_num * 2) / 2.0
                    V_half = np.round(V_num * 2) / 2.0
                    W_half = np.round(W_num * 2) / 2.0
                    
                    T_half = compute_tensor(U_half, V_half, W_half)
                    if np.allclose(T_half, T_target, atol=1e-9):
                        best_U, best_V, best_W = U_half.tolist(), V_half.tolist(), W_half.tolist()
                        best_rank = target_rank
        except Exception as e:
            pass

    # If rank 10 failed, try rank 11?
    # But rank 10 is the theoretical minimum for 2x2x3? 
    # Actually, 2x2x2 is 7. 2x2x3 is 10. 2x2x4 is 12?
    # If we found rank 10, we are done.
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
