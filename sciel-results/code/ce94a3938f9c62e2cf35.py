"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _solve_given_U(V_init, W_init, U_fixed, n, m, p, rank):
    """
    Given U fixed, solve for V and W.
    Strategy: First solve for V given U and W_init, then solve for W given U and V.
    Iterate a few times.
    """
    V = V_init.copy()
    W = W_init.copy()
    
    for _ in range(5):
        # Solve for V: sum_r U_{u,r} V_{v,r} W_{w,r} = 1
        # For each v: sum_r V_{v,r} * (U_{u,r} * W_{w,r}) = 1 for all (u,w)
        # This is a linear system in V_{v,r}
        
        num_eq = n * p  # equations from (u,w) pairs
        num_unknowns = m * p * rank
        
        A = np.zeros((num_eq, num_unknowns))
        b = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                # For each v, the equation is sum_r V_{v,r} * U_{i*m+j, r} * W_{k*n+i, r} = 1
                # But wait, the equation is over all j for a fixed (i,k)?
                # No: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed (i,k), we have m equations (one for each j) in terms of V.
                # Actually, let's just set up the full system for V.
                
                # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Unknowns are V[j*p+k, r] for all j, k, r.
                # Coefficient of V[j*p+k, r] in equation (i,j,k) is U[i*m+j, r] * W[k*n+i, r]
                
                # This is getting complex. Let me just solve for V directly.
                eq_idx += 1
        
        # Actually, let's use a simpler approach: solve for V row by row.
        # For each j, V[j*p+k, r] for k=0..p-1, r=0..rank-1.
        # The equations involving V row j are: for all i, k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is n*p equations in p*rank unknowns (V[j*p+k, r] for k=0..p-1, r=0..rank-1).
        
        for j in range(m):
            num_eq_j = n * p
            num_unk_j = p * rank
            A_j = np.zeros((num_eq_j, num_unk_j))
            b_j = np.zeros(num_eq_j)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    b_j[eq_idx] = 1.0
                    for r in range(rank):
                        # Coefficient of V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
                        A_j[eq_idx, k * rank + r] = U_fixed[i*m+j, r] * W[k*n+i, r]
                    eq_idx += 1
            
            # Solve least squares
            x, _, _, _ = np.linalg.lstsq(A_j, b_j, rcond=None)
            V[j*p:(j+1)*p] = _snap_to_half_int(x.reshape(p, rank))
        
        # Now solve for W similarly
        for i in range(n):
            num_eq_i = m * p
            num_unk_i = p * rank
            A_i = np.zeros((num_eq_i, num_unk_i))
            b_i = np.zeros(num_eq_i)
            
            eq_idx = 0
            for j in range(m):
                for k in range(p):
                    b_i[eq_idx] = 1.0
                    for r in range(rank):
                        # Coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                        A_i[eq_idx, k * rank + r] = U_fixed[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
            
            # Solve least squares
            x, _, _, _ = np.linalg.lstsq(A_i, b_i, rcond=None)
            W[i::n] = _snap_to_half_int(x.reshape(p, rank))  # W is indexed as k*n+i
    
    return V, W

def _exhaustive_search(n, m, p, rank, deadline):
    """
    Exhaustively search over small integer values for U, then solve for V and W.
    """
    # U has shape (n*m, rank) = (6, rank)
    # We'll try values in {-1, 0, 1} for each entry.
    # That's 3^(6*rank) possibilities, which is too many for rank >= 50.
    # Instead, let's fix most of U and only vary a few columns.
    
    # Alternative: Use the fact that we can scale. Let's try a structured approach.
    # For each column r of U, try all combinations of {-1, 0, 1} for the 6 entries.
    # That's 3^6 = 729 possibilities per column. For rank=50, that's 729^50, way too many.
    
    # Better: Use random initialization but with deterministic seeding and more iterations.
    # Or: Use a genetic algorithm approach.
    
    # Let's try a different approach: Start with a known good decomposition (Strassen + schoolbook)
    # and try to reduce rank by finding dependencies.
    
    # For now, let's just do a more systematic random search with deterministic seeds.
    best_result = None
    
    for seed in range(1000):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize U with small integers
        U = np.random.choice([-1, 0, 1], (n*m, rank))
        
        # Ensure each column is non-zero
        for r in range(rank):
            if np.all(U[:, r] == 0):
                U[np.random.randint(n*m), r] = 1
        
        # Initialize V and W randomly
        V = np.random.choice([-1, 0, 1], (m*p, rank))
        W = np.random.choice([-1, 0, 1], (p*n, rank))
        
        # Solve for V and W given U
        V, W = _solve_given_U(V, W, U, n, m, p, rank)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    
    # Fallback
    fallback = _schoolbook(n, m, p)
    
    # Try exhaustive search for different ranks
    for rank in [55, 54, 53, 52, 51, 50]:
        if time.time() > deadline:
            break
        
        res = _exhaustive_search(n, m, p, rank, deadline)
        if res is not None:
            return res
    
    return fallback
# EVOLVE-BLOCK-END
