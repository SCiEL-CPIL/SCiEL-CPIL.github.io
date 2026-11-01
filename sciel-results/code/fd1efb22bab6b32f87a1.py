"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback (rank 18)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Target ranks to try: 14 is known minimal, try 14, 15, 16
    for target_rank in [14, 15, 16]:
        if time.time() > deadline - 10:
            break
        
        # Allocate time for this rank
        time_budget = 40 if target_rank == 14 else 25
        rank_deadline = time.time() + time_budget
        
        result = _try_rank(n, m, p, target_rank, rank_deadline)
        if result is not None:
            return result
            
    return {"U": U_school, "V": V_school, "W": W_school}

def _try_rank(n, m, p, rank, deadline):
    """
    Try to find an exact rank decomposition using vectorized ALS and integer snapping.
    """
    R = rank
    T = np.ones((n, m, p))
    
    # Precompute index mappings for speed
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def compute_error(U, V, W):
        """Compute max absolute error using vectorized einsum."""
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.max(np.abs(result - T))
    
    def als_step(U, V, W):
        """One ALS iteration, vectorized where possible."""
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # Build A: p x R, A[k, t] = V[j*p+k, t] * W[k*n+i, t]
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                b = np.ones(p)
                ATA = A.T @ A + 1e-10 * np.eye(R)
                ATb = A.T @ b
                try:
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    U[i * m + j, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # Build B: n x R, B[i, t] = U[i*m+j, t] * W[k*n+i, t]
                B = np.zeros((n, R))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                b = np.ones(n)
                BTA = B.T @ B + 1e-10 * np.eye(R)
                BTb = B.T @ b
                try:
                    V[j * p + k, :] = np.linalg.solve(BTA, BTb)
                except np.linalg.LinAlgError:
                    V[j * p + k, :] = np.linalg.lstsq(BTA, BTb, rcond=None)[0]
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # Build C: m x R, C[j, t] = U[i*m+j, t] * V[j*p+k, t]
                C = np.zeros((m, R))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                b = np.ones(m)
                CTA = C.T @ C + 1e-10 * np.eye(R)
                CTb = C.T @ b
                try:
                    W[k * n + i, :] = np.linalg.solve(CTA, CTb)
                except np.linalg.LinAlgError:
                    W[k * n + i, :] = np.linalg.lstsq(CTA, CTb, rcond=None)[0]
    
    # Try multiple random seeds
    max_restarts = 50
    for seed in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed * 7919 + 42)
        
        # Initialize with random values
        U = rng.randn(n * m, R) * 0.5
        V = rng.randn(m * p, R) * 0.5
        W = rng.randn(p * n, R) * 0.5
        
        # ALS Iterations
        prev_err = np.inf
        for it in range(400):
            if time.time() > deadline:
                break
            
            als_step(U, V, W)
            err = compute_error(U, V, W)
            
            if err < 1e-10:
                break
            
            # If stuck, add perturbation
            if abs(err - prev_err) < 1e-12 and it > 20:
                U += 0.3 * rng.randn(n * m, R)
                V += 0.3 * rng.randn(m * p, R)
                W += 0.3 * rng.randn(p * n, R)
            
            prev_err = err
        
        if compute_error(U, V, W) < 1e-8:
            # Try to snap to small denominators
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500, 600, 800, 1000]:
                U_snap = np.round(U * denom) / denom
                V_snap = np.round(V * denom) / denom
                W_snap = np.round(W * denom) / denom
                
                if _verify_exact(n, m, p, R, U_snap, V_snap, W_snap):
                    return _to_list(U_snap, V_snap, W_snap, n, m, p, R)
            
            # If snapping failed but error is tiny, try high-precision polish
            # Do more ALS iterations with tighter tolerance
            for polish_it in range(200):
                if time.time() > deadline:
                    break
                als_step(U, V, W)
                err = compute_error(U, V, W)
                if err < 1e-12:
                    break
            
            # Try snapping again after polish
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500, 600, 800, 1000, 2000, 5000, 10000]:
                U_snap = np.round(U * denom) / denom
                V_snap = np.round(V * denom) / denom
                W_snap = np.round(W * denom) / denom
                
                if _verify_exact(n, m, p, R, U_snap, V_snap, W_snap):
                    return _to_list(U_snap, V_snap, W_snap, n, m, p, R)
    
    return None

def _verify_exact(n, m, p, R, U, V, W):
    """
    Verify if U, V, W provide an exact decomposition using Fraction arithmetic.
    U, V, W are numpy arrays.
    """
    def to_frac(val):
        # Convert float to Fraction exactly
        return Fraction(val).limit_denominator(1000000)
    
    U_frac = [[to_frac(U[i, t]) for t in range(R)] for i in range(n * m)]
    V_frac = [[to_frac(V[j, t]) for t in range(R)] for j in range(m * p)]
    W_frac = [[to_frac(W[k, t]) for t in range(R)] for k in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for t in range(R):
                    s += U_frac[i * m + j][t] * V_frac[j * p + k][t] * W_frac[k * n + i][t]
                if s != 1:
                    return False
    return True

def _to_list(U, V, W, n, m, p, R):
    """
    Convert numpy arrays to list of lists of integers/floats.
    """
    def arr_to_list(arr, rows):
        res = []
        for i in range(rows):
            row = []
            for t in range(R):
                val = arr[i, t]
                # If it's an integer value, return int, else float
                if abs(val - round(val)) < 1e-9:
                    row.append(int(round(val)))
                else:
                    # Check if it's a simple fraction
                    frac = Fraction(val).limit_denominator(1000000)
                    if frac.denominator == 1:
                        row.append(int(frac.numerator))
                    else:
                        row.append(float(val))
            res.append(row)
        return res
    
    return {
        "U": arr_to_list(U, n * m),
        "V": arr_to_list(V, m * p),
        "W": arr_to_list(W, p * n)
    }
# EVOLVE-BLOCK-END
