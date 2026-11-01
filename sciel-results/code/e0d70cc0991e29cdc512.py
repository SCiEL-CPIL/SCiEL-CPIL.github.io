"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a low-rank exact decomposition
    # The (2,3,3) tensor should have rank 12
    for target_rank in [12]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to find an exact rank decomposition."""
    if rank == 12:
        result = _construct_rank12(n, m, p)
        if result is not None:
            return result
    
    # For other ranks, use schoolbook if rank >= n*m*p
    if rank >= n * m * p:
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank12(n, m, p):
    """Try to construct a rank-12 decomposition for (2,3,3) using ALS."""
    import numpy as np
    import time
    from fractions import Fraction
    
    start = time.time()
    deadline = start + 130
    R = 12
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # For all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # ALS: alternate solving linear systems
    # For fixed V,W: U[i*m+j, :] is determined by linear equations
    # For each (i,j), the equation involves W[k*n+i, :] for all k, and V[j*p+k, :] for all k
    # Actually, let's think about this more carefully.
    
    # The constraint is: for all (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # 
    # For fixed V and W, for each (i,j), we have p equations (one for each k):
    #   sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1 for k=0..p-1
    # This is a system of p linear equations in R unknowns (the entries of U[i*m+j,:])
    # But p=3 < R=12, so we have underdetermined systems.
    # 
    # Better approach: use a random initialization and do ALS where at each step
    # we minimize the residual. Since the system is underdetermined per row,
    # we can use the pseudoinverse or add regularization.
    
    # Actually, let's use a different ALS formulation:
    # For fixed V and W, minimize sum_{i,j,k} (sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] - 1)^2
    # Taking derivative w.r.t. U[i*m+j,t]:
    #   sum_k (sum_t' U[i*m+j,t']*V[j*p+k,t']*W[k*n+i,t'] - 1) * V[j*p+k,t] * W[k*n+i,t] = 0
    # This gives: sum_k (U[i*m+j,:] · (V[j*p+k,:] * W[k*n+i,:])) * V[j*p+k,t] * W[k*n+i,t] = sum_k V[j*p+k,t]*W[k*n+i,t]
    # 
    # Let's define for each (i,j): A[i,j] is a p x R matrix where A[i,j][k,t] = V[j*p+k,t] * W[k*n+i,t]
    # Then the normal equations are: A[i,j]^T A[i,j] * U[i*m+j,:] = A[i,j]^T * ones(p)
    
    np.random.seed(42)
    
    # Try multiple random initializations
    for seed in range(20):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        U = np.random.randn(n*m, R) * 0.5 + 0.5
        V = np.random.randn(m*p, R) * 0.5 + 0.5
        W = np.random.randn(p*n, R) * 0.5 + 0.5
        
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A: p x R matrix
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A^T A U_row = A^T ones
                    ATA = A.T @ A
                    ATb = A.T @ np.ones(p)
                    # Add small regularization for stability
                    ATA += 1e-8 * np.eye(R)
                    try:
                        U[i*m+j, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        U[i*m+j, :] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Build B: n x R matrix where B[i,t] = U[i*m+j,t] * W[k*n+i,t]
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    BTA = B.T @ B
                    BTb = B.T @ np.ones(n)
                    BTA += 1e-8 * np.eye(R)
                    try:
                        V[j*p+k, :] = np.linalg.solve(BTA, BTb)
                    except np.linalg.LinAlgError:
                        V[j*p+k, :] = np.linalg.lstsq(BTA, BTb, rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build C: m x R matrix where C[j,t] = U[i*m+j,t] * V[j*p+k,t]
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    CTA = C.T @ C
                    CTb = C.T @ np.ones(m)
                    CTA += 1e-8 * np.eye(R)
                    try:
                        W[k*n+i, :] = np.linalg.solve(CTA, CTb)
                    except np.linalg.LinAlgError:
                        W[k*n+i, :] = np.linalg.lstsq(CTA, CTb, rcond=None)[0]
            
            # Check convergence
            max_res = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_res = max(max_res, abs(s - 1))
            
            if max_res < 1e-10:
                break
        
        # Check if we got close
        if max_res < 1e-8:
            # Try to round to rationals
            def to_fraction_array(arr, max_denom=1000):
                frac_arr = []
                for val in arr.flatten():
                    best_frac = Fraction(val).limit_denominator(max_denom)
                    frac_arr.append(best_frac)
                return frac_arr
            
            U_frac = to_fraction_array(U)
            V_frac = to_fraction_array(V)
            W_frac = to_fraction_array(W)
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for t in range(R):
                            s += U_frac[i*m*R + j*R + t] * V_frac[j*p*R + k*R + t] * W_frac[k*n*R + i*R + t]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                U_list = [[int(f) if f.denominator == 1 else float(f) for f in U_frac[i*R:(i+1)*R]] for i in range(n*m)]
                V_list = [[int(f) if f.denominator == 1 else float(f) for f in V_frac[j*R:(j+1)*R]] for j in range(m*p)]
                W_list = [[int(f) if f.denominator == 1 else float(f) for f in W_frac[k*R:(k+1)*R]] for k in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
