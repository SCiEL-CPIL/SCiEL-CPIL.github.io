"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition by checking each (i,j,k) equation directly."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check that for each (i,j,k), the sum equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check that for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), sum is 0
    valid_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_entries.add((i*m+j, j*p+k, k*n+i))
    
    # Sample some invalid entries to check
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        c = rng.randint(0, p*n)
        if (a, b, c) in valid_entries:
            continue
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    
    return True

def _schoolbook(n, m, p):
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

def _try_structured_decomposition(n, m, p, rank, deadline):
    """Try to find a decomposition using structured random search."""
    rng = np.random.RandomState(12345)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize with sparse patterns
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Each rank-1 term should "cover" some (i,j,k) triples
        # Try to assign each term to help satisfy certain equations
        for r in range(rank):
            # Choose which (i,j,k) this term primarily helps with
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            
            # Set values to help satisfy this equation
            sign = 1.0 if rng.random() > 0.5 else -1.0
            U[a, r] = sign
            V[b, r] = sign
            W[c, r] = 1.0
            
            # Also add some cross terms to help cancel unwanted entries
            if rng.random() > 0.7:
                i2 = rng.randint(0, n)
                j2 = rng.randint(0, m)
                k2 = rng.randint(0, p)
                a2 = i2 * m + j2
                b2 = j2 * p + k2
                c2 = k2 * n + i2
                if (a2, b2, c2) != (a, b, c):
                    U[a2, r] += rng.choice([-1.0, 1.0]) * 0.5
                    V[b2, r] += rng.choice([-1.0, 1.0]) * 0.5
                    W[c2, r] += rng.choice([-1.0, 1.0]) * 0.5
        
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _als_decomposition(n, m, p, target_rank, deadline):
    """Use ALS to find a decomposition of given rank."""
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_err = float('inf')
    best_result = None
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(42 + restart)
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        for it in range(50):
            if time.time() > deadline:
                break
            
            # ALS step: solve for each factor
            # Solve for W: fix U, V
            W_new = np.zeros_like(W)
            for c in range(p*n):
                # T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
                M = np.zeros((n*m*m*p, target_rank))
                for r in range(target_rank):
                    M[:, r] = np.outer(U[:, r], V[:, r]).ravel()
                rhs = T[:, :, c].ravel()
                sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                W_new[c, :] = sol
            W = W_new
            
            # Solve for U: fix V, W
            U_new = np.zeros_like(U)
            for a in range(n*m):
                M = np.zeros((m*p*p*n, target_rank))
                for r in range(target_rank):
                    M[:, r] = np.outer(V[:, r], W[:, r]).ravel()
                rhs = T[a, :, :].ravel()
                sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                U_new[a, :] = sol
            U = U_new
            
            # Solve for V: fix U, W
            V_new = np.zeros_like(V)
            for b in range(m*p):
                M = np.zeros((n*m*p*n, target_rank))
                for r in range(target_rank):
                    M[:, r] = np.outer(U[:, r], W[:, r]).ravel()
                rhs = T[:, b, :].ravel()
                sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                V_new[b, :] = sol
            V = V_new
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        s = 0.0
                        for r in range(target_rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                if err < 1e-12:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
        
        if best_err < 1e-12:
            return best_result
    
    return None

def _to_half_integer(mat):
    """Convert to half-integers if close."""
    mat = np.array(mat, dtype=float)
    rounded = np.round(mat * 2) / 2
    if np.allclose(mat, rounded, atol=1e-6):
        mat = rounded
    return mat.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook as safe baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search for lower ranks
    candidate_ranks = [53, 52, 51, 50, 49, 48, 46, 44, 42, 40]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try structured decomposition
        result = _try_structured_decomposition(n, m, p, rank, deadline)
        if result is not None:
            if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
                continue
        
        # Try ALS if we have time
        if time.time() < deadline - 30:
            result = _als_decomposition(n, m, p, rank, deadline)
            if result is not None:
                if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
    
    # Convert to half-integers if possible
    for key in ["U", "V", "W"]:
        best[key] = _to_half_integer(best[key])
    
    return best
# EVOLVE-BLOCK-END
