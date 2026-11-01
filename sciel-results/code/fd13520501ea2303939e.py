"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_kronecker_strassen(n, m, p, time_budget, seed=42):
    """
    Try structured and semi-structured approaches for 2x3 * 3x10.
    
    Key insight: For n=2, the two output rows share the same B matrix.
    We can try to construct solutions where U and V have block structure
    that allows W to be solved linearly with rational entries.
    
    Strategy: Use sparse structured U and V (with patterns like identity blocks,
    Hadamard-like patterns, or Kronecker products) then solve for W.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try candidate ranks from low to high
    candidate_ranks = list(range(20, n*m*p + 1, 2))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                    break
                    
            # Different structured initialization strategies
            strategy = attempt % 5
            
            if strategy == 0:
                # Sparse random with low density
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                # Each row of U/V gets 1-3 non-zero entries
                for row in range(n*m):
                    nnz = rng.randint(1, min(4, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    U[row, cols] = rng.choice([-1, 1], size=nnz)
                for row in range(m*p):
                    nnz = rng.randint(1, min(4, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    V[row, cols] = rng.choice([-1, 1], size=nnz)
                    
            elif strategy == 1:
                # Block diagonal-ish structure
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                # Assign blocks of columns to different (i,j) pairs
                for idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(m)]):
                    start = (idx * r) // (n*m)
                    end = ((idx + 1) * r) // (n*m)
                    if end > start:
                        U[i*m+j, start:end] = rng.choice([-1, 1], size=end-start)
                for idx, (j, k) in enumerate([(j, k) for j in range(m) for k in range(p)]):
                    start = (idx * r) // (m*p)
                    end = ((idx + 1) * r) // (m*p)
                    if end > start:
                        V[j*p+k, start:end] = rng.choice([-1, 1], size=end-start)
                        
            elif strategy == 2:
                # Random ±1 but with some zeros
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.4] = 0
                V[rng.random((m*p, r)) < 0.4] = 0
                
            elif strategy == 3:
                # Kronecker-like: use small base patterns
                # For n=2, m=3, p=10: try 2x3 and 3x10 sub-structures
                base_U = rng.randint(0, 2, size=(n*m, min(r, 12))) * 2 - 1
                base_V = rng.randint(0, 2, size=(m*p, min(r, 12))) * 2 - 1
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                U[:, :min(r, 12)] = base_U
                V[:, :min(r, 12)] = base_V
                # Fill remaining columns sparsely
                if r > 12:
                    U[:, 12:] = rng.randint(0, 2, size=(n*m, r-12)) * 2 - 1
                    V[:, 12:] = rng.randint(0, 2, size=(m*p, r-12)) * 2 - 1
                    U[:, 12:][rng.random((n*m, r-12)) < 0.5] = 0
                    V[:, 12:][rng.random((m*p, r-12)) < 0.5] = 0
                    
            else:
                # Fully random with moderate sparsity
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.25] = 0
                V[rng.random((m*p, r)) < 0.25] = 0
            
            # Solve for W linearly
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build linear system: for each j, sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    # Check if system is consistent
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Verify exact solution
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify full solution
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def _try_linear_program(n, m, p, time_budget, seed=42):
    """
    Systematic linear solve approach: fix U and V, solve for W.
    
    For each (i,k), we have m linear equations in r unknowns (W[k*n+i, :]):
        sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1,  for j = 0,...,m-1
    
    We need the m x r matrix A_{j,c} = U[i*m+j, c] * V[j*p+k, c] to have
    rank m and the system to be consistent.
    
    Key improvement: use structured U and V that are more likely to yield
    full-rank systems with rational solutions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try candidate ranks
    candidate_ranks = list(range(15, n*m*p + 1, 1))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 300
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            # Structured initialization with increasing complexity
            strat = attempt % 7
            
            if strat == 0:
                # Identity-like blocks in U
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for i in range(n):
                    for j in range(m):
                        col = (i * m + j) % r
                        U[i*m+j, col] = 1
                        # Spread some additional non-zeros
                        if r > n*m:
                            extra = rng.randint(0, r, size=min(2, r - n*m))
                            U[i*m+j, extra] = rng.choice([-1, 1], size=len(extra))
                for j in range(m):
                    for k in range(p):
                        col = (j * p + k) % r
                        V[j*p+k, col] = 1
                        if r > m*p:
                            extra = rng.randint(0, r, size=min(2, r - m*p))
                            V[j*p+k, extra] = rng.choice([-1, 1], size=len(extra))
                            
            elif strat == 1:
                # Sparse random, very low density
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for row in range(n*m):
                    nnz = rng.randint(1, min(3, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    U[row, cols] = rng.choice([-1, 1], size=nnz)
                for row in range(m*p):
                    nnz = rng.randint(1, min(3, r))
                    cols = rng.choice(r, size=nnz, replace=False)
                    V[row, cols] = rng.choice([-1, 1], size=nnz)
                    
            elif strat == 2:
                # Random ±1 with 50% zeros
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.5] = 0
                V[rng.random((m*p, r)) < 0.5] = 0
                
            elif strat == 3:
                # Kronecker structure: base 2x3 and 3x10
                # U has shape (6, r), V has shape (30, r)
                # Try to use a 2x3 pattern repeated
                base = rng.randint(0, 2, size=(n*m, min(r, 6))) * 2 - 1
                U = np.zeros((n*m, r))
                U[:, :min(r, 6)] = base
                if r > 6:
                    U[:, 6:] = rng.randint(0, 2, size=(n*m, r-6)) * 2 - 1
                    U[:, 6:][rng.random((n*m, r-6)) < 0.6] = 0
                base_v = rng.randint(0, 2, size=(m*p, min(r, 12))) * 2 - 1
                V = np.zeros((m*p, r))
                V[:, :min(r, 12)] = base_v
                if r > 12:
                    V[:, 12:] = rng.randint(0, 2, size=(m*p, r-12)) * 2 - 1
                    V[:, 12:][rng.random((m*p, r-12)) < 0.6] = 0
                    
            elif strat == 4:
                # Hadamard-like for small r
                if r <= 16:
                    # Use Walsh-Hadamard construction
                    def hadamard(k):
                        if k == 1:
                            return np.array([[1]])
                        H = hadamard(k//2)
                        return np.block([[H, H], [H, -H]])
                    r_h = 1
                    while r_h < r:
                        r_h *= 2
                    H = hadamard(r_h)
                    U = H[:n*m, :r]
                    V = H[:m*p, :r]
                else:
                    U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                    V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                    U[rng.random((n*m, r)) < 0.3] = 0
                    V[rng.random((m*p, r)) < 0.3] = 0
                    
            elif strat == 5:
                # Low-rank factorization: U = U1 @ U2, V = V1 @ V2
                mid = max(2, r // 3)
                U1 = rng.randint(0, 2, size=(n*m, mid)) * 2 - 1
                U2 = rng.randint(0, 2, size=(mid, r)) * 2 - 1
                U = U1 @ U2
                V1 = rng.randint(0, 2, size=(m*p, mid)) * 2 - 1
                V2 = rng.randint(0, 2, size=(mid, r)) * 2 - 1
                V = V1 @ V2
                # Sparsify
                U[np.abs(U) < 1] = 0
                V[np.abs(V) < 1] = 0
                
            else:
                # Pure random with varying sparsity
                sparsity = 0.2 + 0.3 * (attempt % 5) / 5
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < sparsity] = 0
                V[rng.random((m*p, r)) < sparsity] = 0
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try linear program approach with multiple seeds
    time_left = deadline - time.time()
    if time_left > 20:
        for seed in [12345, 42, 777, 99999, 31415]:
            if time.time() > deadline - 15:
                break
            tl = min(time_left, deadline - time.time() - 10)
            if tl < 15:
                break
            result = _try_linear_program(n, m, p, tl, seed=seed)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = r
                        break
    
    # Try structured/Kronecker approach
    time_left = deadline - time.time()
    if time_left > 15:
        for seed in [12345, 42, 777, 99999, 31415]:
            if time.time() > deadline - 15:
                break
            tl = min(time_left, deadline - time.time() - 10)
            if tl < 10:
                break
            result = _try_kronecker_strassen(n, m, p, tl, seed=seed)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best = result
                        best_rank = r
                        break
    
    return best
