"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    """Verify that U, V, W decompose T exactly (within tol)."""
    # T approx sum_c U[:,c] * V[:,c] * W[:,c]
    # Check via tensor reconstruction
    rank = U.shape[1]
    # Use einsum for efficiency
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(recon, T, atol=tol, rtol=0)

def _verify_exact_fraction(U, V, W, T):
    """Exact verification using fractions."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build T as fractions
    T_frac = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_frac[(i * m + j, j * p + k, k * n + i)] = Fraction(1, 1)
    
    # Check each entry
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = Fraction(0, 1)
                for c in range(rank):
                    s += Fraction(U[i][c]) * Fraction(V[j][c]) * Fraction(W[k][c])
                expected = T_frac.get((i, j, k), Fraction(0, 1))
                if s != expected:
                    return False
    return True

def _rank_min_bound(n, m, p):
    """Lower bound on tensor rank for matrix multiplication."""
    # Trivial lower bound: max(n*m, m*p, p*n)
    return max(n * m, m * p, p * n)

def _try_random_decomposition(n, m, p, max_rank, max_iters=5000, seed=42):
    """Try to find a rank decomposition by random search + verification."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    for attempt in range(max_iters):
        rank = _rank_min_bound(n, m, p)
        # Try random integer/half-integer matrices
        scale = 2  # entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1, 1.5, 2}
        U = rng.randint(-scale, scale + 1, size=(n * m, rank))
        V = rng.randint(-scale, scale + 1, size=(m * p, rank))
        W = rng.randint(-scale, scale + 1, size=(p * n, rank))
        
        # Quick check
        recon = np.einsum('ac,bc,dc->abd', U.astype(np.float64), 
                          V.astype(np.float64), W.astype(np.float64))
        if np.allclose(recon, T, atol=1e-10, rtol=0):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known fast matrix multiplication ideas.
    For 2x4 * 4x8, we can try to exploit the structure.
    
    Key insight: A (2x4) * B (4x8) = C (2x8)
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try to use the fact that n=2 is small.
    
    Let's try a decomposition based on splitting the middle dimension.
    """
    # For n=2, m=4, p=8:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # Try rank = m*p = 32 (same as schoolbook, no gain)
    # Try rank = n*m*p / 2 = 32 (no gain either)
    
    # Let's try a more clever approach.
    # The tensor rank of 2x4x4x8 matrix multiplication...
    
    # Actually, let's just try to find the minimum rank by checking
    # if rank < n*m*p is possible.
    
    # Lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Schoolbook gives 64.
    
    # Can we do better than 64?
    # The tensor rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = 32.
    # It's also at least n*m*p / max(n,m,p) in some cases... not sure.
    
    # Let's try rank 48, 40, 36, 32 and see if we can find a decomposition.
    
    return None

def _search_rank(n, m, p, start_rank, end_rank, time_budget=60):
    """Search for a decomposition with rank in [start_rank, end_rank]."""
    T = _build_tensor(n, m, p)
    deadline = time.time() + time_budget
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts at this rank
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # Random integer entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))
            
            recon = np.einsum('ac,bc,dc->abd', U.astype(np.float64),
                              V.astype(np.float64), W.astype(np.float64))
            if np.allclose(recon, T, atol=1e-10, rtol=0):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 64
    lower_bound = max(n * m, m * p, p * n)  # 32
    
    # First, return schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try to find a better decomposition
    # Search from lower_bound up to schoolbook_rank - 1
    T = _build_tensor(n, m, p)
    
    # Try structured approaches first
    # For this specific case, let's try a few known patterns
    
    # Attempt 1: Try rank = 48 with random search
    for target_rank in [48, 44, 40, 36, 32]:
        if time.time() > deadline - 10:
            break
        
        # Try random integer search
        found = False
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randint(-1, 2, size=(n * m, target_rank))
            V = np.random.randint(-1, 2, size=(m * p, target_rank))
            W = np.random.randint(-1, 2, size=(p * n, target_rank))
            
            recon = np.einsum('ac,bc,dc->abd', U.astype(np.float64),
                              V.astype(np.float64), W.astype(np.float64))
            if np.allclose(recon, T, atol=1e-10, rtol=0):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result

# EVOLVE-BLOCK-END
