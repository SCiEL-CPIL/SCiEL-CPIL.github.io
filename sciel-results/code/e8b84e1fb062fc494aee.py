"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32


def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True


def verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using the structure. Only check necessary conditions."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check the 32 required entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # For the zero entries, we can check a subset for speed, but for exactness we need all.
    # Given the small sizes (4*16*16=1024), full check is fine.
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True


def make_half_int_mat(rows, cols, rng):
    """Generate a random matrix with entries in {-1, -1/2, 0, 1/2, 1}."""
    vals = [0, 0, 0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    return [[rng.choice(vals) for _ in range(cols)] for _ in range(rows)]


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W


def _build_tensor():
    """Build the full 4x16x16 tensor T as a numpy array."""
    T = np.zeros((NM, MP, PN), dtype=np.int32)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _tensor_residual(U, V, W, T):
    """Compute the residual tensor: T - sum_r U[:,r] V[:,r] W[:,r].
    U, V, W are lists of lists of Fractions. T is a numpy int array.
    Returns a numpy array of Fraction residuals, or a float approximation if too slow.
    """
    rank = len(U[0])
    # Convert to numpy arrays of floats for speed, then check exactness separately
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    
    # Compute sum_r U[:,r] V[:,r] W[:,r] via einsum
    # Shape: (NM, MP, PN)
    recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    residual = T - recon
    return residual


def _exact_residual_check(U, V, W):
    """Check if the decomposition is exact using Fraction arithmetic.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    # Check the 32 required entries
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # Check a sample of zero entries
    valid = set()
    for i in range(N):
        for j in range(M):
            for k in range(P):
                valid.add((i*M+j, j*P+k, k*N+i))
    
    # Check all zero entries for correctness
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def _local_search(U, V, W, T, rng, deadline, max_iters=200):
    """Local search: try flipping individual entries to reduce the residual norm.
    U, V, W are lists of lists of Fractions.
    Returns improved (U, V, W) or None if no improvement found.
    """
    best_residual = None
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Compute initial residual
    res = _tensor_residual(best_U, best_V, best_W, T)
    best_norm = float(np.sum(res**2))
    
    if best_norm < 1e-10:
        return best_U, best_V, best_W
    
    vals = [Fraction(-1), Fraction(-1,2), Fraction(0), Fraction(1,2), Fraction(1)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try flipping entries in U
        for a in range(NM):
            for r in range(len(best_U[0])):
                old_val = best_U[a][r]
                for new_val in vals:
                    if new_val == old_val:
                        continue
                    best_U[a][r] = new_val
                    res = _tensor_residual(best_U, best_V, best_W, T)
                    norm = float(np.sum(res**2))
                    if norm < best_norm - 1e-10:
                        best_norm = norm
                        improved = True
                        break
                    else:
                        best_U[a][r] = old_val
        
        # Try flipping entries in V
        for b in range(MP):
            for r in range(len(best_V[0])):
                old_val = best_V[b][r]
                for new_val in vals:
                    if new_val == old_val:
                        continue
                    best_V[b][r] = new_val
                    res = _tensor_residual(best_U, best_V, best_W, T)
                    norm = float(np.sum(res**2))
                    if norm < best_norm - 1e-10:
                        best_norm = norm
                        improved = True
                        break
                    else:
                        best_V[b][r] = old_val
        
        # Try flipping entries in W
        for c in range(PN):
            for r in range(len(best_W[0])):
                old_val = best_W[c][r]
                for new_val in vals:
                    if new_val == old_val:
                        continue
                    best_W[c][r] = new_val
                    res = _tensor_residual(best_U, best_V, best_W, T)
                    norm = float(np.sum(res**2))
                    if norm < best_norm - 1e-10:
                        best_norm = norm
                        improved = True
                        break
                    else:
                        best_W[c][r] = old_val
        
        if not improved:
            break
    
    return best_U, best_V, best_W


def search_deadline():
    """Main search with time limit."""
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    best = None
    best_rank = SCHOOBOOK_RANK
    
    # Start with schoolbook
    U0, V0, W0 = schoolbook()
    if verify_fast(U0, V0, W0):
        best = (U0, V0, W0)
        best_rank = SCHOOBOOK_RANK
    
    rng = random.Random(42)
    T = _build_tensor()
    
    # Try random matrices of decreasing rank, with local search refinement
    # Known bounds: for 2x2 * 2x8, the rank is likely 16-24 or so.
    # Let's try from rank 16 upward, with random searches.
    
    for target_rank in range(16, 32):
        if time.time() > deadline:
            break
        
        # Number of random trials for this rank
        # More trials for smaller ranks since they're harder but more rewarding
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        n_trials = 300 if target_rank <= 18 else 150 if target_rank <= 22 else 80
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Generate random matrices
            U = make_half_int_mat(NM, target_rank, rng)
            V = make_half_int_mat(MP, target_rank, rng)
            W = make_half_int_mat(PN, target_rank, rng)
            
            # Quick check: is it already valid?
            if verify_fast(U, V, W):
                best = (U, V, W)
                best_rank = target_rank
                break
            
            # Try local search to improve
            if time.time() > deadline:
                break
            
            # Compute residual norm to see if it's promising
            res = _tensor_residual(U, V, W, T)
            norm = float(np.sum(res**2))
            
            # Only do local search if the residual is reasonably small
            if norm < 50:  # threshold for promising candidates
                improved = _local_search(U, V, W, T, rng, deadline, max_iters=50)
                if improved is not None:
                    iU, iV, iW = improved
                    if _exact_residual_check(iU, iV, iW):
                        best = (iU, iV, iW)
                        best_rank = target_rank
                        break
        
        if best_rank <= target_rank:
            # Found something at this rank or better
            if best_rank < 16:
                break
    
    # If we found something, return it. Otherwise return schoolbook.
    if best is not None:
        U, V, W = best
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}


def construct():
    """Entry point: return the best decomposition found."""
    try:
        return search_deadline()
    except Exception:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
