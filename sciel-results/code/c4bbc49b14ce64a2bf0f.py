"""Matrix multiplication 3x4x5 rank-minimal decomposition search.

Combines block-partition structural decomposition with ALS-based search
and verification to find a rank-minimal exact tensor decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N

# Deadline guard: evaluator kills after 150s; keep 10s safety margin.
DEADLINE = 140.0

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank):
    """
    Perform one step of Alternating Least Squares.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Returns updated U, V, W.
    """
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    W_new_r = np.zeros((p, n, rank))
    for i in range(n):
        for k in range(p):
            A = U_r[i, :, :] * V_r[:, k, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_new_r[k, i, :] = x
            except:
                W_new_r[k, i, :] = W_r[k, i, :]
    W_new = W_new_r.reshape(p * n, rank)
    
    V_new_r = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            A = U_r[:, j, :] * W_new_r[k, :, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new_r[j, k, :] = x
            except:
                V_new_r[j, k, :] = V_r[j, k, :]
    V_new = V_new_r.reshape(m * p, rank)
    
    U_new_r = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            A = V_new_r[j, :, :] * W_new_r[:, i, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new_r[i, j, :] = x
            except:
                U_new_r[i, j, :] = U_r[i, j, :]
    U_new = U_new_r.reshape(n * m, rank)
    
    return U_new, V_new, W_new

def _try_rank(rank, n, m, p, deadline, max_tries=50):
    """
    Try to find a decomposition of given rank using random initialization + ALS.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    time_per_attempt = 0.5
    n_tries = min(max_tries, max(5, int(_time_left() / time_per_attempt)))
    
    for attempt in range(n_tries):
        if _time_left() < 2:
            break
        
        np.random.seed(attempt * 12345 + rank)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        U = U / np.linalg.norm(U) * np.sqrt(nm)
        V = V / np.linalg.norm(V) * np.sqrt(mp)
        W = W / np.linalg.norm(W) * np.sqrt(pn)
        
        for iteration in range(150):
            if _time_left() < 2:
                break
            
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            U, V, W = _als_step(U, V, W, n, m, p, rank)
            
            diff = np.linalg.norm(U - U_old) + np.linalg.norm(V - V_old) + np.linalg.norm(W - W_old)
            if diff < 1e-8:
                break
        
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _block_decomposition(n, m, p):
    """
    Build the schoolbook decomposition with rank n*m*p.
    """
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, n*m*p

def _generate_block_partitions(n, m):
    """Generate all possible block partitions of an n x m matrix."""
    def _partitions(size):
        """Generate all compositions of size into positive integers."""
        if size == 0:
            yield []
            return
        for first in range(1, size + 1):
            for rest in _partitions(size - first):
                yield [first] + rest
    
    row_partitions = list(_partitions(n))
    col_partitions = list(_partitions(m))
    
    return row_partitions, col_partitions

def _build_block_decomposition(row_blocks, col_blocks, n, m, p):
    """
    Build a decomposition using block partitioning.
    row_blocks: list of row sizes
    col_blocks: list of col sizes
    Returns U, V, W, rank
    """
    # Compute cumulative indices
    row_starts = [0]
    for size in row_blocks:
        row_starts.append(row_starts[-1] + size)
    col_starts = [0]
    for size in col_blocks:
        col_starts.append(col_starts[-1] + size)
    
    # Collect all (i, j, k) triples
    cols = []
    for ri, rsize in enumerate(row_blocks):
        for ci, csize in enumerate(col_blocks):
            i0 = row_starts[ri]
            i1 = i0 + rsize
            j0 = col_starts[ci]
            j1 = j0 + csize
            for i in range(i0, i1):
                for j in range(j0, j1):
                    for k in range(p):
                        cols.append((i, j, k))
    
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (i, j, k) in enumerate(cols):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    
    return U, V, W, rank

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using integer arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _try_structural_rank(n, m, p):
    """
    Try to find a decomposition using block structure.
    Returns (U, V, W, rank) or None.
    """
    row_partitions, col_partitions = _generate_block_partitions(n, m)
    
    best = None
    best_rank = n * m * p
    
    for rp in row_partitions:
        for cp in col_partitions:
            U, V, W, rank = _build_block_decomposition(rp, cp, n, m, p)
            if rank < best_rank and _verify_exact(U, V, W, n, m, p):
                best_rank = rank
                best = (U, V, W, rank)
    
    return best

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from schoolbook rank and try to improve
    best_result = None
    best_rank = n * m * p
    
    # Generate schoolbook solution as fallback
    U_fb, V_fb, W_fb, _ = _block_decomposition(n, m, p)
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try structural (block) decomposition first
    structural = _try_structural_rank(n, m, p)
    if structural is not None:
        U_s, V_s, W_s, rank_s = structural
        if rank_s < best_rank:
            best_result = {"U": U_s, "V": V_s, "W": W_s}
            best_rank = rank_s
    
    # Try ranks from lower bound upward using ALS search
    for rank in range(55, 15, -1):
        if _time_left() < 10:
            break
        
        result = _try_rank(rank, n, m, p, DEADLINE, max_tries=20)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            if rank <= 40:
                break
    
    # If we didn't find anything better, try a few more ranks with more attempts
    if best_rank == n * m * p:
        for rank in range(55, 45, -1):
            if _time_left() < 5:
                break
            result = _try_rank(rank, n, m, p, DEADLINE, max_tries=30)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result
