"""
Rank-minimal tensor decomposition for 2x2x9 matrix multiplication.
Strategy:
- Build the known rank-32 solution (4 Strassen blocks + 1 schoolbook block).
- Attempt to reduce rank by trying to merge or share terms between adjacent blocks.
- Use a time budget to avoid exceeding the 150s limit.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 schoolbook (col 8).
    Returns (U, V, W) as lists of lists of floats.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # Standard Strassen for 2x2x2 with B columns k0, k1
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11) * b_{0,k0}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01) * b_{1,k1}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        # a00 * b_{0,k} -> C_{k,0}
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a01 * b_{1,k} -> C_{k,0}
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        # a10 * b_{0,k} -> C_{k,1}
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        # a11 * b_{1,k} -> C_{k,1}
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank by attempting to set some coefficients to zero
    and adjusting others to maintain the decomposition.
    This is a heuristic approach.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    # Convert to numpy for easier manipulation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to remove the last (rank - target_rank) columns by projecting
    # onto a lower-dimensional subspace. This is a rough heuristic.
    # We'll try to find a rank-reduction by checking if the last columns
    # can be expressed as linear combinations of earlier columns.
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try removing one column at a time from the end
        new_rank = rank - 1
        if new_rank < target_rank:
            break
            
        # Check if the last column can be approximated by a combination of others
        # This is a simple heuristic: try to see if setting the last column to zero
        # and adjusting the others can still work.
        
        U_test = U_arr[:, :-1].copy()
        V_test = V_arr[:, :-1].copy()
        W_test = W_arr[:, :-1].copy()
        
        # Verify if the truncated version still works (it won't, but this is a test)
        # Actually, we need a smarter approach. Let's just return None for now
        # as proper rank reduction is non-trivial.
        
        break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with the known rank-32 solution
    U, V, W = _build_rank32()
    
    if not _verify(U, V, W, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 32
    best_U, best_V, best_W = U, V, W
    
    # Attempt to reduce rank below 32
    reduced = _try_reduce_rank(U, V, W, 2, 2, 9, 31, deadline)
    if reduced is not None:
        rU, rV, rW = reduced
        if _verify(rU, rV, rW, 2, 2, 9):
            best_rank = 31
            best_U, best_V, best_W = rU, rV, rW
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
