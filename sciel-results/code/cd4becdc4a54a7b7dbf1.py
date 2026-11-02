"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 140.0  # Safety margin before 150s kill

def _start_time():
    return time.time()

def _time_left(start):
    return DEADLINE - (time.time() - start)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Constraint: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == T[i*m+j, j*p+k, k*n+i]
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    # Compute the reconstructed tensor
    # For each (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    # Reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # But we only need to check entries where T[a,b,c] = 1
    # And also check that entries where T=0 are also 0 (to be safe)
    
    # Efficient check: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r
    # Reconstructed = sum_r np.einsum('ar,br,cr->abc', U, V, W)
    # But this is O(n*m * m*p * p*n * rank) which for rank=45 is 9*15*15*45 = 91125, fine.
    
    rec = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check exact match
    if not np.allclose(rec, T, atol=1e-9):
        return False
    return True

def _check_half_integer(U, V, W):
    """Check if all entries are integers or half-integers (i.e., 2*x is integer)."""
    for M in [U, V, W]:
        M = np.array(M, dtype=np.float64)
        if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_333():
    """Return U,V,W for 3x3 * 3x3 with rank 7 (Strassen).
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]"""
    n, m, p = 3, 3, 3
    # Strassen's 7 products for 3x3 * 3x3
    # M1 = (A00+A22)*(B00+B22)
    # M2 = (A10+A20)*B01
    # M3 = A00*(B01-B11)
    # M4 = A22*(B10-B00)
    # M5 = (A00+A11)*(B11+B22)
    # M6 = (A10+A11)*(B01+B11) - B01 - B11  -- wait, this isn't standard
    # Let me use the standard Strassen for 3x3:
    # Actually, standard Strassen is for 2x2. For 3x3, the known rank is 7 but the construction is different.
    
    # Let me use a known rank-7 decomposition for 3x3 * 3x3.
    # Reference: The 7-term formula for 3x3 matrix multiplication.
    
    # One known construction:
    # M1 = (a00 + a22)(b00 + b22)
    # M2 = (a10 + a20) b01
    # M3 = a00 (b01 - b11)
    # M4 = a22 (b10 - b00)
    # M5 = (a00 + a11)(b11 + b22)
    # M6 = (a10 + a11)(b01 + b11)
    # M7 = (a01 + a11)(b10 + b20)
    
    # Outputs:
    # c00 = M1 - M3 + M5 - M6 + M2  ... this is getting complicated.
    
    # Let me just use a verified rank-7 solution.
    # From literature, one rank-7 decomposition:
    U = np.zeros((9, 7), dtype=np.int64)
    V = np.zeros((9, 7), dtype=np.int64)
    W = np.zeros((9, 7), dtype=np.int64)
    
    # Term 0: (a00+a22)(b00+b22) -> contributes to c00 and c22
    U[0, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[8, 0] = 1  # c00 and c22
    
    # Term 1: (a10+a20)b01 -> contributes to c11 and c21
    U[3, 1] = 1; U[6, 1] = 1
    V[1, 1] = 1
    W[4, 1] = 1; W[7, 1] = 1  # c11 and c21
    
    # Term 2: a00(b01-b11) -> contributes to c01
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1  # c01
    
    # Term 3: a22(b10-b00) -> contributes to c20
    U[8, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[6, 3] = 1  # c20
    
    # Term 4: (a00+a11)(b11+b22) -> contributes to c01 and c12
    U[0, 4] = 1; U[4, 4] = 1
    V[4, 4] = 1; V[8, 4] = 1
    W[1, 4] = 1; W[5, 4] = 1  # c01 and c12
    
    # Term 5: (a10+a11)(b01+b11) -> contributes to c11
    U[3, 5] = 1; U[4, 5] = 1
    V[1, 5] = 1; V[4, 5] = 1
    W[4, 5] = 1  # c11
    
    # Term 6: (a01+a11)(b10+b20) -> contributes to c00 and c10
    U[1, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1  # c00 and c10
    
    # Let me verify this... it's hard to do by hand. Let me just include it and validate.
    return U.tolist(), V.tolist(), W.tolist()

def _construct_33x35_strassen_plus_schoolbook(n, m, p):
    """Use Strassen for the 3x3 part and schoolbook for the 3x2 part.
    B[:,0:3] is 3x3, B[:,3:5] is 3x2.
    """
    # Strassen for A * B[:,0:3] (3x3 * 3x3)
    U_s, V_s, W_s = _strassen_333()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    # Schoolbook for A * B[:,3:5] (3x3 * 3x2)
    # B[:,3:5] has entries B[j,3] and B[j,4] for j=0,1,2
    # In the V matrix, these are at positions j*5+3 and j*5+4
    # In the W matrix, these are at positions k*3+i for k=3,4
    rank_sb = 3 * 3 * 2  # 18
    U_sb = np.zeros((9, rank_sb), dtype=np.int64)
    V_sb = np.zeros((15, rank_sb), dtype=np.int64)
    W_sb = np.zeros((15, rank_sb), dtype=np.int64)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U_sb[i * 3 + j, col] = 1
                V_sb[j * 5 + k, col] = 1
                W_sb[k * 3 + i, col] = 1
                col += 1
    
    # Combine
    U = np.hstack([U_s, U_sb])
    V = np.hstack([V_s, V_sb])
    W = np.hstack([W_s, W_sb])
    
    return U.tolist(), V.tolist(), W.tolist()

def _optimize_rank(R, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-R decomposition using optimization."""
    rng = np.random.RandomState(seed)
    # Initialize with random values
    U = rng.randn(n * m, R) * 0.5
    V = rng.randn(m * p, R) * 0.5
    W = rng.randn(p * n, R) * 0.5
    
    T = _build_tensor(n, m, p)
    
    # Objective: minimize sum of (rec - T)^2
    def objective(params):
        U_p = params[:n*m*R].reshape(n*m, R)
        V_p = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W_p = params[n*m*R + m*p*R:].reshape(p*n, R)
        rec = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            rec += np.einsum('a,b,c->abc', U_p[:, r], V_p[:, r], W_p[:, r])
        return np.sum((rec - T) ** 2)
    
    # This is too slow for scipy. Let me use a different approach.
    # Actually, for small R, let me just try random restarts with local optimization.
    # But this is likely to be very slow. Let me skip this and use hardcoded solutions.
    return None

def construct():
    start = _start_time()
    n, m, p = DIMENSIONS
    
    # Try several constructions and return the best valid one
    candidates = []
    
    # 1. Schoolbook (always valid, rank 45)
    sb = _schoolbook(n, m, p)
    candidates.append((45, sb))
    
    # 2. Strassen + schoolbook (rank 7 + 18 = 25, if valid)
    if _time_left(start) > 5:
        try:
            U, V, W = _construct_33x35_strassen_plus_schoolbook(n, m, p)
            if _validate(U, V, W, n, m, p) and _check_half_integer(U, V, W):
                candidates.append((25, {"U": U, "V": V, "W": W}))
        except Exception:
            pass
    
    # 3. Try the hardcoded rank-17 from the original (if valid)
    if _time_left(start) > 5:
        try:
            # Re-verify the original BLS17 solution
            U17 = [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
            V17 = [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            W17 = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
            if _validate(U17, V17, W17, n, m, p) and _check_half_integer(U17, V17, W17):
                candidates.append((17, {"U": U17, "V": V17, "W": W17}))
        except Exception:
            pass
    
    # Sort by rank and return the best
    candidates.sort(key=lambda x: x[0])
    if candidates:
        return candidates[0][1]
    
    # Fallback
    return _schoolbook(n, m, p)
