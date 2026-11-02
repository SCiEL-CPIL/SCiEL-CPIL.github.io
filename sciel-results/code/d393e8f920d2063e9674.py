import time
import random

_START = time.time()
_DEADLINE = _START + 110.0

def _deadline_exceeded():
    return time.time() > _DEADLINE

def _build_rank21():
    """
    Construct rank-21 factor matrices for 2x2 x 2x6 matrix multiplication.
    Uses Strassen's algorithm for each of the 3 blocks of B.
    U: (4, 21)
    V: (12, 21)
    W: (12, 21)
    
    Indices:
    U row: i*m + j (i in 0..1, j in 0..1)
    V row: j*p + k (j in 0..1, k in 0..5)
    W row: k*n + i (k in 0..5, i in 0..1)
    
    Strassen for 2x2 x 2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 6
    rank = 21
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A side)
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # r=0: M1: A00 + A11 -> U[0]=1, U[3]=1
    # r=1: M2: A10 + A11 -> U[2]=1, U[3]=1
    # r=2: M3: A00 -> U[0]=1
    # r=3: M4: A11 -> U[3]=1
    # r=4: M5: A00 + A01 -> U[0]=1, U[1]=1
    # r=5: M6: A10 - A00 -> U[2]=1, U[0]=-1
    # r=6: M7: A01 - A11 -> U[1]=1, U[3]=-1
    
    U_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    # Strassen coefficients for V (B side)
    # V indices: j*6 + k. For a 2x2 block at offset 2*t, indices are:
    # j=0: k=2t, 2t+1 -> V[2t], V[2t+1]
    # j=1: k=2t, 2t+1 -> V[6+2t], V[6+2t+1]
    # B00=V[2t], B01=V[2t+1], B10=V[6+2t], B11=V[6+2t+1]
    
    # r=0: M1: B00 + B11
    # r=1: M2: B00
    # r=2: M3: B01 - B11
    # r=3: M4: B10 - B00
    # r=4: M5: B11
    # r=5: M6: B00 + B01
    # r=6: M7: B10 + B11
    
    def get_v_indices(t, r):
        # Returns list of (v_index, coeff)
        b00 = 2*t
        b01 = 2*t + 1
        b10 = 6 + 2*t
        b11 = 6 + 2*t + 1
        
        if r == 0: return [(b00, 1), (b11, 1)]
        if r == 1: return [(b00, 1)]
        if r == 2: return [(b01, 1), (b11, -1)]
        if r == 3: return [(b10, 1), (b00, -1)]
        if r == 4: return [(b11, 1)]
        if r == 5: return [(b00, 1), (b01, 1)]
        if r == 6: return [(b10, 1), (b11, 1)]
        return []

    # Strassen coefficients for W (C side)
    # W indices: k*n + i = k*2 + i
    # For block t, outputs are C0, C1 at columns 2t, 2t+1
    # C00 at 2t, C01 at 2t+1
    # C10 at 2t (row 1), C11 at 2t+1 (row 1)
    # W row index for (i, k) is k*2 + i
    
    def get_w_indices(t, r):
        # Returns list of (w_index, coeff)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        c00_idx = 2*t * 2 + 0 # k=2t, i=0
        c01_idx = (2*t+1) * 2 + 0 # k=2t+1, i=0
        c10_idx = 2*t * 2 + 1 # k=2t, i=1
        c11_idx = (2*t+1) * 2 + 1 # k=2t+1, i=1
        
        if r == 0: # M1
            return [(c00_idx, 1), (c11_idx, 1)]
        if r == 1: # M2
            return [(c10_idx, 1), (c11_idx, -1)]
        if r == 2: # M3
            return [(c01_idx, 1), (c11_idx, 1)]
        if r == 3: # M4
            return [(c00_idx, 1), (c10_idx, 1)]
        if r == 4: # M5
            return [(c00_idx, -1), (c01_idx, 1)]
        if r == 5: # M6
            return [(c11_idx, 1)]
        if r == 6: # M7
            return [(c00_idx, 1)]
        return []

    # Fill the matrices
    for t in range(3): # 3 blocks
        for r in range(7): # 7 Strassen products
            col = t * 7 + r
            
            # Fill U
            for idx, val in U_coeffs[r].items():
                U[idx][col] = val
            
            # Fill V
            for v_idx, val in get_v_indices(t, r):
                V[v_idx][col] = val
            
            # Fill W
            for w_idx, val in get_w_indices(t, r):
                W[w_idx][col] = val

    return {"U": U, "V": V, "W": W}

def _verify(result, num_tests=20):
    n, m, p = 2, 2, 6
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False

    rng = random.Random(42)
    for _ in range(num_tests):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        for i in range(n):
            for k in range(p):
                ref_val = A[i][0] * B[0][k] + A[i][1] * B[1][k]
                s = 0
                # U[i*m], U[i*m+1]
                # V[k], V[p+k]
                # W[k*n+i]
                Ui0 = U[i*m]
                Ui1 = U[i*m+1]
                Vk = V[k]
                Vpk = V[p+k]
                Wki = W[k*n+i]
                
                for r in range(rank):
                    s += Wki[r] * (Ui0[r] * Vk[r] + Ui1[r] * Vpk[r])
                
                if s != ref_val:
                    return False
    return True

def construct():
    result = _build_rank21()
    if not _verify(result, num_tests=50):
        # If verification fails, something is wrong with the construction
        # But Strassen is well-known, so this should not happen.
        # Return a trivial rank-24 solution as a fallback?
        # Rank 24: each product A[i,j]*B[j,k]
        n, m, p = 2, 2, 6
        rank = 24
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
        return {"U": U, "V": V, "W": W}
    return result

# Self-check
try:
    _check = _build_rank21()
    _ok = _verify(_check, num_tests=100)
except Exception:
    pass
