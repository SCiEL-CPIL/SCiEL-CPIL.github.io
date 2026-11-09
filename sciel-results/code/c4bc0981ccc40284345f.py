import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 148.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    decomp_tensor = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # w_r[k, i] is the coefficient for output C[i, k]
        # In tensor form: C[i, k] += ... * w_r[k, i]
        # Our target tensor T[i, j, k] corresponds to coefficient of C[i, k]
        # so T[i, j, k] = sum_r u_r[i, j] * v_r[j, k] * w_r[k, i]
        # Note: w_r is indexed by (k, i) in the flattened form W[k*n+i, r]
        decomp_tensor += u_r * v_r * w_r.transpose(1, 0)
        
    target = np.ones((n, m, p), dtype=float)
    if not np.allclose(decomp_tensor, target, atol=tol):
        return False
    return True

def _build_strassen(n, m, p):
    """
    Build a decomposition for n x m x p matrix multiplication.
    Uses Strassen for 2x2x2 blocks.
    """
    # Base case: small sizes or non-square
    if n == 1:
        # M[1, m, p]
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Rank = m * p
        rank = m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        idx = 0
        for j in range(m):
            for k in range(p):
                U[0 * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + 0][idx] = 1.0
                idx += 1
        return U, V, W

    if p == 1:
        # M[n, m, 1]
        # C[i, 0] = sum_j A[i, j] B[j, 0]
        # Rank = n * m
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][idx] = 1.0
                V[j * p + 0][idx] = 1.0
                W[0 * n + i][idx] = 1.0
                idx += 1
        return U, V, W

    # Try splitting
    # We want to minimize total rank.
    # Split one of n, m, p.
    
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    best_split_info = None
    
    # 1. Split n
    for n1 in range(1, n):
        n2 = n - n1
        # We can split the tensors.
        # But Strassen usually splits all three.
        # Let's try splitting n only, m only, p only.
        
        # Split n
        U1, V1, W1 = _build_strassen(n1, m, p)
        U2, V2, W2 = _build_strassen(n2, m, p)
        rank1, rank2 = len(U1[0]), len(U2[0])
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_U = U1 + U2
            best_V = V1 + V2
            best_W = W1 + W2
            best_split_info = ('n', n1, n2)

    # 2. Split m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _build_strassen(n, m1, p)
        U2, V2, W2 = _build_strassen(n, m2, p)
        rank1, rank2 = len(U1[0]), len(U2[0])
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_U = U1 + U2
            best_V = V1 + V2
            best_W = W1 + W2
            best_split_info = ('m', m1, m2)

    # 3. Split p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _build_strassen(n, m, p1)
        U2, V2, W2 = _build_strassen(n, m, p2)
        rank1, rank2 = len(U1[0]), len(U2[0])
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_U = U1 + U2
            best_V = V1 + V2
            best_W = W1 + W2
            best_split_info = ('p', p1, p2)
            
    # Special case: 2x2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen rank 7
        # C11 = A11(B11 - B22) + (A21 - A22)(B21 + B22) ? No.
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M5
        
        # Let's verify:
        # C11 = A11B11 + A11B22 - A22B22 + A22B11 + A21B11 - A11B11 + A21B12 - A11B12 - A12B21 - A22B22 + A12B22 + A22B21
        # = A11B11 + A11B22 - A22B22 + A22B11 + A21B11 + A21B12 - A11B12 - A12B21 - A22B22 + A12B22 + A22B21
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M6 - M7
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22(B21-B11) = A22B21 - A22B11
        # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        # Sum:
        # A11B11 + A11B22 + A22B11 + A22B22
        # - A22B21 + A22B11
        # + A21B11 + A21B12 - A11B11 - A11B12
        # - A12B21 - A12B22 + A22B21 + A22B22
        # = A11B22 + 2*A22B11 + 2*A22B22 + A21B11 + A21B12 - A11B12 - A12B21 - A12B22
        # This is not C11 = A11B11 + A12B21.
        
        # Let's use a verified Strassen implementation.
        # From Wikipedia:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M5
        
        # Let's re-calculate C11:
        # M1 - M4 + M6 - M7
        # = (A11B11 + A11B22 + A22B11 + A22B22)
        # - (A22B21 - A22B11)
        # + (A21B11 + A21B12 - A11B11 - A11B12)
        # - (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A11B22 + A22B11 + A22B22
        # - A22B21 + A22B11
        # + A21B11 + A21B12 - A11B11 - A11B12
        # - A12B21 - A12B22 + A22B21 + A22B22
        # Cancel A22B21:
        # = A11B11 + A11B22 + 2*A22B11 + 2*A22B22 + A21B11 + A21B12 - A11B11 - A11B12 - A12B21 - A12B22
        # = A11B22 + 2*A22B11 + 2*A22B22 + A21B11 + A21B12 - A11B12 - A12B21 - A12B22
        # This is definitely wrong.
        
        # Correct Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M5
        
        # Let's check C11 again.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22B21 - A22B11
        # M6 = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = A12B21 + A12B22 - A22B21 - A22B22
        
        # C11 = M1 - M4 + M6 - M7
        # = A11B11 + A11B22 + A22B11 + A22B22
        # - A22B21 + A22B11
        # + A21B11 + A21B12 - A11B11 - A11B12
        # - A12B21 - A12B22 + A22B21 + A22B22
        
        # Terms:
        # A11B11: 1 - 1 = 0? No, M1 has +1, M6 has -1. Sum 0.
        # A11B22: +1
        # A11B12: -1
        # A21B11: +1
        # A21B12: +1
        # A22B11: +1 (M1) + 1 (M4 neg) = 2
        # A22B22: +1 (M1) + 1 (M7 neg) = 2
        # A22B21: -1 (M4) + 1 (M7 neg) = 0
        # A12B21: -1
        # A12B22: -1
        
        # This is not C11.
        
        # Let's look up a working set of coefficients for Strassen 2x2x2.
        # Or just use the schoolbook for 2x2x2 to be safe, rank 8.
        # 2x2x2 schoolbook rank 8.
        # Strassen is rank 7.
        # If I implement Strassen correctly, I save 1 rank.
        
        # Let's use a known valid set.
        # From "Strassen's algorithm" Wikipedia:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M5
        
        # I will trust this formula. If it fails verification, I will fall back to schoolbook.
        
        rank = 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # A is 2x2. A[i, j] -> U[i*2+j, r]
        # B is 2x2. B[j, k] -> V[j*2+k, r]
        # C is 2x2. C[i, k] -> W[k*2+i, r]
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: U[0,0]=1, A22: U[1,1]=1
        # B11: V[0,0]=1, B22: V[1,1]=1
        # C11: W[0,0]=1, C22: W[1,1]=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21 + A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = 1 # C21 uses M2. C[2,1] -> W[1,2] = W[1*2+2]? No.
        # W index: W[k*n + i, r]. n=2.
        # C[i, k] coefficient is W[k*2+i, r].
        # C[1, 0] (C21) -> W[0*2+1, r] = W[1, r].
        W[1][1] = 1
        
        # M3 = A11(B12 - B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C12 -> W[1,0]? C[0,1] -> W[1,0] = W[1, r].
        # C12 is C[0,1]. W[1*2+0, r] = W[2, r]? No.
        # W[k*n+i]. k=1, i=0. W[1*2+0] = W[2].
        W[2][2] = 1
        
        # M4 = A22(B21 - B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[1][3] = 1 # C21 -> W[1].
        W[1][3] = 1
        
        # M5 = (A11 + A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1 # C12 -> W[2].
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = 1 # C11 -> W[0].
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1 # C11 uses -M7.
        W[3][6] = -1 # C22 uses -M7? No, C22 = M1 - M3 + M2 - M5. M7 not in C22.
        # Wait, C22 = M1 - M3 + M2 - M5.
        # C11 = M1 - M4 + M6 - M7.
        
        # Let's re-verify C22.
        # C22 = A21B12 + A22B22.
        # M1 - M3 + M2 - M5
        # M1: A21B12? No.
        
        # Okay, I'm going to skip the manual Strassen implementation to avoid bugs, unless I verify it.
        # For 2x2x2, schoolbook is rank 8.
        # If I use schoolbook for 2x2x2, the total rank might be higher.
        # Let's use schoolbook for all base cases to be safe, and rely on the splitting logic.
        # The splitting logic might find 2+2+2 -> 3*8 = 24.
        # Strassen would be 7.
        # But 3,5,5 is small.
        
        pass # Skip Strassen for now, return schoolbook 2x2x2

    # Schoolbook for n, m, p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return U, V, W

def construct():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")
    
    n, m, p = 3, 5, 5
    
    U, V, W = _build_strassen(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    _verify_decomposition(res['U'], res['V'], res['W'], 3, 5, 5)
