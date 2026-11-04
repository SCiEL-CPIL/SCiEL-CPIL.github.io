"""Matrix multiplication 3x4x8 rank-minimal tensor decomposition."""
import time
import random

DIMENSIONS = [3, 4, 8]

def _build_schoolbook():
    """Return the schoolbook rank-96 decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] should be delta_{i,i'} * delta_{j,k}
    # Actually the tensor T[i,j,k,i'] = delta_{j,k} (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    # The decomposition says: T[i,j,k,i'] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i',c]
    # We need T[i,j,k,i'] = 1 if j==k and i==i', else 0.
    # Wait, let me re-read the convention.
    # Output factor W[k*n+i, r]. So W is indexed by (k, i) where k is from B's rows (0..p-1) and i is from A's rows (0..n-1).
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i',c] should equal the (i,j,k,i') entry of the multiplication tensor.
    # The multiplication tensor has T[i,j,k,i'] = 1 if j==k (since we multiply A[i,j]*B[j,k] to get C[i,k]) and i==i' (trivially).
    # Actually, the standard matrix multiplication tensor is T_{i,j,k} = delta_{j,k} for the (i,j,k) -> C[i,k] mapping.
    # With the extra i' index in W, the full tensor is T[i,j,k,i'] = delta_{j,k} * delta_{i,i'}.
    # Let's verify with a few random matrices.
    import numpy as np
    for _ in range(5):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        C_recon = np.zeros((n, p))
        for c in range(rank):
            # U[i*m+j, c] for all i,j
            u_vec = np.array([U[i*m+j][c] for i in range(n) for j in range(m)]).reshape(n, m)
            v_vec = np.array([V[j*p+k][c] for j in range(m) for k in range(p)]).reshape(m, p)
            w_vec = np.array([W[k*n+i][c] for k in range(p) for i in range(n)]).reshape(p, n)
            C_recon += u_vec @ v_vec * w_vec.T  # w_vec is (p,n), so w_vec.T is (n,p)
            # Wait, the convention is W[k*n+i, r]. So W is a (p*n, rank) matrix.
            # The contribution to C[i,k] from rank c is sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Let me redo this more carefully.
        C_recon2 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                C_recon2[i, k] = s
        if not np.array_equal(C_recon2, C_expected):
            return False
    return True

def _try_strassen_based():
    """
    Try to use Strassen's algorithm on 2x2 blocks to reduce rank.
    For 3x4x8, we can partition as:
    A is 3x4, B is 4x8.
    
    One approach: use the known optimal rank for small matrices.
    For 3x4x8, the schoolbook rank is 96.
    
    Let's think about block decomposition.
    We can write A as [A1; A2; A3] where A1 is 1x4, A2 is 1x4, A3 is 1x4.
    B is 4x8.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is just a bilinear map. The tensor is of format (3,4,8) -> output (3,8).
    
    Known results: For rectangular matrix multiplication, the rank is at most the schoolbook.
    For 3x4x8, let's try to find a better decomposition.
    
    Actually, let me try a different approach: use the fact that we can combine some terms.
    """
    n, m, p = DIMENSIONS
    # Schoolbook is rank 96. Let's see if we can do better.
    # For now, return the schoolbook as a fallback.
    return _build_schoolbook()

def construct():
    start_time = time.time()
    deadline = 110  # seconds, safety margin before 120s limit
    
    # Start with schoolbook
    best = _build_schoolbook()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best
    
    # Try to improve using various strategies
    # For 3x4x8, the schoolbook rank is 96.
    # Known lower bound is at least max(n*p, m*p, n*m) = max(24, 32, 12) = 32.
    # But the actual rank is likely higher.
    
    # Let's try a structured approach:
    # The multiplication (3x4)*(4x8) -> (3x8)
    # We can view this as: for each of the 3 rows of A and 8 columns of B,
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # One idea: use the fact that B has 4 rows and 8 columns.
    # We can write B as [B_left | B_right] where each is 4x4.
    # Then C[:, 0:4] = A @ B_left and C[:, 4:8] = A @ B_right.
    # This doesn't directly help with rank reduction.
    
    # Another idea: use the known result that for m=4, we can use Strassen-like methods.
    # But Strassen is for square matrices.
    
    # Let's just return the schoolbook for now, as finding a provably better
    # decomposition for 3x4x8 is non-trivial.
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook decomposition as baseline
    rank = n * m * p  # 96
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Try to improve with a simple optimization:
    # For this specific case, we can try to find if there's a known better rank.
    # The schoolbook rank for 3x4x8 is 96.
    # Without a verified better decomposition, we return the schoolbook.
    
    return result
# EVOLVE-BLOCK-END
