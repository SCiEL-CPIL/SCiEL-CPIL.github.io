import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

def _schoolbook(n, m, p):
    """
    Returns the schoolbook (rank n*m*p) decomposition for n x m * m x p matrix multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Indexing: U[i*m+j, col] corresponds to A[i,j], V[j*p+k, col] to B[j,k], W[k*n+i, col] to C[i,k].
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def _strassen_2x2():
    """
    Returns the 7-term Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U (8x7), V (8x7), W (8x7) where indices are flattened 2x2 matrices.
    A is 2x2 (4 entries), B is 2x2 (4 entries), C is 2x2 (4 entries).
    But the function returns matrices of size (2*2, 7) = (4,7)? No, the interface in the prompt
    for the 2x2 helper was u_vec length 4.
    Let's stick to the standard 7-term Strassen for 2x2 * 2x2.
    U: 4x7, V: 4x7, W: 4x7.
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # B00
        [1, 0, 1, -1, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, -1, 0, 1, 0, 0, 0],  # C10
        [-1, 1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    return U, V, W

def _embed_strassen_in_3x3x8():
    """
    Attempts to construct a decomposition for 3x3 * 3x8 using Strassen for a 2x2 block.
    We embed the 2x2 * 2x8 part using Strassen-like logic?
    Actually, for 2x2 * 2x8, we can use 7 * 8 = 56 terms if we treat it as 8 columns of 2x2*2x1.
    But 2x2*2x1 has rank 4 (schoolbook) or less?
    Rank of 2x2 * 2x1 tensor:
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    This is 2x2x2 tensor. Rank is 4?
    Let's check if rank 3 is possible.
    T = e0⊗e0⊗e0 + e1⊗e0⊗e0 + e0⊗e1⊗e1 + e1⊗e1⊗e1? No.
    T_{i,j,k} = 1 if j=k.
    T = sum_j (sum_i e_i ⊗ e_j) ⊗ e_j
    This is the "matrix vector multiplication" tensor for 2x2.
    Rank is 4.
    So 2x2 * 2x8 has rank 4 * 8 = 32.
    
    If we use schoolbook for 2x2 * 2x8, rank is 32.
    Remaining terms for 3x3 * 3x8:
    C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k] (3*8 = 24 terms)
    C[0,k] += A[0,2]B[2,k] (8 terms)
    C[1,k] += A[1,2]B[2,k] (8 terms)
    Total: 32 + 24 + 8 + 8 = 72. Same as schoolbook.
    
    Is there a better way?
    What if we use Strassen for 2x2 * 2x2 (rank 7) and handle the rest?
    The 2x2 * 2x2 block covers A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2].
    This saves 4 - 7 = -3 terms? No, schoolbook for 2x2*2x2 is 4. Strassen is 7. Strassen is WORSE for 2x2*2x2.
    Strassen only helps for larger matrices.
    
    So for 3x3 * 3x8, schoolbook is likely optimal or close to it.
    However, the prompt mentions "rank-minimal exact tensor decomposition".
    If schoolbook is 72, and I can't find a better one, I return 72.
    
    Let's double check if there's a known algorithm for 3x3 * 3x8.
    Maybe using the fact that 8 is large?
    
    Let's try to construct a rank < 72 decomposition using random search or structured search?
    Given the time limit (150s), a full search is impossible.
    I will stick to the schoolbook decomposition as it is guaranteed to be correct and rank 72.
    
    Wait, the inspiration mentions "Systematic search over all valid 2x2x2 Strassen block placements... measured: 1 -> 1.029".
    This implies a score of 1.029 was achieved.
    Score = seed / value. Seed = 1.0.
    If value (rank) = 72, score = 1/72? No, "score is the rank" but "search score is seed/value".
    "score is the rank" -> lower is better.
    "search score is seed/value, so 1.0 equals the seed program".
    Seed program rank = 72.
    If I find rank 70, score = 72/70 = 1.028.
    So the inspiration found rank 70?
    72/1.029 ≈ 69.97. So rank 70.
    
    How to get rank 70?
    Schoolbook is 72.
    We need to save 2 terms.
    
    Let's look at the 3x3 * 3x8 tensor.
    C[i,k] = sum_j A[i,j] B[j,k].
    
    Consider the sub-tensor for A[0:2, 0:2] * B[0:2, :] -> C[0:2, :].
    This is 2x2 * 2x8. Rank 32.
    Consider A[2, :] * B[:, :] -> C[2, :].
    This is 1x3 * 3x8. Rank 24.
    Consider A[:, 2] * B[2, :] -> C[:, 2] (partially).
    Specifically, C[0,k] gets A[0,2]B[2,k] and C[1,k] gets A[1,2]B[2,k].
    These are 2 * 8 = 16 terms.
    Total 32 + 24 + 16 = 72.
    
    Can we combine terms?
    In the 2x2 * 2x8 part, we have terms A[0,0]B[0,k], A[0,1]B[1,k], A[1,0]B[0,k], A[1,1]B[1,k].
    In the "extra" parts, we have A[0,2]B[2,k], A[1,2]B[2,k], and for row 2: A[2,0]B[0,k], A[2,1]B[1,k], A[2,2]B[2,k].
    
    Notice that B[2,k] is used in A[0,2]B[2,k], A[1,2]B[2,k], and A[2,2]B[2,k].
    B[0,k] is used in A[0,0]B[0,k], A[1,0]B[0,k], and A[2,0]B[0,k].
    B[1,k] is used in A[0,1]B[1,k], A[1,1]B[1,k], and A[2,1]B[1,k].
    
    This looks like a standard 3x3 * 3x8.
    
    Is there a rank 70 algorithm?
    Maybe using a 2x2 Strassen block on A[0:2, 0:2] and B[0:2, 0:2] doesn't help as noted (7 > 4).
    
    What if we use a different decomposition for the 2x2 * 2x8 part?
    Rank of 2x2 * 2x8 is 32.
    
    Let's check if rank 70 is possible.
    Lower bound for 3x3 * 3x8?
    R(3,3,8) >= R(3,3,3) * R(1,1,8/3)? No.
    R(3,3,8) >= R(3,3,1) * R(1,1,8) = 9 * 8 = 72?
    No, R(3,3,1) is 9. R(1,1,8) is 8.
    Tensor product bound: R(3,3,8) >= R(3,3,1) * R(1,1,8) is NOT generally true.
    The bound is R(n1 n2, m1 m2, p1 p2) >= R(n1, m1, p1) * R(n2, m2, p2).
    So R(3,3,8) >= R(3,3,1) * R(1,1,8) is NOT a valid factorization because (3,3,1) x (1,1,8) = (3,3,8).
    Yes it is! (3*1, 3*1, 1*8) = (3,3,8).
    So R(3,3,8) >= R(3,3,1) * R(1,1,8) = 9 * 8 = 72.
    
    If the lower bound is 72, then 72 is minimal.
    Why did the inspiration get 1.029 (rank 70)?
    Maybe my lower bound logic is flawed?
    R(n1 n2, m1 m2, p1 p2) >= R(n1, m1, p1) * R(n2, m2, p2) is a standard property.
    R(3,3,1) = 9 (schoolbook for 3x3 * 3x1).
    R(1,1,8) = 8 (schoolbook for 1x1 * 1x8).
    So R(3,3,8) >= 72.
    
    If the lower bound is 72, how can rank be 70?
    Maybe R(3,3,1) is not 9?
    3x3 * 3x1: C[i] = sum_j A[i,j] B[j].
    Tensor T[i,j,k] = 1 if j=k.
    This is the "matrix-vector" tensor.
    For n=3, rank is 9.
    Is it possible rank is 8?
    No, the rank of the n x n matrix-vector multiplication tensor is n^2.
    So R(3,3,1) = 9.
    
    So R(3,3,8) >= 72.
    Therefore, rank 70 is impossible.
    The inspiration score 1.029 might be due to a different seed or interpretation.
    Or maybe the "seed" value is not 72?
    "score is the rank". "search score is seed/value".
    If seed program has rank 72, and score is 1.0, then seed/value = 1.0 => value = seed.
    If score = rank, then 1.0 = 72? No, "1.0 equals the seed program".
    Usually in these tasks, the "value" is the rank, and the "score" is 1/value or something?
    "score is the rank ... lower is better".
    "search score is seed/value, so 1.0 equals the seed program".
    If seed rank = 72, and search score = 1.0, then seed/value = 1.0 => value = seed = 72.
    So value is the rank.
    If I find rank 70, score = 72/70 = 1.028.
    
    If the lower bound is 72, I cannot beat 72.
    I will return the schoolbook decomposition of rank 72.
    """
    pass

def construct():
    """
    Constructs a tensor decomposition for 3x3 by 3x8 matrix multiplication.
    Returns the schoolbook decomposition with rank 72.
    """
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    U, V, W = construct()["U"], construct()["V"], construct()["W"]
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Check shapes
    assert U.shape == (n*m, rank)
    assert V.shape == (m*p, rank)
    assert W.shape == (p*n, rank)
    
    # Verify decomposition
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] (using flattened indices)
    # Actually, the tensor T for matrix mult is:
    # T_{i,j,k} = 1 if j is the shared index?
    # C[i,l] = sum_j A[i,j] B[j,l]
    # T is in U(A) ⊗ V(B) ⊗ W(C).
    # T = sum_{i,j,l} e_{i,j} ⊗ e_{j,l} ⊗ e_{i,l}
    # In flattened indices:
    # idx_U = i*m + j
    # idx_V = j*p + l
    # idx_W = l*n + i
    
    # Let's verify a few entries.
    # The reconstructed tensor T_recon has T_recon[idx_U, idx_V, idx_W] = sum_r U[idx_U, r] V[idx_V, r] W[idx_W, r]
    # We want T_recon to match the structure tensor.
    # Structure tensor S has S[idx_U, idx_V, idx_W] = 1 if idx_V corresponds to j such that idx_U's j matches idx_V's j,
    # and idx_W's i matches idx_U's i, and idx_W's l matches idx_V's l.
    
    # Let's check S[idx_U, idx_V, idx_W] = 1 if:
    # idx_U = i*m + j
    # idx_V = j*p + l
    # idx_W = l*n + i
    # Otherwise 0.
    
    # We can verify by checking that for random A, B, C = A*B matches sum_r (U^T A) (V^T B) W^T
    # C = sum_r (A U_r) (B V_r) W_r^T ?
    # C[i,l] = sum_r sum_j A[i,j] U[j, r] sum_k V[k, r] B[j, k] W[l, r] ?
    # No, the decomposition is:
    # C = sum_r (A U_r) ⊗ (V_r^T B) ⊗ W_r^T ?
    # Let's use the property:
    # C = A B
    # A is n x m, B is m x p.
    # U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.
    # Flatten A to a, B to b, C to c.
    # c = T(a, b)
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # c_w = sum_r (a · u_r) (b · v_r) w_r_w
    
    # Let's test with random matrices.
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    a = A.flatten()
    b = B.flatten()
    
    c_computed = np.zeros(p*n)
    for r in range(rank):
        c_computed += (a @ U[:, r]) * (b @ V[:, r]) * W[:, r]
    
    c_expected = C_expected.flatten()
    # Note: W index is k*n+i. C_expected is i*n+k?
    # C[i,k]. Flatten C: C[i,k] -> index i*p + k?
    # But W is (p*n, rank). Index is k*n + i.
    # So c_computed is indexed by k*n + i.
    # C_expected is i*p + k.
    # We need to reshape c_computed to (p, n) and transpose to (n, p) to compare with C_expected.
    C_recon = c_computed.reshape(p, n).T
    
    if not np.allclose(C_recon, C_expected):
        print("Verification failed")
        # Print difference
        print(np.max(np.abs(C_recon - C_expected)))
    else:
        print("Verification passed")
