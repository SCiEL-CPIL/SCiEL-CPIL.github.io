import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
    Indices:
    U: i*2+j (0..3)
    V: j*2+k (0..3)
    W: k*2+i (0..3)
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    V = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 1, 0, -1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 1]
    ], dtype=float)
    return U, V, W

def _construct_2x2x2x3():
    """
    Constructs a rank-10 decomposition for 2x2 * 2x3 = 2x3.
    Indices:
    U: i*2+j (0..3)
    V: j*3+k (0..5)
    W: k*2+i (0..5)
    
    Strategy:
    Use Strassen (rank 7) for the first two columns of Y (2x2 subproblem).
    Use Schoolbook (rank 4) for the third column of Y (2x1 subproblem).
    Total rank 11.
    Can we do better?
    Known result: Rank(2x2 x 2xn) = 3n+1 for n>=2.
    For n=3, Rank = 10.
    
    How to get rank 10?
    One way is to share one term between the Strassen part and the Schoolbook part.
    Or use a different algorithm.
    
    Let's try to find a rank 10 decomposition.
    If we can't find one easily, we'll use rank 11.
    
    Actually, let's look at the structure.
    Z = X * Y.
    Y = [Y0, Y1, Y2].
    Z0 = X Y0, Z1 = X Y1, Z2 = X Y2.
    
    Strassen for X * [Y0, Y1]:
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11) Y00
    M3 = X00 (Y10-Y00)
    M4 = X11 (Y11-Y01)
    M5 = (X00+X01) Y11
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
    
    Z00 = M1 - M4 + M5 - M7
    Z01 = M3 + M4
    Z10 = M2 + M7
    Z11 = M1 - M2 + M3 + M6
    
    Schoolbook for X * Y2:
    N1 = X00 Y02
    N2 = X01 Y12
    N3 = X10 Y02
    N4 = X11 Y12
    
    Z02 = N1 + N2
    Z12 = N3 + N4
    
    Total 11 terms.
    
    Is there a rank 10?
    If we combine N1 and M1? No, different Y indices.
    
    Let's check if rank 10 is possible.
    The tensor is 4x6x6.
    Rank 10 means 10 terms.
    
    There is a known rank 10 algorithm for 2x2 x 2x3.
    It is derived from the 2x2 x 2x2 Strassen by extending it.
    
    Let's try to search for a rank 10 decomposition numerically if needed,
    but for now, let's implement the rank 11 one which is guaranteed correct.
    Total rank for 2x4 x 4x6 would be 11 + 11 = 22.
    
    If we can find rank 10, total is 20.
    
    Let's try to construct rank 10.
    Reference: "On the rank of the 2x2x2xn tensor"
    The rank is 3n+1.
    For n=3, 10.
    
    The construction usually involves 3 terms for each new column, plus 1 shared.
    
    Let's try to find the specific coefficients.
    
    Alternatively, we can use the rank 11 decomposition which is simple and correct.
    Score for rank 22 is 48/22 = 2.18.
    Score for rank 20 is 48/20 = 2.40.
    
    Let's try to implement rank 10.
    
    I found a reference for rank 10:
    It uses 7 Strassen terms for the first 2 cols, and 3 terms for the 3rd col?
    No, 3 terms for 2x2 x 2x1 is not enough (rank 4).
    
    Maybe it shares 1 term?
    If we share 1 term, we have 7 + 4 - 1 = 10.
    
    Which term can be shared?
    M1 uses Y00, Y11.
    N1 uses Y02.
    No obvious overlap.
    
    Let's stick to rank 11 for safety, or try to search for rank 10.
    Given the time limit, I will implement rank 11 first.
    If I have time, I can try to optimize.
    
    Actually, let's look at the inspiration.
    Inspiration 1, 2, 3 scored 1.1429.
    Current score is 1.0000 (Rank 48).
    Rank 22 gives 48/22 = 2.18.
    Wait, the score is seed/value.
    Seed program score 1.0000 implies value = seed.
    If seed program is Rank 48, value = 48?
    "score is the rank".
    "evaluator measures the raw value (lower is better)".
    "search score is seed/value".
    If seed is 1.0, and value is rank, then score = 1.0 / rank?
    No, "1.0 equals the seed program".
    So if seed program has rank R_seed, score = 1.0.
    If my program has rank R, score = R_seed / R.
    
    If R_seed = 48 (schoolbook), and I find R=22, score = 48/22 = 2.18.
    This is a huge improvement.
    
    The inspirations scored 1.1429.
    48 / 1.1429 = 42.
    So inspirations found Rank 42.
    My Rank 22 would be much better.
    
    So Rank 22 is a very good target.
    
    Let's implement Rank 22 (2 blocks of Rank 11).
    """
    
    U22, V22, W22 = _strassen_2x2x2()
    
    U = np.zeros((4, 11), dtype=float)
    V = np.zeros((6, 11), dtype=float)
    W = np.zeros((6, 11), dtype=float)
    
    # Part 1: Strassen for Y[:, 0:2]
    U[:, 0:7] = U22
    
    # Map V22 (4x7) to V (6x7)
    # V22 index j*2+k_sub -> V index j*3+k_sub
    for r in range(7):
        for j in range(2):
            for k_sub in range(2):
                V[j*3 + k_sub, r] = V22[j*2 + k_sub, r]
    
    # Map W22 (4x7) to W (6x7)
    # W22 index k_sub*2+i -> W index k_sub*2+i
    for r in range(7):
        for k_sub in range(2):
            for i in range(2):
                W[k_sub*2 + i, r] = W22[k_sub*2 + i, r]
    
    # Part 2: Schoolbook for Y[:, 2]
    # Terms 7-10
    # 1. X00 * Y02 -> Z02
    U[0, 7] = 1
    V[2, 7] = 1  # Y02 index 0*3+2
    W[4, 7] = 1  # Z02 index 2*2+0
    
    # 2. X01 * Y12 -> Z02
    U[1, 8] = 1
    V[5, 8] = 1  # Y12 index 1*3+2
    W[4, 8] = 1
    
    # 3. X10 * Y02 -> Z12
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1  # Z12 index 2*2+1
    
    # 4. X11 * Y12 -> Z12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        
        # Efficient check:
        # C = sum_r (U^T A) (V^T B) W^T
        # But U, V, W are matrices.
        # u_r = U[:, r] . A_flat
        # v_r = V[:, r] . B_flat
        # C += w_r * u_r * v_r
        
        # Let's do it directly for small sizes.
        A_flat = A.flatten() # i*m+j
        B_flat = B.flatten() # j*p+k
        
        for r in range(R):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            if u_val == 0 or v_val == 0:
                continue
            # W[:, r] is vector of length p*n.
            # Index k*n+i.
            w_vec = W[:, r]
            # Reshape w_vec to (p, n) ? No, index is k*n+i.
            # So w_vec[k*n+i] is the coefficient for C[i, k].
            # C[i, k] += w_vec[k*n+i] * u_val * v_val
            # This is equivalent to:
            # C += u_val * v_val * np.outer(np.zeros((n,p)), np.ones((1,1)))? No.
            # Let's just loop or use reshape.
            # w_vec is length p*n.
            # If we reshape w_vec to (p, n), w_vec_res[k, i] = w_vec[k*n+i].
            # Then C += u_val * v_val * w_vec_res.T (since C is n x p, and we want C[i, k]).
            # C[i, k] gets w_vec_res[k, i] * u_val * v_val.
            # So C += u_val * v_val * w_vec_res.T
            
            w_mat = w_vec.reshape(p, n)
            C_computed += u_val * v_val * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Try Rank 20 (2 blocks of Rank 10) if we can find it.
    # For now, we have Rank 11 blocks.
    # Let's see if we can improve Rank 11 to Rank 10.
    
    # Let's try to search for Rank 10 for 2x2x2x3.
    # If not found, use Rank 11.
    
    # Rank 11 decomposition
    U11, V11, W11 = _construct_2x2x2x3()
    
    # Build full Rank 22 decomposition
    rank = 22
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: A0 * B0
    for r in range(11):
        col = r
        # U: i*4+jj
        for i in range(2):
            for jj in range(2):
                U[i * 4 + jj, col] = U11[i*2+jj, r]
        
        # V: jj*6+kk
        for jj in range(2):
            for kk in range(3):
                V[jj * 6 + kk, col] = V11[jj*3+kk, r]
        
        # W: kk*2+i
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]
    
    # Block 2: A1 * B1
    for r in range(11):
        col = 11 + r
        # U: i*4+(2+jj)
        for i in range(2):
            for jj in range(2):
                U[i * 4 + (2 + jj), col] = U11[i*2+jj, r]
        
        # V: (2+jj)*6+kk
        for jj in range(2):
            for kk in range(3):
                V[(2 + jj) * 6 + kk, col] = V11[jj*3+kk, r]
        
        # W: kk*2+i
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W11[kk*2+i, r]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, num_trials=3):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook Rank 48
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
