import time
import random
import numpy as np
import sys

DEADLINE = 120.0
T_START = time.time()

def _remaining():
    return DEADLINE - (time.time() - T_START)

def _get_strassen_4x4x4_decomp():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    This is the tensor product of two 2x2x2 Strassen decompositions.
    """
    # Strassen 2x2x2 terms
    # M0 = (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
    # M1 = (A10+A11)B00 -> C10+=M1
    # M2 = A00(B01-B11) -> C01+=M2
    # M3 = A11(B10-B00) -> C00+=M3, C11+=M3
    # M4 = (A00+A01)B11 -> C01+=M4, C11+=M4
    # M5 = (A10-A00)(B00+B01) -> C10-=M5
    # M6 = (A01-A11)(B10+B11) -> C01-=M6, C11-=M6
    
    # We represent each term as (u_vec, v_vec, w_vec)
    # u_vec is length 4 (2x2 flattened), v_vec length 4, w_vec length 4
    terms = []
    
    # Helper to create vectors
    def vec(d):
        v = np.zeros(4)
        for idx, val in d.items():
            v[idx] = val
        return v

    # Term 0
    u0 = vec({0:1, 3:1})
    v0 = vec({0:1, 3:1})
    w0 = vec({0:1, 3:1})
    terms.append((u0, v0, w0))
    
    # Term 1
    u1 = vec({2:1, 3:1})
    v1 = vec({0:1})
    w1 = vec({2:1})
    terms.append((u1, v1, w1))
    
    # Term 2
    u2 = vec({0:1})
    v2 = vec({1:1, 3:-1})
    w2 = vec({1:1})
    terms.append((u2, v2, w2))
    
    # Term 3
    u3 = vec({3:1})
    v3 = vec({2:1, 0:-1})
    w3 = vec({0:1, 3:1})
    terms.append((u3, v3, w3))
    
    # Term 4
    u4 = vec({0:1, 1:1})
    v4 = vec({3:1})
    w4 = vec({1:1, 3:1})
    terms.append((u4, v4, w4))
    
    # Term 5
    u5 = vec({2:1, 0:-1})
    v5 = vec({0:1, 1:1})
    w5 = vec({2:-1})
    terms.append((u5, v5, w5))
    
    # Term 6
    u6 = vec({1:1, 3:-1})
    v6 = vec({2:1, 3:1})
    w6 = vec({1:-1, 3:-1})
    terms.append((u6, v6, w6))
    
    # Construct 4x4x4 decomposition via Kronecker product
    # A is 4x4, split into 2x2 blocks A_ij where i,j in {0,1}
    # B is 4x4, split into 2x2 blocks B_jk
    # C is 4x4, split into 2x2 blocks C_ik
    
    U_44 = []
    V_44 = []
    W_44 = []
    
    # There are 7x7 = 49 terms
    for t1 in terms:
        for t2 in terms:
            u1, v1, w1 = t1
            u2, v2, w2 = t2
            
            # Kronecker product of vectors
            # u = u1 x u2
            u_kron = np.kron(u1, u2)
            v_kron = np.kron(v1, v2)
            w_kron = np.kron(w1, w2)
            
            U_44.append(u_kron)
            V_44.append(v_kron)
            W_44.append(w_kron)
            
    U_44 = np.array(U_44).T # Shape (16, 49)
    V_44 = np.array(V_44).T # Shape (16, 49)
    W_44 = np.array(W_44).T # Shape (16, 49)
    
    return U_44, V_44, W_44

def _construct_candidate_decomp(rank_target=90):
    """
    Attempt to construct a rank-90 decomposition for 4x4x8.
    Strategy:
    Use the rank-49 decomposition for 4x4x4.
    The output C is 4x8. We can view it as two 4x4 matrices C_left and C_right.
    C = A @ B.
    B = [B_left, B_right].
    C_left = A @ B_left (4x4 matmul)
    C_right = A @ B_right (4x4 matmul)
    
    If we used independent rank-49 for each, total rank is 98.
    Can we share terms?
    
    Let the decomposition be sum_{r=1}^{R} u_r (A) v_r (B) w_r (C).
    For C_left, we need sum_{r} (u_r A) (v_r B_left) w_r = A B_left.
    For C_right, we need sum_{r} (u_r A) (v_r B_right) w_r = A B_right.
    
    This is equivalent to finding a decomposition of the tensor T_44x8.
    
    Let's try a specific structure:
    Take the 49 terms from 4x4x4.
    For each term r in 1..49:
       u_r is fixed (from 4x4x4 decomp)
       w_r is fixed (from 4x4x4 decomp)
       v_r is a vector of length 32 (4x8).
       
    We want:
    sum_{r=1}^{49} u_r (A) v_r (B) w_r (C) = A B
    
    This requires that for each A, B:
    (sum_r u_r (A) w_r (C)) * (v_r (B)) ... wait, the summation is over r.
    
    The tensor identity is:
    T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    If we fix U and W from the 4x4x4 decomposition (padded to 4x8?), does that work?
    In 4x4x4, U has size 16x49, V 16x49, W 16x49.
    Here U is 16xR, V is 32xR, W is 32xR.
    
    If we set R=49, we can't fit the 8-wide output easily because W has 32 entries.
    The 4x4x4 W has 16 entries (4x4). We need 32 entries (8x4).
    
    Let's try R=90.
    We can start with the 49 terms for the left half (k=0..3) and 49 terms for the right half (k=4..7).
    This gives rank 98.
    
    Can we reduce to 90?
    The "border rank" idea suggests that the 4x4x8 tensor is a degeneration of a larger tensor.
    However, a simpler structural approach:
    
    Notice that B is 4x8.
    Let B = [B1, B2].
    C = A B1 + A B2 (block concat).
    
    Actually, let's look at the known results.
    The rank of 4x4x4 is 49 (Strassen).
    The rank of 4x4x8 is at most 98 (two independent 4x4x4).
    Is it lower?
    
    There is a known result that the rank of m x n x p matmul is related to the border rank.
    For 4x4x8, the border rank is known to be 49 * (8/4) = 98? No, border rank is usually lower.
    
    Actually, let's try to solve for a rank-90 decomposition directly using optimization.
    We define U, V, W with random integer initializations, then refine.
    But random initialization is unlikely to converge to a low-rank exact solution.
    
    Alternative:
    Use the 49 terms for the first 4 columns of output.
    Use the 49 terms for the last 4 columns of output.
    Total 98.
    
    Can we merge 8 terms?
    If we can find a decomposition of rank 45 for 4x4x4, then 45*2 = 90.
    But rank of 4x4x4 is 49. It cannot be 45.
    
    So we must share terms between the two halves.
    
    Let's define the decomposition as:
    R = 90.
    U: 16x90
    V: 32x90
    W: 32x90
    
    We want to find U, V, W with integer entries.
    
    Let's try a heuristic search.
    Start with the 98-term decomposition.
    Try to merge pairs of terms.
    
    Or, better:
    Use the fact that the tensor is linear in B and C.
    
    Let's implement a generic optimizer that tries to find a rank-90 solution.
    Since exact integer solutions are hard to find via floating point optimization,
    we will use a "rounding" strategy:
    1. Initialize U, V, W with the 98-term decomposition (padded with zeros to 90? No, 98 > 90).
    2. We need to REDUCE the rank.
    
    This is a hard problem.
    
    Let's look at the "Inspiration" notes.
    "Tensor product of Strassen 2x2x2 gives exact 4x4x4 rank-49... replicating for 8-wide output yields rank 98".
    
    Is there a known rank-90 decomposition?
    The border rank of 4x4x4 is 49.
    The border rank of 4x4x8 is...
    
    Actually, the task asks for "rank-minimal exact tensor decomposition".
    If the minimal rank is 98, then 98 is the best we can do.
    The current program scores 1.0 (which corresponds to rank 112 or similar, wait).
    The current program returns rank 112 (direct tiling) or 128 (schoolbook).
    The "inspiration" says rank 98 beats schoolbook.
    So the island best might be 98.
    
    If the island best is 98, can we do better?
    The border rank of 4x4x8 is likely 98? Or lower?
    
    Actually, for rectangular matrices, the rank can sometimes be lower than the product of ranks.
    However, 4x4x8 is a "wide" matrix.
    
    Let's try to find a rank-90 decomposition by solving a system of linear equations.
    We fix U and W, and solve for V? No, it's bilinear.
    
    Let's try a different strategy:
    Use the 49 terms for 4x4x4.
    Let the decomposition for 4x4x8 be:
    Sum_{r=1}^{49} u_r A v_r B w_r C  (for left half)
    Sum_{r=1}^{49} u_r A v'_r B w'_r C (for right half)
    
    If we can find v'_r, w'_r such that they are linear combinations of the first 49 v's and w's, we might reduce the rank.
    
    Specifically, if we can express the right-half multiplication using the same U factors,
    then the rank is determined by the span of the (v, w) pairs.
    
    Let's assume the rank is 98.
    The current program has a bug or inefficiency?
    The current program returns rank 112.
    If we just return rank 98, we improve from 1.0 to 112/98 = 1.14.
    
    The "Inspiration" 3 says "Tensor product ... yields rank 98".
    So the island best is likely 98.
    
    To beat 98, we need rank < 98.
    Is rank 90 possible?
    
    Let's try to construct a rank-90 decomposition using a specific structure.
    We will use 49 terms for the left half.
    We will use 41 terms for the right half, sharing 8 terms with the left half?
    
    Actually, let's just implement the rank-98 solution first to ensure we beat 112.
    Then, if time permits, attempt a rank-90 search.
    
    Wait, the prompt says "score is the rank".
    "seed/value" where seed is the current program's rank.
    Current program rank: 112.
    If we return 98, score = 112/98 = 1.14.
    If we return 90, score = 112/90 = 1.24.
    
    Let's implement the rank-98 solution (Kronecker product of Strassen) as the baseline.
    Then, add a module to try and reduce the rank to 90 using a local search.
    
    Local Search Strategy:
    Start with the 98-term decomposition.
    Randomly pick a term and try to remove it by adjusting the others?
    This is very hard.
    
    Alternative:
    Use the known rank-49 decomposition for 4x4x4.
    The tensor is T_{i,j,k}.
    
    Let's stick to the rank-98 solution as it is a guaranteed improvement over 112.
    The "Inspiration" 1 says "Tensor product ... yields rank 98, beating schoolbook 128".
    It also says "Measured: 1 -> 1.306". This implies that rank 98 gave a score of 1.306?
    Wait. If seed=128 (schoolbook) and value=98, score=128/98=1.306.
    But the CURRENT program has score 1.0.
    This implies the seed program for THIS run has rank 112?
    112/112 = 1.0.
    
    So if I provide rank 98, my score will be 112/98 = 1.14.
    
    Can I do better than 98?
    The border rank of 4x4x8 is not definitively known to be < 98 in simple constructions.
    However, there is a construction for 4x4x4 with rank 49.
    
    Let's try to find a rank-90 decomposition.
    We will use a greedy algorithm to merge terms.
    
    Actually, let's just output the rank-98 solution. It is a solid structural change from 112.
    
    """
    if _remaining() < 10:
        return _build_schoolbook()
        
    U44, V44, W44 = _get_strassen_4x4x4_decomp()
    
    # U44: 16x49, V44: 16x49, W44: 16x49
    # We need U: 16xR, V: 32xR, W: 32xR
    
    # Left half (k=0..3)
    # U_left = U44
    # V_left = V44 (rows 0..3 of B, cols 0..3)
    # W_left = W44 (rows 0..3 of C, cols 0..3? No, W is indexed by k*n+i)
    
    # Output convention: W[k*n+i, r]
    # For 4x4x4, C is 4x4. k in 0..3, i in 0..3.
    # W44 has indices 0..15 corresponding to k*4+i.
    
    # For 4x4x8, C is 4x8. k in 0..7, i in 0..3.
    # W index is k*4+i.
    
    # Left half (k=0..3):
    # W indices 0..15.
    # Right half (k=4..7):
    # W indices 16..31.
    
    # We can construct the 98-term decomposition:
    # Terms 0..48: Left half
    # Terms 49..97: Right half
    
    R = 98
    U = np.zeros((16, R))
    V = np.zeros((32, R))
    W = np.zeros((32, R))
    
    # Left half
    U[:, 0:49] = U44
    # V for left half: B rows 0..3, cols 0..3.
    # V44 is 16x49. Index j*4+k. j in 0..3, k in 0..3.
    # In 4x4x8, B is 4x8. Index j*8+k.
    # For left half, k is 0..3. So index j*8 + k.
    # V44 index is j*4 + k.
    # We need to map V44[r, c] to V[8*j + k, r].
    
    for c in range(49):
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            new_idx = j * 8 + k
            V[new_idx, c] = V44[idx, c]
            
    # W for left half: C rows 0..3, cols 0..3.
    # W44 is 16x49. Index k*4+i.
    # In 4x4x8, W index is k*4+i.
    # For left half, k is 0..3. So index k*4+i is 0..15.
    W[0:16, 0:49] = W44
    
    # Right half (k=4..7)
    # U is the same (A is the same)
    U[:, 49:98] = U44
    
    # V for right half: B rows 0..3, cols 4..7.
    # V44 index is j*4 + k (k in 0..3).
    # In 4x4x8, index is j*8 + (4+k).
    for c in range(49):
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            new_idx = j * 8 + (4 + k)
            V[new_idx, 49 + c] = V44[idx, c]
            
    # W for right half: C rows 0..3, cols 4..7.
    # W44 index is k*4+i (k in 0..3).
    # In 4x4x8, index is (4+k)*4 + i = 16 + k*4 + i.
    W[16:32, 49:98] = W44
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), R):
        # Fallback to schoolbook if verification fails (should not happen)
        return _build_schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, rank):
    try:
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
    except Exception:
        return False
        
    n, m, p = 4, 4, 8
    rng = random.Random(42)
    for _ in range(3):
        A = np.array([[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)], dtype=np.int64)
        B = np.array([[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)], dtype=np.int64)
        C_true = A.astype(np.int64) @ B.astype(np.int64)
        C = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            C += (u_vec @ v_vec) * w_vec
        if not np.array_equal(C, C_true):
            return False
    return True

def construct():
    if _remaining() < 5.0:
        return _build_schoolbook()
        
    try:
        return _construct_candidate_decomp(98)
    except Exception as e:
        print(f"Error in construct: {e}", file=sys.stderr)
        return _build_schoolbook()
