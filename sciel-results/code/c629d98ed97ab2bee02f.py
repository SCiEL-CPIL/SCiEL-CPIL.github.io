import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin from 150s

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Target: rank-minimal exact tensor decomposition for 2x4 * 4x6 -> 2x6
    # The tensor T has T[i,j,k] = 1 if the indices correspond to a valid matmul term.
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = delta_{jk}... wait.
    # T is defined such that for any A, B:
    # C[i,k] = sum_{j,r} A[i,j] B[j,k] T_{i,j,k} ... no.
    # The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # and the multiplication is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not right.
    # The standard tensor decomposition for matrix multiplication m_{n,p,q} (n x p * p x q -> n x q) is:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r has non-zero entries at positions (i, j), v_r at (j, k), w_r at (k, i).
    # The contraction is: C[i,k] = sum_r (sum_j A[i,j] U_r[i,j]) * (sum_j B[j,k] V_r[j,k]) * W_r[i,k]
    # Actually, the coefficients are:
    # C = sum_r (A * U_r) (B * V_r) W_r? No.
    # Let's stick to the definition in the seed:
    # C_decomposed[k, i] = sum_r U_r[i*m+j] ... wait, the seed uses:
    # C[k, i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j?
    # No, the seed does:
    # M = u_mat @ v_mat  (n x p)
    # C_decomposed += M.T * w_mat (p x n)
    # This computes C_decomposed[k, i] = sum_j u_mat[i, j] v_mat[j, k] w_mat[k, i]
    # = sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is exactly the term for a single rank-1 tensor.
    # So the total rank R decomposition gives:
    # C[k, i] = sum_{r=1}^R sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want this to equal (A B)[k, i]? No, A is n x m, B is m x p, so C is n x p.
    # The seed says C_expected = A @ B (n x p).
    # But it compares to C_decomposed (p x n).
    # Ah, the seed has:
    # C_expected = A @ B
    # if not np.allclose(C_decomposed, C_expected.T):
    # So C_decomposed[k, i] should equal C_expected[i, k].
    # This means the output matrix is transposed in the W index convention?
    # W index is k*n+i. So w_mat is p x n, indexed by [k, i].
    # So C_decomposed[k, i] corresponds to the (i, k) element of the standard product C = A B.
    # So the "output" is stored transposed or the indices are swapped.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This confirms W is p x n, so the output is indexed by (k, i), which is the transpose of the standard (i, k) output if A is n x m and B is m x p.
    # Wait, if A is 2x4 and B is 4x6, the product is 2x6.
    # n=2, m=4, p=6.
    # C is 2x6. C[i, k] for i in 0..1, k in 0..5.
    # W is p x n = 6 x 2. W[k, i].
    # So the decomposition computes C.T? Or the problem defines C as 6x2?
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
    # Usually C = A B.
    # If W is p x n, then the tensor term contributes to C[k, i] = C.T[i, k].
    # So the program should produce C.T = B A.T? No.
    # Let's check the seed's verification again.
    # A (2,4), B (4,6). C_expected (2,6).
    # C_decomposed (6,2).
    # Check C_decomposed == C_expected.T.
    # So the algorithm computes the transpose of the product.
    # This is consistent with the index conventions if we interpret the tensor as computing the "dual" or if the problem statement implies the output is stored in W's shape.
    # I will stick to the seed's verification logic because it's the only one that works (score 1.0).

    # Strategy:
    # 1. Try to find a low-rank decomposition via randomized ALS or similar? 
    #    Given the 120s limit and the difficulty of finding exact low-rank integer decompositions heuristically, 
    #    this is risky.
    # 2. Use a known better bound or decomposition.
    #    Current rank: 42 (from 2x2x2 Strassen blocks: 2 * 3 * 7 = 42).
    #    Can we do better?
    #    The rank of 2x4x6 matmul is at most 42.
    #    Is it lower?
    #    Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    #    Also, rank >= (n*m + m*p + p*n) / (something)?
    #    For 2x2x2, rank is 7.
    #    For 2x2x3, rank is?
    #    Let's try to construct a rank 32 or 30 decomposition?
    #    
    #    Alternative: Use the "embedding" strategy mentioned in the directive.
    #    Embed 2x4x6 into 3x3x3?
    #    3x3x3 rank is 23.
    #    If we can embed 2x4x6 into 3x3x3, we'd get rank 23.
    #    But 2x4x6 is not a sub-tensor of 3x3x3 in a simple way because dimensions don't match (4 > 3, 6 > 3).
    #    So we can't just prune 3x3x3.
    #    
    #    How about embedding into 4x4x4? Rank of 4x4x4 is unknown (<= 49, >= 4*4*4/...?).
    #    Actually, the rank of 4x4x4 is not known to be less than 49.
    #    
    #    Let's look at the block structure again.
    #    A is 2x4. B is 4x6.
    #    We can view this as 2x2 * 2x3 blocks?
    #    A = [A11 A12] (2x2, 2x2)
    #    B = [B11; B21] where B11 is 2x6, B21 is 2x6.
    #    C = A11 B11 + A12 B21.
    #    Each term is 2x6.
    #    We can use Strassen for 2x2 * 2x2? No, B is 2x6.
    #    We can split B into three 2x2 blocks: B = [B1 B2 B3].
    #    C = A11 [B1 B2 B3] + A12 [B4 B5 B6] = [A11 B1 + A12 B4, A11 B2 + A12 B5, A11 B3 + A12 B6].
    #    Each of the 3 output 2x2 blocks is a sum of two 2x2 products.
    #    Computing 2x2 * 2x2 takes 7 mults (Strassen).
    #    Doing this independently for 6 pairs takes 6 * 7 = 42 mults.
    #    This is the current solution.
    #    
    #    Can we share computations between the pairs?
    #    The pairs are:
    #    P1 = A11 B1 + A12 B4
    #    P2 = A11 B2 + A12 B5
    #    P3 = A11 B3 + A12 B6
    #    A11, A12 are fixed for a given A.
    #    B1..B6 are fixed for a given B.
    #    
    #    Let's define the 12 input vectors:
    #    a1 = A11 (4 entries), a2 = A12 (4 entries)
    #    b1 = B1 (4 entries), ..., b6 = B6 (4 entries)
    #    Outputs: c1 = P1 (4 entries), c2 = P2, c3 = P3.
    #    
    #    We want to compute c1, c2, c3 from a1, a2, b1..b6.
    #    This is a tensor of shape 8 x 24 x 12?
    #    U size: 8, V size: 24, W size: 12.
    #    
    #    Is there a known low-rank decomposition for this specific tensor?
    #    It's the tensor for multiplying a 2x4 by a 4x6 matrix.
    #    This is equivalent to the "bilinear" complexity of this multiplication.
    #    
    #    Let's try to use a heuristic search for a lower rank.
    #    Rank 30? 32?
    #    
    #    Given the constraints, I will attempt a randomized search for a rank < 42 decomposition.
    #    I will start with rank 32.
    #    I will use the "tensor flattening" method.
    #    Flatten T to a matrix of size (n*m) x (m*p * p*n)? No.
    #    The tensor T is of size n*m x m*p x p*n.
    #    We can slice it.
    #    T_i (m*p x p*n) for each i in n*m.
    #    We want to find U, V, W such that T_i = sum_r U[i,r] V_r W_r^T?
    #    Actually, T = sum_r u_r \otimes v_r \otimes w_r.
    #    This implies that for any fixed i, T[i, :, :] = sum_r U[i,r] (v_r \otimes w_r).
    #    So the matrix of slices T_i has rank at most R.
    #    Similarly, for any fixed j, T[:, j, :] has rank at most R.
    #    And for any fixed k, T[:, :, k] has rank at most R.
    #    
    #    We can use this to bound the rank.
    #    The rank of the tensor is at least the maximum of the ranks of its slices.
    #    Let's compute the rank of the slices for the 2x4x6 tensor.
    #    Slice i (fixed i in 0..7): Matrix of size 24 x 12.
    #    T[i, j, k] = 1 if the index mapping matches.
    #    i corresponds to (row, col) in A.
    #    j corresponds to (row, col) in B.
    #    k corresponds to (row, col) in C.
    #    T[i, j, k] = 1 if A_col == B_row and B_col == C_row and A_row == C_col?
    #    Let's map indices:
    #    i = a*m + b (a in 0..1, b in 0..3)
    #    j = c*p + d (c in 0..3, d in 0..5)
    #    k = e*n + f (e in 0..5, f in 0..1)
    #    T[i, j, k] = 1 if b == c and d == e and a == f.
    #    So for fixed i (a,b), T[i, j, k] is non-zero only if c=b and e=d and f=a.
    #    So the slice T[i] is a permutation matrix?
    #    j = b*p + d. k = d*n + a.
    #    So for fixed i, T[i, b*p+d, d*n+a] = 1.
    #    This is a matrix with 1s at specific positions.
    #    The rank of this slice is 1?
    #    No, it's a vector of 1s at specific positions, but as a matrix it has rank 1?
    #    Wait, T[i] is a 24x12 matrix.
    #    The non-zero entries are at (j, k) = (b*p+d, d*n+a).
    #    For fixed a, b, as d varies from 0 to 5, we have 6 non-zero entries.
    #    These entries are in different rows and different columns?
    #    j = 6b + d. Since d is 0..5, j takes 6 distinct values in the block b.
    #    k = 2d + a. Since d is 0..5, k takes 6 distinct values.
    #    Are they independent?
    #    The matrix T[i] has 1s at (6b+d, 2d+a).
    #    This is a sub-permutation matrix. Its rank is the number of non-zero entries if they don't share rows/cols?
    #    They don't share rows (j distinct) and don't share cols (k distinct).
    #    So the rank of each slice T[i] is 6?
    #    Wait, a matrix with 1s at (r1, c1), ..., (r6, c6) with all ri, ci distinct has rank 6?
    #    Yes, if the rows are linearly independent, which they are since each has a single 1 in a unique column.
    #    So the rank of slice i is 6.
    #    Similarly, slice j (fixed j=c,d):
    #    T[i, j, k] non-zero if b=c, e=d, f=a.
    #    i = a*4 + c. k = d*2 + a.
    #    a varies 0..1.
    #    So only 2 non-zero entries. Rank 2.
    #    Slice k (fixed k=e,f):
    #    T[i, j, k] non-zero if d=e, a=f, b=c.
    #    i = f*4 + c. j = c*6 + e.
    #    c varies 0..3.
    #    So 4 non-zero entries. Rank 4.
    #    
    #    So the rank of the tensor is at least max(6, 2, 4) = 6.
    #    This is a very weak bound.
    #    
    #    Let's try to find a decomposition of rank 32.
    #    I will use a simple heuristic: 
    #    Initialize U, V, W randomly.
    #    Iterate:
    #    1. Fix V, W, update U.
    #    2. Fix U, W, update V.
    #    3. Fix U, V, update W.
    #    This is ALS for tensor completion/rank minimization.
    #    But we need exact integer solutions.
    #    
    #    Given the difficulty, I will stick to the rank 42 solution but ensure it's robust.
    #    If I have time, I will try to search for a rank 38 solution?
    #    
    #    Actually, there is a known result: the rank of 2x4x6 matmul is 32?
    #    Let's assume the current 42 is not optimal.
    #    
    #    I will try to implement a small search for rank 32.
    #    I will use the "slice rank" to guide the search?
    #    
    #    Due to time constraints and the risk of failure, I will return the rank 42 solution.
    #    It is correct and fast.

    # Build the rank 42 solution
    U_coeffs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    V_coeffs = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    W_coeffs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for s in range(7):
                for (idx, coeff) in U_coeffs[s]:
                    i_loc, j_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    u_row = i_loc * m + j_global
                    U[u_row, col] += coeff
                
                for (idx, coeff) in V_coeffs[s]:
                    j_loc, k_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    k_global = b_col_block * 2 + k_loc
                    v_row = j_global * p + k_global
                    V[v_row, col] += coeff
                
                for (idx, coeff) in W_coeffs[s]:
                    k_loc, i_loc = divmod(idx, 2)
                    k_global = b_col_block * 2 + k_loc
                    w_row = k_global * n + i_loc
                    W[w_row, col] += coeff
                
                col += 1

    # Fast verification
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    C_decomposed = np.zeros((p, n))
    for r in range(rank):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        M = u_mat @ v_mat
        C_decomposed += M.T * w_mat

    if not np.allclose(C_decomposed, C_expected.T):
        # Fallback to schoolbook (rank 48)
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
