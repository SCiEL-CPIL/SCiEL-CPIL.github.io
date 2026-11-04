import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms of the rank-7 decomposition for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) where each vector has length 4 (flattened 2x2).
    U is for A (2x2), V is for B (2x2), W is for C (2x2).
    The tensor is T[i,j,k] = sum_t U_t[i*2+j] * V_t[j*2+k] * W_t[k*2+i] = 1 for all i,j,k in {0,1}.
    """
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    # C = [[i, j], [k, l]] -> indices: i=0, j=1, k=2, l=3
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    
    # Let's verify the tensor identity: sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = 1
    # This means for each (i,j,k) in {0,1}^3, the sum should be 1.
    # Actually, the matrix multiplication tensor T for 2x2x2 has T[i,j,k] = 1 for all i,j,k.
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} is defined such that
    # C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form, the multiplication tensor is T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C
    # where the indices are (i, j, k) with i in A, j in B, k in C.
    # But the convention here is:
    # U has shape (n*m, rank) -> indices (i, j) for A
    # V has shape (m*p, rank) -> indices (j, k) for B
    # W has shape (p*n, rank) -> indices (k, i) for C (trace convention: W[k*n+i])
    
    # So the condition is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # For 2x2x2 (n=2, m=2, p=2):
    # sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
    
    # Let's construct the 7 terms based on Strassen.
    # Each Strassen product M_t is of the form (linear combo of A) * (linear combo of B).
    # And it contributes to specific entries of C with specific coefficients.
    
    # M1 = (a+d)(e+h)
    # U1 = [1, 0, 0, 1]  (a + d)
    # V1 = [1, 0, 0, 1]  (e + h)
    # Contribution to C:
    # C[0,0] += M1, C[1,1] += M1
    # So W1 should have 1 at positions corresponding to (k,i) where the contribution is +1.
    # C[0,0] corresponds to i=0, k=0 -> W index = 0*2+0 = 0
    # C[1,1] corresponds to i=1, k=1 -> W index = 1*2+1 = 3
    # So W1 = [1, 0, 0, 1]
    
    # M2 = (c+d)(e)
    # U2 = [0, 0, 1, 1]  (c + d)
    # V2 = [1, 0, 0, 0]  (e)
    # Contribution to C:
    # C[1,0] += M2
    # C[1,0] corresponds to i=1, k=0 -> W index = 0*2+1 = 1? No, W[k*n+i] = W[0*2+1] = W[1]
    # Wait, W index is k*n + i. For C[1,0], i=1, k=0. So W[0*2+1] = W[1].
    # So W2 = [0, 1, 0, 0]
    
    # M3 = a(f-h)
    # U3 = [1, 0, 0, 0]  (a)
    # V3 = [0, 1, 0, -1]  (f - h)
    # Contribution to C:
    # C[0,1] += M3
    # C[0,1] corresponds to i=0, k=1 -> W index = 1*2+0 = 2
    # So W3 = [0, 0, 1, 0]
    
    # M4 = d(g-e)
    # U4 = [0, 0, 0, 1]  (d)
    # V4 = [0, 0, 1, -1]  (g - e)  -> wait, g is index 2, e is index 0. So V4 = [-1, 0, 1, 0]?
    # V[j*p+k] for 2x2: V[0]=e, V[1]=f, V[2]=g, V[3]=h.
    # g - e = V[2] - V[0]. So V4 = [-1, 0, 1, 0].
    # Contribution to C:
    # C[0,0] += -M4, C[1,0] += M4
    # C[0,0]: i=0, k=0 -> W[0] = -1
    # C[1,0]: i=1, k=0 -> W[1] = 1
    # So W4 = [-1, 1, 0, 0]
    
    # M5 = (a+b)(h)
    # U5 = [1, 1, 0, 0]  (a + b)
    # V5 = [0, 0, 0, 1]  (h)
    # Contribution to C:
    # C[0,0] += -M5, C[0,1] += M5
    # C[0,0]: i=0, k=0 -> W[0] = -1
    # C[0,1]: i=0, k=1 -> W[2] = 1
    # So W5 = [-1, 0, 1, 0]
    
    # M6 = (c-a)(e+f)
    # U6 = [0, 0, 1, -1]  (c - a) -> wait, c is index 2, a is index 0. So U6 = [-1, 0, 1, 0].
    # V6 = [1, 1, 0, 0]  (e + f)
    # Contribution to C:
    # C[1,1] += M6
    # C[1,1]: i=1, k=1 -> W[3] = 1
    # So W6 = [0, 0, 0, 1]
    
    # M7 = (b-d)(g+h)
    # U7 = [0, 1, 0, -1]  (b - d) -> b is index 1, d is index 3. So U7 = [0, 1, 0, -1].
    # V7 = [0, 0, 1, 1]  (g + h)
    # Contribution to C:
    # C[0,0] += -M7, C[1,1] += M7
    # C[0,0]: i=0, k=0 -> W[0] = -1
    # C[1,1]: i=1, k=1 -> W[3] = 1
    # So W7 = [-1, 0, 0, 1]
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])  # M7
    ]
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomp():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen's 2x2 algorithm.
    This treats 4x4 matrices as 2x2 blocks of 2x2 matrices.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    base_terms = _strassen_2x2_terms()
    
    # Block positions for 4x4 as 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    # For block matrix multiplication C = A * B where A, B, C are 4x4:
    # C[block_i, block_j] = sum_block_k A[block_i, block_k] * B[block_k, block_j]
    # But Strassen computes 2x2 block multiplication in 7 products.
    # So we have 7 block-level products, each involving 2x2 blocks.
    # Each block product is a 2x2 * 2x2 = 2x2 multiplication, which itself has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # The 7 Strassen terms for the 2x2 block structure:
    # Let's denote the 2x2 block indices as (r, c) where r, c in {0, 1}.
    # A blocks: A[0,0], A[0,1], A[1,0], A[1,1]
    # B blocks: B[0,0], B[0,1], B[1,0], B[1,1]
    # C blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen's 7 products for the 2x2 block structure:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[1,1] - A[0,1]) * (B[1,0] + B[1,1])
    
    # Each M_t is a 2x2 matrix (block). It contributes to specific C blocks with coefficients.
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    
    # For each block-level Strassen term, we need to expand it into 7 rank-1 terms
    # for the underlying 2x2 matrix multiplication.
    
    # Let's define the block-level Strassen terms with their A and B block coefficients.
    # Each term is: (coeffs_A, coeffs_B, coeffs_C)
    # where coeffs_A is a 2x2 matrix of coefficients for the A blocks,
    # coeffs_B is a 2x2 matrix of coefficients for the B blocks,
    # coeffs_C is a 2x2 matrix of coefficients for the C blocks.
    
    block_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
        # M4
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 0], [1, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 1]])
    ]
    
    # Wait, I need to be more careful. Let me re-derive.
    # Actually, let's just iterate over the 7 block-level terms and for each,
    # iterate over the 7 inner Strassen terms for the 2x2 multiplication.
    
    # For a block-level term with A_coeffs, B_coeffs, C_coeffs:
    # The contribution is: sum_{r,s,t} A_coeffs[r,s] * B_coeffs[s,t] * C_coeffs[t,r]
    # But this is for the block level. Each "block" is a 2x2 matrix.
    # So the actual contribution involves the elements of the 2x2 blocks.
    
    # Let's think of it differently. The 4x4 matrix multiplication tensor can be
    # decomposed by first decomposing the 2x2 block structure using Strassen (7 terms),
    # and then for each block term, decomposing the 2x2 * 2x2 = 2x2 multiplication
    # using Strassen again (7 terms).
    
    # So for each of the 7 block-level terms, we have 7 inner terms, giving 49 total.
    
    # Let's construct this explicitly.
    
    # For block-level term t (0 to 6), with A_coeffs_t, B_coeffs_t, C_coeffs_t:
    # The contribution to the tensor is:
    # sum_{r,s,t_block} A_coeffs_t[r,s] * B_coeffs_t[s,t_block] * C_coeffs_t[t_block,r] * (A_block[r,s] * B_block[s,t_block] * C_block[t_block,r])
    
    # But A_block[r,s] * B_block[s,t_block] is a 2x2 matrix multiplication.
    # We can decompose this using the 7 inner Strassen terms.
    
    # Let me just build the 49 terms directly.
    
    col = 0
    for t_block in range(7):
        A_coeffs, B_coeffs, C_coeffs = block_terms[t_block]
        for t_inner in range(7):
            U_inner, V_inner, W_inner = base_terms[t_inner]
            # U_inner is for the 2x2 A block (4 elements)
            # V_inner is for the 2x2 B block (4 elements)
            # W_inner is for the 2x2 C block (4 elements)
            
            # Now, for each (r, s) in {0,1}^2, if A_coeffs[r,s] != 0,
            # we add A_coeffs[r,s] * U_inner to the corresponding part of U.
            # Similarly for V and W.
            
            for r in range(2):
                for s in range(2):
                    if A_coeffs[r][s] != 0:
                        # A block at (r, s) corresponds to rows 2*r to 2*r+1, cols 2*s to 2*s+1
                        for dr in range(2):
                            for dc in range(2):
                                u_idx = (2*r + dr) * 4 + (2*s + dc)
                                u_inner_idx = dr * 2 + dc
                                U[u_idx][col] += A_coeffs[r][s] * U_inner[u_inner_idx]
            
            for s in range(2):
                for t in range(2):
                    if B_coeffs[s][t] != 0:
                        # B block at (s, t) corresponds to rows 2*s to 2*s+1, cols 2*t to 2*t+1
                        for dr in range(2):
                            for dc in range(2):
                                v_idx = (2*s + dr) * 4 + (2*t + dc)
                                v_inner_idx = dr * 2 + dc
                                V[v_idx][col] += B_coeffs[s][t] * V_inner[v_inner_idx]
            
            for t in range(2):
                for r in range(2):
                    if C_coeffs[t][r] != 0:
                        # C block at (t, r) corresponds to rows 2*t to 2*t+1, cols 2*r to 2*r+1
                        # But W index is k*n + i, where k is col of C, i is row of C.
                        # So W index = (2*t + dc) * 4 + (2*r + dr)
                        for dr in range(2):
                            for dc in range(2):
                                w_idx = (2*t + dc) * 4 + (2*r + dr)
                                w_inner_idx = dr * 2 + dc
                                W[w_idx][col] += C_coeffs[t][r] * W_inner[w_inner_idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_443():
    """
    Schoolbook decomposition for 4x4 * 4x3 matrix multiplication.
    Rank = 4*4*3 = 48.
    """
    n, m, p = 4, 4, 3
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

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions:
    decomp1: for n x m * m x p1, with U1 (n*m x r1), V1 (m*p1 x r1), W1 (p1*n x r1)
    decomp2: for n x m * m x p2, with U2 (n*m x r2), V2 (m*p2 x r2), W2 (p2*n x r2)
    
    Result: for n x m * m x (p1+p2), with U (n*m x (r1+r2)), V (m*(p1+p2) x (r1+r2)), W ((p1+p2)*n x (r1+r2))
    
    U = [U1, U2] (horizontal concat)
    V = [V1_top, V2_top; V1_bottom, V2_bottom] -> actually V1 is (m*p1 x r1), V2 is (m*p2 x r2)
    We need V to be (m*(p1+p2) x (r1+r2)).
    The first m*p1 rows of V correspond to B's first p1 columns, and the next m*p2 rows to B's last p2 columns.
    So V = [[V1, 0], [0, V2]]
    
    W = [W1, W2] (vertical concat) -> W1 is (p1*n x r1), W2 is (p2*n x r2)
    W = [[W1, 0], [0, W2]]
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * (p1 + p2))]
    W = [[0] * r for _ in range((p1 + p2) * n)]
    
    # Copy U1 and U2
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = decomp1["U"][i][c]
        for c in range(r2):
            U[i][r1 + c] = decomp2["U"][i][c]
    
    # Copy V1 and V2
    for i in range(m * p1):
        for c in range(r1):
            V[i][c] = decomp1["V"][i][c]
    for i in range(m * p2):
        for c in range(r2):
            V[m * p1 + i][r1 + c] = decomp2["V"][i][c]
    
    # Copy W1 and W2
    for i in range(p1 * n):
        for c in range(r1):
            W[i][c] = decomp1["W"][i][c]
    for i in range(p2 * n):
        for c in range(r2):
            W[p1 * n + i][r1 + c] = decomp2["W"][i][c]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_447():
    """
    Schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    Rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook (rank 112)
    result = _schoolbook_447()
    
    # Try hybrid: Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = rank 97
    try:
        decomp_444 = _strassen_4x4_decomp()
        decomp_443 = _schoolbook_443()
        hybrid = _combine_decomps(decomp_444, decomp_443, n, m, 4, 3)
        
        # Verify
        if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            result = hybrid
    except Exception:
        pass
    
    return result
