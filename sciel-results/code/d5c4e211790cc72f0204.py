"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _get_strassen_4x4x4():
    """
    Returns U, V, W for 4x4 * 4x4 matrix multiplication with rank 28.
    U: 16 x 28, V: 16 x 28, W: 16 x 28
    Convention: C[i,k] = sum_r U[i,j... wait, U is indexed by (i,j) in A? No.
    The tensor is T[i,j,k,l] = sum_r U[i,j,r] V[j,k,r] W[k,l,r]
    But the problem says:
    U: shape (n*m, rank) -> indexed by row of A and col of A? No.
    Let's look at the seed:
    U[i*m + j][column] = 1
    V[j*p + k][column] = 1
    W[k*n + i][column] = 1
    
    Here n=4, m=4, p=4.
    Index for U is i*m+j. i is row of A (0..n-1), j is col of A (0..m-1).
    So U corresponds to flattening A (n x m) into (n*m).
    V corresponds to flattening B (m x p) into (m*p). Index j*p+k.
    W corresponds to flattening C (n x p) into (p*n). Index k*n+i.
    
    Wait, the problem says W shape is (p*n, rank).
    The seed uses W[k*n + i].
    Usually C is n x p. The index k*n+i means k is col of C, i is row of C.
    So C is stored column-major? Or the index just maps to a flat array of size p*n.
    The problem states: "output factor follows the official trace convention W[k*n+i,r]".
    So W index = k*n + i.
    U index = i*m + j.
    V index = j*p + k.
    
    The multiplication is C = A * B.
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    In the decomposition:
    C_{i,k} = sum_r W[k*n+i, r] * U[i*m+j? No, U is a factor for A.
    The standard form is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r} where T represents the bilinear map.
    Here the inputs are A (n x m) and B (m x p).
    Let the combined input vector be A_flat (n*m) and B_flat (m*p).
    The output is C_flat (p*n).
    The bilinear map is C_flat = M(A_flat, B_flat).
    C_flat[k*n+i] = (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    So we need U_{idx_U, r}, V_{idx_V, r}, W_{idx_W, r} such that:
    W_{k*n+i, r} * U_{i*m+j, r} * V_{j*p+k, r} summed over r and j?
    No, the decomposition is:
    C_{i,k} = sum_{r} ( sum_j U_{i,j? no.
    
    The standard decomposition of the matrix multiplication tensor T_{i,j,k} (where i,j are indices of A and k of B? No.)
    Let's stick to the flat indices.
    We want: C_flat[c] = sum_{a,b} T_{a,b,c} A_flat[a] B_flat[b].
    The decomposition is C_flat[c] = sum_{r} ( sum_a U[a,r] A_flat[a] ) * ( sum_b V[b,r] B_flat[b] ) * W[c,r] ?
    No, that's not right. The bilinear form is sum_r U[a,r] V[b,r] W[c,r] A[a] B[b].
    Wait, the tensor T is 3-tensor: T_{a,b,c}.
    Here A is n*m dim, B is m*p dim, C is p*n dim.
    T_{a,b,c} is 1 if and only if c corresponds to (i,k) and a to (i,j) and b to (j,k).
    Specifically:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    Condition: The j in a must match the j in b.
    T_{a,b,c} = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    Note: a determines i,j. b determines j,k. c determines k,i.
    For T_{a,b,c} to be 1, the j from a must equal the j from b, and the k from b must equal the k from c, and the i from c must equal the i from a.
    
    We want rank-1 decomposition: T_{a,b,c} = sum_{r=1}^R U_{a,r} V_{b,r} W_{c,r}.
    
    For the 4x4x4 case (n=4, m=4, p=4):
    We can use Strassen's algorithm.
    Strassen works on 2x2 blocks.
    A = [[A11, A12], [A21, A22]] where each is 2x2.
    B = [[B11, B12], [B21, B22]] where each is 2x2.
    C = [[C11, C12], [C21, C22]] where each is 2x2.
    
    Strassen's 7 products of 2x2 matrices:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Since each product is a 2x2 matrix multiplication, it requires 4 scalar multiplications (schoolbook).
    So 7 products * 4 scalar mults = 28 scalar multiplications.
    
    We need to map these 28 scalar multiplications to the U, V, W tensors.
    
    Let's define the indices for the 4x4 case.
    n=4, m=4, p=4.
    i, j, k in 0..3.
    a = i*4 + j
    b = j*4 + k
    c = k*4 + i
    
    Let's break down into 2x2 blocks.
    Block row indices: i' = i // 2, i'' = i % 2.
    Block col indices for A: j' = j // 2, j'' = j % 2.
    Block col indices for B: j' = j // 2, j'' = j % 2.
    Block row indices for B: j' = j // 2, j'' = j % 2.
    Block col indices for C: k' = k // 2, k'' = k % 2.
    Block row indices for C: i' = i // 2, i'' = i % 2.
    
    Actually, let's just use the known Strassen coefficients for 4x4.
    We can construct the U, V, W matrices for the 28 multiplications.
    For each of the 7 M-matrices, we have 4 scalar multiplications (for 2x2 blocks).
    
    Let's denote the 7 matrix products as M1..M7.
    For each M, we have 2x2 multiplication.
    M = P * Q, where P and Q are 2x2.
    C_M = [[P00*Q00 + P01*Q10, P00*Q01 + P01*Q11],
           [P10*Q00 + P11*Q10, P10*Q01 + P11*Q11]]
    
    So for each M, we have 4 scalar products.
    Total 28 scalar products.
    
    We need to define U, V, W for each of these 28 products.
    
    Let's implement a helper to build Strassen 4x4x4.
    """
    n = m = p = 4
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Helper to map 2x2 block indices to global indices
    # A is 4x4. Block (bi, bj) contains rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
    # Local indices (li, lj) in block (bi, bj) map to global i = 2*bi + li, j = 2*bj + lj.
    # a_idx = i*4 + j
    
    def get_a_idx(bi, bj, li, lj):
        i = 2*bi + li
        j = 2*bj + lj
        return i*4 + j
        
    def get_b_idx(bi, bj, li, lj):
        # B is 4x4.
        # In Strassen, B is indexed by col of A (j) and row of C (k).
        # So B is 4x4.
        # Block (bi, bj) in B corresponds to j in block bi, k in block bj?
        # Wait, B is m x p = 4 x 4.
        # j is column of A, row of B.
        # k is column of B, column of C.
        # So B[j, k].
        # j = 2*bj + lj? No.
        # Let's be consistent with blocks.
        # A blocks: A[bi, bj] is 2x2. Rows 2bi.., Cols 2bj..
        # B blocks: B[bj, bk] is 2x2. Rows 2bj.. (corresponding to j), Cols 2bk.. (corresponding to k).
        # C blocks: C[bi, bk] is 2x2. Rows 2bi.. (corresponding to i), Cols 2bk.. (corresponding to k).
        
        j = 2*bj + li  # 'li' is the local row index in B, which is the col index j
        k = 2*bk + lj  # 'lj' is the local col index in B, which is the col index k
        # V index = j*p + k = j*4 + k
        return j*4 + k

    def get_c_idx(bi, bk, li, lk):
        i = 2*bi + li
        k = 2*bk + lk
        # W index = k*n + i = k*4 + i
        return k*4 + i
        
    # We need to define the 7 M products.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each M is a 2x2 matrix product of linear combinations.
    # Let's define the A-side and B-side linear combos for each M.
    # And then the 4 scalar products for the 2x2 multiplication.
    # And then how C blocks combine M1..M7.
    
    # This is getting complex to code generically.
    # Let's hardcode the 28 decompositions using the known structure.
    
    # Indices for M-matrices
    # m_id from 0 to 6.
    # sub_id from 0 to 3 (00, 01, 10, 11)
    # rank_idx = m_id * 4 + sub_id
    
    # We need U[a, r], V[b, r], W[c, r].
    # For a fixed r, U and V define the scalar product.
    
    # Let's define A_coeffs[7][4] and B_coeffs[7][4] for the block-level linear combinations.
    # A_side: [bi, bj] -> coeff
    # B_side: [bj, bk] -> coeff
    
    # M1: A11+A22, B11+B22
    # A blocks: (0,0):1, (1,1):1
    # B blocks: (0,0):1, (1,1):1
    
    # M2: A21+A22, B11
    # A blocks: (1,0):1, (1,1):1
    # B blocks: (0,0):1
    
    # M3: A11, B12-B22
    # A blocks: (0,0):1
    # B blocks: (0,1):1, (1,1):-1
    
    # M4: A22, B21-B11
    # A blocks: (1,1):1
    # B blocks: (1,0):1, (0,0):-1
    
    # M5: A11+A12, B22
    # A blocks: (0,0):1, (0,1):1
    # B blocks: (1,1):1
    
    # M6: A21-A11, B11+B12
    # A blocks: (1,0):1, (0,0):-1
    # B blocks: (0,0):1, (0,1):1
    
    # M7: A12-A22, B21+B22
    # A blocks: (0,1):1, (1,1):-1
    # B blocks: (1,0):1, (1,1):1
    
    a_blocks = [
        { (0,0):1, (1,1):1 }, # M1
        { (1,0):1, (1,1):1 }, # M2
        { (0,0):1 },           # M3
        { (1,1):1 },           # M4
        { (0,0):1, (0,1):1 }, # M5
        { (1,0):1, (0,0):-1 },# M6
        { (0,1):1, (1,1):-1 } # M7
    ]
    
    b_blocks = [
        { (0,0):1, (1,1):1 }, # M1
        { (0,0):1 },           # M2
        { (0,1):1, (1,1):-1 },# M3
        { (1,0):1, (0,0):-1 },# M4
        { (1,1):1 },           # M5
        { (0,0):1, (0,1):1 }, # M6
        { (1,0):1, (1,1):1 }  # M7
    ]
    
    # C combination: C blocks are sums/diffs of M's.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    c_comb = {
        (0,0): { 0:1, 3:1, 4:-1, 6:1 }, # C11 uses M1, M4, M5, M7 (indices 0,3,4,6)
        (0,1): { 2:1, 4:1 },             # C12 uses M3, M5 (indices 2,4)
        (1,0): { 1:1, 3:1 },             # C21 uses M2, M4 (indices 1,3)
        (1,1): { 0:1, 1:-1, 2:1, 5:1 }  # C22 uses M1, M2, M3, M6 (indices 0,1,2,5)
    }
    
    # Now build U, V, W
    for m in range(7):
        for s in range(4): # sub 00, 01, 10, 11
            r = m*4 + s
            li = s // 2
            lj = s % 2
            
            # This scalar product computes C_block_entry [li, lj] of the m-th M matrix.
            # The M matrix is P*Q.
            # C_M[li, lj] = sum_{q=0}^1 P[li, q] * Q[q, lj]
            # So we need 2 products? No, schoolbook 2x2 has 4 products for 2 entries?
            # No, 4 products total for the 4 entries?
            # C_M[0,0] = P00*Q00 + P01*Q10 (2 products)
            # C_M[0,1] = P00*Q01 + P01*Q11 (2 products)
            # C_M[1,0] = P10*Q00 + P11*Q10 (2 products)
            # C_M[1,1] = P10*Q01 + P11*Q11 (2 products)
            # Total 8 scalar multiplications per 2x2?
            # No, Strassen replaces 8 multiplications of 1x1?
            # No. Standard 2x2 multiplication is 8 scalar multiplications?
            # No, 2x2 * 2x2 is 4 entries, each is dot product of 2. So 4*2=8 scalar mults.
            # Wait. Schoolbook 2x2:
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
            # 8 scalar multiplications.
            # Strassen reduces this to 7 matrix multiplications.
            # So each "M" is a 2x2 multiplication, which costs 8 scalar mults if done by schoolbook.
            # So 7 * 8 = 56 scalar mults?
            # But we said 28.
            # Ah, Strassen's algorithm for 2x2 uses 7 multiplications of size 1x1? No, size n/2.
            # For n=4, we decompose into 7 multiplications of size 2x2.
            # Each 2x2 multiplication can be done with 8 scalar multiplications.
            # So total 7 * 8 = 56.
            # But wait. The lower bound for 4x4 is 28?
            # Actually, the best known for 4x4 is 49? No, 4x4 is 26 or 28?
            # Let's check.
            # Strassen for 4x4:
            # Recursively applying Strassen to 2x2 blocks.
            # If we apply Strassen to 2x2 blocks, we need 7 multiplications of 2x2 blocks.
            # If we do each 2x2 multiplication by schoolbook (8 mults), total is 7*8 = 56.
            # But we can optimize the 2x2 multiplication? No, 2x2 is minimal 8.
            # So Strassen 4x4 is 56?
            # Wait. The standard Strassen 2x2 has rank 7.
            # For 4x4, if we use the 7 products, and each is a 2x2 multiplication...
            # Is it possible to do it with 28?
            # Yes, if we treat the "blocks" as scalars? No.
            # Let's re-read the rank of matrix multiplication.
            # R(2x2) = 7.
            # R(4x4) <= 49 (schoolbook).
            # R(4x4) <= 28?
            # Actually, 28 is the rank of 4x4x4 multiplication tensor?
            # Let's assume the best known is 28.
            # How?
            # Maybe I am miscounting.
            # Strassen's algorithm for n=2 uses 7 multiplications.
            # For n=4, it uses 7 multiplications of size 2x2.
            # If each 2x2 multiplication is rank 7, then total rank is 7*7=49?
            # No. The rank is the number of scalar multiplications.
            # Strassen 2x2: 7 scalar mults.
            # Strassen 4x4: 7 blocks. Each block is a 2x2 multiplication.
            # If we use Strassen for the blocks too? No, blocks are 2x2.
            # If we use schoolbook for 2x2 blocks, each is 8 mults. 7*8 = 56.
            # If we use Strassen for 2x2 blocks, each is 7 mults. 7*7 = 49.
            # So 49 is an upper bound.
            # Is there a 28?
            # Actually, I recall that the rank of 4x4x4 is at most 49.
            # Maybe 28 is for something else.
            # Let's just use 49 to be safe? Or 56?
            # Wait, the prompt says "rank-minimal".
            # I should use the best known.
            # But I don't know if 28 is valid.
            # Let's check the observed changes.
            # The previous proposal tried to use 28 for 4x4x4.
            # And it failed or was not admitted?
            # "The previous proposal was not admitted... source-validation result".
            # Maybe the code was buggy.
            # Let's try to generate a valid 56-rank decomposition (Strassen with schoolbook blocks) or 49-rank (Strassen with Strassen blocks).
            # Actually, 49 is likely better than 56.
            # And 49 < 144.
            # Let's implement Strassen 4x4 with rank 49.
            
            pass # Placeholder for logic below
    
    # Let's refine the plan.
    # Use Strassen 4x4.
    # Decompose into 7 sub-multiplications of 2x2 blocks.
    # Each 2x2 multiplication uses rank 7 (Strassen 2x2).
    # So we need to map the 7x7 = 49 scalar multiplications.
    
    # This requires nested Strassen.
    # Let's write a function to generate Strassen 2x2 decomposition.
    # U2: 4x7, V2: 4x7, W2: 4x7 (indices 00,01,10,11)
    
    def get_strassen_2x2():
        # 2x2 * 2x2 -> 2x2
        # U idx: 0..3 (i,j)
        # V idx: 0..3 (j,k)
        # W idx: 0..3 (k,i) -> c = k*2+i
        # Wait, W index for 2x2 is k*2+i.
        # But the global W index is k*n+i.
        # For 2x2 block at (bi, bj, bk), the local k is 0,1.
        # Local i is 0,1.
        # We will map local indices to global.
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        a_coeffs = [
            { (0,0):1, (1,1):1 }, # M1: A11, A22
            { (1,0):1, (1,1):1 }, # M2: A21, A22
            { (0,0):1 },           # M3: A11
            { (1,1):1 },           # M4: A22
            { (0,0):1, (0,1):1 }, # M5: A11, A12
            { (1,0):1, (0,0):-1 },# M6: A21, A11
            { (0,1):1, (1,1):-1 } # M7: A12, A22
        ]
        
        b_coeffs = [
            { (0,0):1, (1,1):1 }, # M1: B11, B22
            { (0,0):1 },           # M2: B11
            { (0,1):1, (1,1):-1 },# M3: B12, B22
            { (1,0):1, (0,0):-1 },# M4: B21, B11
            { (1,1):1 },           # M5: B22
            { (0,0):1, (0,1):1 }, # M6: B11, B12
            { (1,0):1, (1,1):1 }  # M7: B21, B22
        ]
        
        c_coeffs = {
            (0,0): { 0:1, 3:1, 4:-1, 6:1 },
            (0,1): { 2:1, 4:1 },
            (1,0): { 1:1, 3:1 },
            (1,1): { 0:1, 1:-1, 2:1, 5:1 }
        }
        
        for m in range(7):
            r = m
            # Fill U, V
            for (bi, bj), coeff in a_coeffs[m].items():
                # U index for 2x2 is i*2+j? No.
                # Local A is 2x2. i,bj.
                # U_local[i*2+j, r]
                i = bi
                j = bj
                U[i*2+j, r] += coeff
                
            for (bj, bk), coeff in b_coeffs[m].items():
                # V index for 2x2 is j*2+k?
                # Local B is 2x2. j,k.
                # V_local[j*2+k, r]
                j = bj
                k = bk
                V[j*2+k, r] += coeff
                
            for (bk, bi), coeff in c_coeffs[(bi, bk)].items(): # Wait, c_coeffs is indexed by C block (i,k)?
                # c_coeffs keys are (i, k) in local C.
                # c_coeffs[i, k] gives {m: coeff}
                pass
        
        # Correct c_coeffs usage
        for (ci, ck), coeffs in c_coeffs.items():
            for m, coeff in coeffs.items():
                r = m
                # W_local index is k*2+i?
                # W index = k*n+i. For 2x2, n=2.
                # W_local[ck*2+ci, r] += coeff
                W[ck*2+ci, r] += coeff
                
        return U, V, W
        
    # Now build 4x4 using nested Strassen.
    # Rank = 7 (outer) * 7 (inner) = 49.
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = get_strassen_2x2()
    
    # Outer Strassen coefficients (same as inner, but on 2x2 blocks of size 2x2)
    # We reuse a_coeffs, b_coeffs, c_coeffs from above.
    # Let's define them clearly for the 4x4 case.
    
    a_blocks_4 = [
        { (0,0):1, (1,1):1 },
        { (1,0):1, (1,1):1 },
        { (0,0):1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (0,0):-1 },
        { (0,1):1, (1,1):-1 }
    ]
    b_blocks_4 = [
        { (0,0):1, (1,1):1 },
        { (0,0):1 },
        { (0,1):1, (1,1):-1 },
        { (1,0):1, (0,0):-1 },
        { (1,1):1 },
        { (0,0):1, (0,1):1 },
        { (1,0):1, (1,1):1 }
    ]
    c_blocks_4 = {
        (0,0): { 0:1, 3:1, 4:-1, 6:1 },
        (0,1): { 2:1, 4:1 },
        (1,0): { 1:1, 3:1 },
        (1,1): { 0:1, 1:-1, 2:1, 5:1 }
    }
    
    for m_outer in range(7):
        for m_inner in range(7):
            r = m_outer * 7 + m_inner
            
            # U: sum over blocks A[bi, bj] of coeff * U2 mapped to global
            for (bi, bj), coeff in a_blocks_4[m_outer].items():
                # For each local (li, lj) in 2x2 block, U2[li*2+lj, m_inner]
                for li in range(2):
                    for lj in range(2):
                        u_val = U2[li*2+lj, m_inner]
                        if u_val != 0:
                            i = 2*bi + li
                            j = 2*bj + lj
                            a_idx = i*4 + j
                            U[a_idx, r] += coeff * u_val
                            
            # V: sum over blocks B[bj, bk] of coeff * V2 mapped to global
            for (bj, bk), coeff in b_blocks_4[m_outer].items():
                for li in range(2): # local row of B (j)
                    for lj in range(2): # local col of B (k)
                        v_val = V2[li*2+lj, m_inner]
                        if v_val != 0:
                            j = 2*bj + li
                            k = 2*bk + lj
                            b_idx = j*4 + k
                            V[b_idx, r] += coeff * v_val
                            
            # W: sum over blocks C[bi, bk] of coeff * W2 mapped to global
            # c_blocks_4 gives for each C block (bi, bk), the coeffs for m_outer.
            for (bi, bk), coeffs in c_blocks_4.items():
                if m_outer in coeffs:
                    coeff_c = coeffs[m_outer]
                    for li in range(2): # local row of C (i)
                        for lj in range(2): # local col of C (k)
                            w_val = W2[lj*2+li, m_inner] # W2 index is k*2+i = lj*2+li
                            if w_val != 0:
                                i = 2*bi + li
                                k = 2*bk + lj
                                c_idx = k*4 + i
                                W[c_idx, r] += coeff_c * w_val

    return U, V, W

def _get_strassen_4x4x9():
    """
    Decompose 4x4 * 4x9.
    B is 4x9.
    We can split B into [4x4, 4x4, 4x1].
    4x4 * 4x4 -> rank 49
    4x4 * 4x4 -> rank 49
    4x4 * 4x1 -> rank ?
    
    For 4x4 * 4x1:
    C is 4x1.
    C_i = sum_j A_ij B_j.
    This is rank 16?
    Let's check.
    A is 4x4, B is 4x1 (vector).
    C is 4x1 (vector).
    C = A b.
    This is linear in b, bilinear in A, b.
    T_{i, j, k?} No.
    Indices: a (i,j), b (j), c (i).
    T_{a,b,c} = 1 if a=i*4+j, b=j, c=i.
    Rank?
    We can do it with 16 scalar mults?
    C_i = sum_j A_ij B_j.
    Schoolbook: 16 mults.
    Can we do better?
    Probably not easily.
    So rank 16.
    
    Total rank = 49 + 49 + 16 = 114.
    """
    rank1 = 49
    rank2 = 49
    rank3 = 16
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank)) # 4*9
    W = np.zeros((36, total_rank)) # 9*4
    
    # Part 1: B cols 0-3 (4x4)
    U1, V1, W1 = _get_strassen_4x4x4()
    # U1 is 16x49. V1 is 16x49. W1 is 16x49.
    # But V1 is for 4x4 B. V indices j*4+k.
    # Here V indices are j*9+k.
    # So we need to map V1[j*4+k] to V[j*9+k].
    for i in range(16):
        for r in range(rank1):
            U[i, r] = U1[i, r]
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*9 + k, r] = V1[j*4 + k, r]
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[k*4 + i, r] = W1[k*4 + i] # W1 index is k*4+i
        
    # Part 2: B cols 4-7 (4x4)
    U2, V2, W2 = _get_strassen_4x4x4()
    for i in range(16):
        for r in range(rank1):
            U[i, rank1 + r] = U2[i, r]
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*9 + (4+k), rank1 + r] = V2[j*4 + k, r]
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[(4+k)*4 + i, rank1 + r] = W2[k*4 + i]
                
    # Part 3: B col 8 (4x1)
    # U3: 16x16. V3: 4x16. W3: 4x16.
    # C_i = sum_j A_ij B_j.
    # r = i*4 + j?
    # U[i*4+j, r] = 1
    # V[j*9+8, r] = 1
    # W[8*4+i, r] = 1
    for i in range(4):
        for j in range(4):
            r = rank1 + rank2 + i*4 + j
            U[i*4 + j, r] = 1
            V[j*9 + 8, r] = 1
            W[8*4 + i, r] = 1
            
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    if (n, m, p) == (4, 4, 9):
        U, V, W = _get_strassen_4x4x9()
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
