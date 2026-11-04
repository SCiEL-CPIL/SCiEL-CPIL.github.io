import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    # Check if the decomposition exactly reproduces the matrix multiplication tensor
    # T[i, j, k] = 1 if the indices match the matmul structure, 0 otherwise.
    # The tensor is defined by C[i, k] = sum_j A[i, j] B[j, k].
    # The rank-1 term r contributes U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
    # We need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{j, j} (which is 1 if j matches? No.)
    # The tensor M_{n,m,p} has entries T_{i,j,k} = 1 if the "inner" index j of A and B is the same.
    # In the decomposition, the term r has indices i_u, j_u for U; i_v, j_v for V; i_w, j_w for W.
    # U index is i_A, j_A. V index is j_B, k_B. W index is k_C, i_C.
    # The product is non-zero only if j_A == j_B (inner index match) and i_A == i_C and k_B == k_C.
    # So for a fixed output (i, k) and inputs (i, j) and (j, k), the coefficient should be 1.
    # Let's check the coefficient of A[i, j] B[j, k] in the computed C[i, k].
    # Computed C[i, k] = sum_r (sum_{j'} U[i*m+j', r] A[i, j']) * (sum_{j''} V[j''*p+k, r] B[j'', k]) * W[k*n+i, r]
    # This is not quite right. The bilinear form is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k])?
    # No, the standard form is:
    # C[i, k] = sum_r (sum_{j} U[i*m+j, r] A[i, j]) * (sum_{j} V[j*p+k, r] B[j, k]) * W[k*n+i, r]
    # Wait, the W factor is just a scalar multiplier for the term r?
    # Yes, the tensor is sum_r u_r \otimes v_r \otimes w_r.
    # The value is sum_r (u_r^T A) (v_r^T B) w_r.
    # Here u_r is a vector of length n*m, v_r is length m*p, w_r is length p*n.
    # u_r has a 1 at (i, j) if U[i*m+j, r]=1.
    # v_r has a 1 at (j, k) if V[j*p+k, r]=1.
    # w_r has a 1 at (k, i) if W[k*n+i, r]=1.
    
    # For the schoolbook decomposition:
    # Term r corresponds to (i, j, k).
    # u_r has 1 at (i, j).
    # v_r has 1 at (j, k).
    # w_r has 1 at (k, i).
    # Product: A[i, j] * B[j, k] * 1.
    # Summing over all r, we get sum_{j} A[i, j] B[j, k] for C[i, k].
    # This is correct.
    
    # To verify, we can check if the tensor defined by U, V, W matches the identity tensor.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should be 1 if the indices are consistent?
    # No, T[i, j, k] is the coefficient of A[i, j] B[j, k] in the bilinear form.
    # In the schoolbook case, T[i, j, k] = 1 for all i, j, k?
    # No, the tensor M_{n,m,p} is defined such that T(i, j, k) = 1 if the indices match the matmul flow.
    # Actually, the tensor is fixed. The decomposition must match it.
    # The tensor is: M = sum_{i, j, k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ?
    # No, the indices are:
    # A has indices (i, j). B has indices (j, k). C has indices (i, k).
    # The tensor is in A^* \otimes B^* \otimes C^*.
    # The entry T_{(i,j), (j,k), (k,i)} is 1.
    # So for any i, j, k, the entry at (i, j, j, k, k, i) is 1?
    # The indices for U are (i, j). For V are (j, k). For W are (k, i).
    # So T[i, j, j, k, k, i] = 1.
    # In our flattened indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor value at (u, v, w) is 1 if the indices are consistent.
    # Consistency: j_U == j_V and k_V == k_W and i_W == i_U.
    # Let's check the sum: sum_r U[u, r] V[v, r] W[w, r].
    # For schoolbook, r is (i, j, k).
    # U[u, r] is 1 if u = i*m + j and r = (i, j, k).
    # V[v, r] is 1 if v = j*p + k and r = (i, j, k).
    # W[w, r] is 1 if w = k*n + i and r = (i, j, k).
    # So the product is 1 only if u, v, w correspond to the same (i, j, k).
    # So T[u, v, w] = 1 if u, v, w are consistent, 0 otherwise.
    # This is the correct tensor.
    
    # We can verify by checking a few random points.
    # But for exactness, we should check all points if possible.
    # 125 * 125 * 125 = 2 million points. Feasible.
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Check consistency
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                
                expected = 1.0 if (j_u == j_v and k_v == k_w and i_w == i_u) else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[u, r] * V[v, r] * W[w, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen formulas for C = AB:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 (idx 0), B11 (idx 3)
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 (idx 0), C11 (idx 3)
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10 (idx 2), A11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 (idx 0)
    V[0][1] = 1
    # W: C10 (idx 2)
    W[2][1] = 1
    
    # M3: A00(B10-B11)
    # U: A00 (idx 0)
    U[0][2] = 1
    # V: B10 (idx 2), -B11 (idx 3)
    V[2][2] = 1
    V[3][2] = -1
    # W: C01 (idx 1)
    W[1][2] = 1
    
    # M4: A11(B01-B10)
    # U: A11 (idx 3)
    U[3][3] = 1
    # V: B01 (idx 1), -B10 (idx 2)
    V[1][3] = 1
    V[2][3] = -1
    # W: C00 (idx 0), C10 (idx 2)
    W[0][3] = 1
    W[2][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00 (idx 0), A01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # V: B11 (idx 3)
    V[3][4] = 1
    # W: C00 (idx 0), C01 (idx 1)
    W[0][4] = 1
    W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (idx 2), -A00 (idx 0)
    U[2][5] = 1
    U[0][5] = -1
    # V: B00 (idx 0), B01 (idx 1)
    V[0][5] = 1
    V[1][5] = 1
    # W: C11 (idx 3)
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (idx 1), -A11 (idx 3)
    U[1][6] = 1
    U[3][6] = -1
    # V: B10 (idx 2), B11 (idx 3)
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 (idx 0), C11 (idx 3)
    W[0][6] = 1
    W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition():
    """
    Construct a block decomposition for 5x5x5.
    Partition indices 0..4 into blocks:
    B0: {0, 1} (size 2)
    B1: {2, 3} (size 2)
    B2: {4}    (size 1)
    
    Total rank = sum over block pairs (i, j) of rank(i, j)
    Where rank(i, j) is the rank needed to multiply a (size_i x size_j) block of A by (size_j x size_k) block of B?
    No, the blocks are aligned.
    A is 5x5, B is 5x5, C is 5x5.
    We partition rows/cols of A, B, C into 3 blocks of sizes 2, 2, 1.
    A = [A00 A01 A02]
        [A10 A11 A12]
        [A20 A21 A22]
    B = [B00 B01 B02]
        [B10 B11 B12]
        [B20 B21 B22]
    C = [C00 C01 C02]
        [C10 C11 C12]
        [C20 C21 C22]
    
    C_ab = sum_k A_ak B_kb
    
    For each pair of output blocks (a, b), we need to compute C_ab.
    C_ab = A_a0 B_0b + A_a1 B_1b + A_a2 B_2b
    
    This is a sum of products of blocks.
    The tensor is the sum of tensors for each term.
    We can decompose each term A_ak B_kb separately.
    Term (a, k, b) involves A block (a, k) of size s_a x s_k, B block (k, b) of size s_k x s_b, contributing to C block (a, b) of size s_a x s_b.
    
    Wait, the tensor decomposition is for the whole matmul.
    The rank is the sum of the ranks of the decompositions of the individual "block matmuls" if we can separate them?
    Yes, because the indices (i, j, k) in the original tensor map to block indices (a, k, b) and local indices.
    If we use a decomposition for the block matmul A_ak * B_kb -> C_ab, it has rank R_akb.
    The total rank is sum_{a, k, b} R_akb.
    
    Let's calculate R_akb for each combination of block sizes.
    Sizes: 2, 2, 1.
    
    Pairs (a, k, b):
    a=0 (size 2), k=0 (size 2), b=0 (size 2): 2x2 * 2x2 -> 2x2. Rank 7 (Strassen).
    a=0 (2), k=0 (2), b=1 (2): 2x2 * 2x2 -> 2x2. Rank 7.
    a=0 (2), k=0 (2), b=2 (1): 2x2 * 2x1 -> 2x1. Rank 4 (schoolbook 2*2*1=4).
    a=0 (2), k=1 (2), b=0 (2): 2x2 * 2x2 -> 2x2. Rank 7.
    a=0 (2), k=1 (2), b=1 (2): 2x2 * 2x2 -> 2x2. Rank 7.
    a=0 (2), k=1 (2), b=2 (1): 2x2 * 2x1 -> 2x1. Rank 4.
    a=0 (2), k=2 (1), b=0 (2): 2x1 * 1x2 -> 2x2. Rank 4 (schoolbook 2*1*2=4).
    a=0 (2), k=2 (1), b=1 (2): 2x1 * 1x2 -> 2x2. Rank 4.
    a=0 (2), k=2 (1), b=2 (1): 2x1 * 1x1 -> 2x1. Rank 2.
    
    a=1 (2), k=0 (2), b=0 (2): 7
    a=1 (2), k=0 (2), b=1 (2): 7
    a=1 (2), k=0 (2), b=2 (1): 4
    a=1 (2), k=1 (2), b=0 (2): 7
    a=1 (2), k=1 (2), b=1 (2): 7
    a=1 (2), k=1 (2), b=2 (1): 4
    a=1 (2), k=2 (1), b=0 (2): 4
    a=1 (2), k=2 (1), b=1 (2): 4
    a=1 (2), k=2 (1), b=2 (1): 2
    
    a=2 (1), k=0 (2), b=0 (2): 1x2 * 2x2 -> 1x2. Rank 4.
    a=2 (1), k=0 (2), b=1 (2): 1x2 * 2x2 -> 1x2. Rank 4.
    a=2 (1), k=0 (2), b=2 (1): 1x2 * 2x1 -> 1x1. Rank 2.
    a=2 (1), k=1 (2), b=0 (2): 4
    a=2 (1), k=1 (2), b=1 (2): 4
    a=2 (1), k=1 (2), b=2 (1): 2
    a=2 (1), k=2 (1), b=0 (2): 1x1 * 1x2 -> 1x2. Rank 2.
    a=2 (1), k=2 (1), b=1 (2): 1x1 * 1x2 -> 1x2. Rank 2.
    a=2 (1), k=2 (1), b=2 (1): 1x1 * 1x1 -> 1x1. Rank 1.
    
    Total Rank =
    4 (from a=0, k=0, b=0,1: 7+7=14? No.
    Let's sum carefully.
    
    a=0:
      k=0: b=0(7), b=1(7), b=2(4) -> 18
      k=1: b=0(7), b=1(7), b=2(4) -> 18
      k=2: b=0(4), b=1(4), b=2(2) -> 10
      Sum a=0: 46
    
    a=1:
      k=0: 18
      k=1: 18
      k=2: 10
      Sum a=1: 46
    
    a=2:
      k=0: b=0(4), b=1(4), b=2(2) -> 10
      k=1: b=0(4), b=1(4), b=2(2) -> 10
      k=2: b=0(2), b=1(2), b=2(1) -> 5
      Sum a=2: 25
    
    Total = 46 + 46 + 25 = 117.
    
    This matches the inspiration finding.
    """
    n, m, p = DIMENSIONS
    # Block sizes
    sizes = [2, 2, 1]
    num_blocks = 3
    
    # Precompute offsets
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    # Total rank
    total_rank = 0
    # We need to store the decomposition for each block interaction
    # To assemble the final U, V, W matrices, we need to map local indices to global indices.
    
    # Let's build the matrices directly.
    # First, determine the total rank to allocate columns.
    # We'll use a list of (U_block, V_block, W_block, u_offset, v_offset, w_offset)
    
    block_decomps = []
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa = sizes[a]
                sk = sizes[k]
                sb = sizes[b]
                
                # Get decomposition for sa x sk * sk x sb -> sa x sb
                # We need a function to get the decomposition for arbitrary sizes.
                # For now, we'll use schoolbook for non-2x2x2 cases, and Strassen for 2x2x2.
                
                if sa == 2 and sk == 2 and sb == 2:
                    decomp = _strassen_2x2()
                else:
                    # Schoolbook
                    decomp = _schoolbook_for(sa, sk, sb)
                
                # Store with offsets
                # U block corresponds to A block (a, k). Rows are local i in [0, sa), local j in [0, sk).
                # Global U index: (offsets[a] + i) * m + (offsets[k] + j)
                # V block corresponds to B block (k, b). Rows are local j in [0, sk), local k in [0, sb).
                # Global V index: (offsets[k] + j) * p + (offsets[b] + k)
                # W block corresponds to C block (a, b). Rows are local k in [0, sb), local i in [0, sa).
                # Global W index: (offsets[b] + k) * n + (offsets[a] + i)
                
                block_decomps.append({
                    "U": decomp["U"],
                    "V": decomp["V"],
                    "W": decomp["W"],
                    "u_off_a": offsets[a],
                    "u_off_k": offsets[k],
                    "v_off_k": offsets[k],
                    "v_off_b": offsets[b],
                    "w_off_b": offsets[b],
                    "w_off_a": offsets[a],
                    "sa": sa,
                    "sk": sk,
                    "sb": sb
                })
                
                total_rank += decomp["U"][0].__len__() if decomp["U"] else 0
    
    # Allocate final matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    for bd in block_decomps:
        sa, sk, sb = bd["sa"], bd["sk"], bd["sb"]
        U_block = bd["U"]
        V_block = bd["V"]
        W_block = bd["W"]
        rank_block = len(U_block[0])
        
        for local_i in range(sa):
            for local_j in range(sk):
                global_u_row = (bd["u_off_a"] + local_i) * m + (bd["u_off_k"] + local_j)
                for r in range(rank_block):
                    U[global_u_row][current_col + r] = U_block[local_i * sk + local_j][r]
        
        for local_j in range(sk):
            for local_k in range(sb):
                global_v_row = (bd["v_off_k"] + local_j) * p + (bd["v_off_b"] + local_k)
                for r in range(rank_block):
                    V[global_v_row][current_col + r] = V_block[local_j * sb + local_k][r]
        
        for local_k in range(sb):
            for local_i in range(sa):
                global_w_row = (bd["w_off_b"] + local_k) * n + (bd["w_off_a"] + local_i)
                for r in range(rank_block):
                    W[global_w_row][current_col + r] = W_block[local_k * sa + local_i][r]
        
        current_col += rank_block
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_for(n, m, p):
    """Generic schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition first
    # It should be fast to construct
    try:
        block_decomp = _block_decomposition()
        # Verify it
        if _verify(block_decomp, n, m, p):
            return block_decomp
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
