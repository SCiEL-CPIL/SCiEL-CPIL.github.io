"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _construct_rank49():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via two levels of Strassen.
    
    Level 1: Treat 4x4 as 2x2 blocks of 2x2. Apply Strassen -> 7 block products.
    Level 2: Each block product is 2x2 * 2x2. Apply Strassen -> 7 scalar products.
    Total: 7 * 7 = 49 scalar multiplications.
    
    Output convention: W[k*n + i, r] where result C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, the tensor is T[i,j,k] = A[i,j] * B[j,k] = C[i,k].
    The decomposition is C[i,k] = sum_r sum_j U[i,j,r] V[j,k,r] W[i,k,r]? No.
    
    Standard tensor decomposition:
    T = sum_r U_r ⊗ V_r ⊗ W_r
    where T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]? 
    The prompt says: U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Let's implement the two-level Strassen.
    
    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for 2x2 matrices
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # A forms:
    # M1: A11 + A22 -> [1, 0, 0, 1]
    # M2: A21 + A22 -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: A11 + A12 -> [1, 1, 0, 0]
    # M6: A21 - A11 -> [-1, 0, 1, 0]
    # M7: A12 - A22 -> [0, 1, 0, -1]
    
    A_forms = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # B forms:
    # M1: B11 + B22 -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B21 - B22 -> [0, 0, 1, -1]
    # M4: B12 - B11 -> [-1, 1, 0, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: B11 + B12 -> [1, 1, 0, 0]
    # M7: B21 + B22 -> [0, 0, 1, 1]
    
    B_forms = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # C forms (output combination):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    C_forms = [
        [1, 0, 0, 1, -1, 0, 1],  # C11 (from M1..M7)
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    # Two-level Strassen
    # Outer Strassen (level 1): 4x4 -> 2x2 blocks of 2x2
    # Block indices: A: [00, 01, 10, 11], B: [00, 01, 10, 11], C: [00, 01, 10, 11]
    # Each block is 2x2.
    
    # Outer Strassen forms for blocks:
    # Same structure as above, but now each "element" is a 2x2 block.
    # We apply the same linear forms to the blocks.
    
    # For each outer Strassen step s (0..6), we compute block product P_s.
    # P_s is a 2x2 matrix (sum of block products).
    # Then we apply inner Strassen to each P_s? No.
    
    # Wait, the outer Strassen gives 7 block products. Each block product is a 2x2 matrix product.
    # So we need to decompose each 2x2 * 2x2 into 7 scalar products.
    # Total 7 * 7 = 49 scalar products.
    
    # Let's index the 49 products by (s, t) where s is outer Strassen step, t is inner Strassen step.
    
    # Outer Strassen:
    # A_block_forms[s][b] where b is block index (0..3)
    # B_block_forms[s][b]
    # C_block_forms[s][c] where c is block index (0..3)
    
    # These are the same as A_forms, B_forms, C_forms above, but applied to blocks.
    
    # For a fixed outer step s, the block product is:
    # P_s = sum_b A_block_forms[s][b] * B_block_forms[s][b]
    # Wait, no. Strassen computes:
    # M_s = A_s * B_s where A_s = sum_b A_block_forms[s][b] * A_b, B_s = sum_b B_block_forms[s][b] * B_b
    # So the block product for step s is M_s = A_s * B_s.
    # M_s is a 2x2 matrix.
    
    # Now, M_s = A_s * B_s where A_s and B_s are 2x2 matrices.
    # We decompose M_s using inner Strassen:
    # M_s = sum_t M_s,t where M_s,t are scalar products.
    # M_s,t = (A_s_inner[t]) * (B_s_inner[t]) where A_s_inner[t] is a scalar linear form of A_s's elements.
    
    # So the overall decomposition is:
    # C = sum_s C_block_forms[s] * M_s = sum_s sum_c C_block_forms[s][c] * (sum_t M_s,t)
    
    # Let's construct U, V, W.
    # U[i*m+j, r] where r = s*7 + t
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # For a given r = s*7 + t:
    # U[i*m+j, r] = A_block_forms[s][blockA(i,j)] * A_inner_forms[t][elemA(i,j)]
    # V[j*p+k, r] = B_block_forms[s][blockB(j,k)] * B_inner_forms[t][elemB(j,k)]
    # W[k*n+i, r] = C_block_forms[s][blockC(i,k)] * C_inner_forms[t][elemC(i,k)]
    
    # Wait, this is not quite right. The inner Strassen applies to the 2x2 blocks A_s and B_s.
    # A_s is a 2x2 matrix. Its elements are A_s[0,0], A_s[0,1], A_s[1,0], A_s[1,1].
    # A_s[0,0] = sum_b A_block_forms[s][b] * A_b[0,0]
    # A_s[0,1] = sum_b A_block_forms[s][b] * A_b[0,1]
    # etc.
    
    # The inner Strassen linear forms A_inner_forms[t] are applied to A_s's elements.
    # So A_inner_value[t] = sum_e A_inner_forms[t][e] * A_s[e]
    # = sum_e A_inner_forms[t][e] * (sum_b A_block_forms[s][b] * A_b[e])
    # = sum_b A_block_forms[s][b] * (sum_e A_inner_forms[t][e] * A_b[e])
    
    # So for a fixed s, t, the coefficient for A_b[e] is A_block_forms[s][b] * A_inner_forms[t][e].
    # Similarly for B.
    # And for C, the coefficient is C_block_forms[s][c] * C_inner_forms[t][f] where f is the element index in C_b.
    
    # Let's define:
    # For A: block index b (0..3), element index e (0..3) within the block.
    # Global index i*m+j = blockA * 2 + elemA where blockA is (row_block, col_block) and elemA is (row_elem, col_elem).
    # blockA = i_block * 2 + j_block, elemA = i_elem * 2 + j_elem.
    # i = i_block * 2 + i_elem, j = j_block * 2 + j_elem.
    
    # So for U[i*m+j, r]:
    # i_block = i // 2, i_elem = i % 2
    # j_block = j // 2, j_elem = j % 2
    # blockA = i_block * 2 + j_block
    # elemA = i_elem * 2 + j_elem
    # U[i*m+j, r] = A_block_forms[s][blockA] * A_inner_forms[t][elemA]
    
    # For V[j*p+k, r]:
    # j_block = j // 2, j_elem = j % 2
    # k_block = k // 2, k_elem = k % 2
    # blockB = j_block * 2 + k_block
    # elemB = j_elem * 2 + k_elem
    # V[j*p+k, r] = B_block_forms[s][blockB] * B_inner_forms[t][elemB]
    
    # For W[k*n+i, r]:
    # k_block = k // 2, k_elem = k % 2
    # i_block = i // 2, i_elem = i % 2
    # blockC = k_block * 2 + i_block
    # elemC = k_elem * 2 + i_elem
    # W[k*n+i, r] = C_block_forms[s][blockC] * C_inner_forms[t][elemC]
    
    # Wait, the C_inner_forms are for the output of the inner Strassen, which is a 2x2 matrix.
    # The inner Strassen produces M_s, which is a 2x2 matrix.
    # M_s[e] = sum_t C_inner_forms[t][e] * (A_inner_value[t] * B_inner_value[t])
    # So the coefficient for M_s,e is C_inner_forms[t][e].
    # And M_s is added to C with weight C_block_forms[s][blockC].
    # So the total coefficient for C_block[e] is C_block_forms[s][blockC] * C_inner_forms[t][e].
    
    # Let's implement this.
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    i_block = i // 2
                    i_elem = i % 2
                    j_block = j // 2
                    j_elem = j % 2
                    blockA = i_block * 2 + j_block
                    elemA = i_elem * 2 + j_elem
                    U[i * m + j][r] = A_forms[s][blockA] * A_forms[t][elemA]
                    
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    j_elem = j % 2
                    k_block = k // 2
                    k_elem = k % 2
                    blockB = j_block * 2 + k_block
                    elemB = j_elem * 2 + k_elem
                    V[j * p + k][r] = B_forms[s][blockB] * B_forms[t][elemB]
                    
            for k in range(4):
                for i in range(4):
                    k_block = k // 2
                    k_elem = k % 2
                    i_block = i // 2
                    i_elem = i % 2
                    blockC = k_block * 2 + i_block
                    elemC = k_elem * 2 + i_elem
                    W[k * n + i][r] = C_forms[s][blockC] * C_forms[t][elemC]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try rank-49 Strassen decomposition
    try:
        result = _construct_rank49()
        # Validate
        U = result["U"]
        V = result["V"]
        W = result["W"]
        n, m, p = 4, 4, 4
        rank = len(U[0])
        
        # Quick validation: check a few entries
        valid = True
        for _ in range(100):
            i = (int(time.time() * 1000) + _) % 4
            j = (int(time.time() * 2000) + _ * 3) % 4
            k = (int(time.time() * 3000) + _ * 7) % 4
            
            expected = 1.0  # T[i,j,k] = 1 if this is the tensor for matrix mult
            
            # Actually T[i,j,k] is the coefficient in the multiplication tensor.
            # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # So the tensor T[i,j,k] = 1.
            
            computed = 0.0
            for r in range(rank):
                computed += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            
            if abs(computed - expected) > 1e-9:
                valid = False
                break
        
        if valid and time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    n, m, p = DIMENSIONS
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
