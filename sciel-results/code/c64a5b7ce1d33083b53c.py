"""Matrix multiplication 6x6x6: search for a lower-rank exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the 6x6x6 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Tensor T[i,j,k] = 1 if and only if the corresponding term is present
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # Actually the tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k
    # Wait, let me reconsider. The matrix multiplication tensor has entries T[i,j,k] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need this sum to equal 1 for all i,j,k.
    
    # Use numpy for speed
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_cyclic_decomposition():
    """
    Try a cyclic group-based decomposition.
    
    For Z_n with n=6, there are known constructions. Let's try the following:
    Use the fact that for cyclic groups, the tensor rank of matrix multiplication
    over Z_n can be bounded. 
    
    One known approach: For n=6, use the structure of Z_6 = Z_2 x Z_3.
    
    Let's try a direct construction based on roots of unity / DFT-like patterns.
    
    Actually, let's try a simpler approach first: the "strassen-like" recursive
    decomposition. For 6x6, we can split into 3x3 blocks and use Strassen's
    algorithm recursively.
    
    Strassen's algorithm for 2x2 uses 7 multiplications instead of 8.
    For 3x3, we can split into 2x2 blocks with some overlap... this gets complex.
    
    Let me try a different approach: use the fact that 6 = 2*3 and try to
    combine a Strassen-like decomposition for 2x2 with a simple 3x3 approach.
    """
    n, m, p = DIMENSIONS
    # This is complex to construct directly. Let's try a randomized search instead.
    pass

def _random_search(max_rank, time_limit):
    """
    Randomized search for a rank-r decomposition.
    Start with the schoolbook rank and try to reduce it.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start from schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank by random optimization
    # This is a heuristic approach - try random rank-r decompositions and see if any work
    # For now, just return the schoolbook decomposition as a baseline
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.
    
    The output follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s, keep safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Return the schoolbook decomposition (rank 216)
    # This is guaranteed to work and is the baseline.
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Check if we have time to try something better
    if time.time() < deadline - 30:
        # Try a structured decomposition based on block structure
        # 6x6 = (2x2 blocks of 3x3) or (3x3 blocks of 2x2)
        # Let's try to use Strassen's algorithm for the 2x2 block structure
        # with 3x3 blocks, where each 3x3 multiplication uses rank 27
        # Strassen for 2x2: 7 multiplications instead of 8
        # So total rank would be 7 * 27 = 189 < 216
        
        # But we need to be careful about the block structure.
        # Let A and B be 6x6 matrices partitioned into 2x2 blocks of 3x3:
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C = A*B = [[C11, C12], [C21, C22]]
        
        # Strassen's formulas for 2x2 block matrix multiplication:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 3x3 matrix product, which has rank 27 in the standard decomposition.
        # Total rank = 7 * 27 = 189
        
        # Let's construct this decomposition
        rank_strassen = 7 * 27
        U_s = [[0] * rank_strassen for _ in range(n * m)]
        V_s = [[0] * rank_strassen for _ in range(m * p)]
        W_s = [[0] * rank_strassen for _ in range(p * n)]
        
        # Define the 7 Strassen products
        # Each product involves adding/subtracting blocks of A and B
        # We need to map this to the tensor decomposition
        
        # For each of the 7 Strassen products, we have a 3x3 matrix multiplication
        # The blocks are:
        # A11: rows 0-2, cols 0-2
        # A12: rows 0-2, cols 3-5
        # A21: rows 3-5, cols 0-2
        # A22: rows 3-5, cols 3-5
        # Similarly for B
        
        # Let's define the block offsets
        block_size = 3
        
        # For each Strassen product, define which A-blocks and B-blocks are involved
        # and with what signs
        strassen_products = [
            # (A_blocks, B_blocks) where each is a list of (block_row, block_col, sign)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
            ([(1,0,1), (1,1,1)], [(0,0,1)]),            # M2
            ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # M3
            ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # M4
            ([(0,0,1), (0,1,1)], [(1,1,1)]),            # M5
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # M6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # M7
        ]
        
        # For each Strassen product and each (i,j,k) in the 3x3 block,
        # we add a rank-1 term
        col = 0
        for prod_idx in range(7):
            A_blocks, B_blocks = strassen_products[prod_idx]
            for bi in range(block_size):
                for bj in range(block_size):
                    for bk in range(block_size):
                        # For each A block contribution
                        for (ar, ac, a_sign) in A_blocks:
                            # The row index in A is ar*block_size + bi, col is ac*block_size + bj
                            a_row = ar * block_size + bi
                            a_col = ac * block_size + bj
                            U_s[a_row * m + a_col][col] += a_sign
                        # For each B block contribution
                        for (br, bc, b_sign) in B_blocks:
                            # The row index in B is br*block_size + bj, col is bc*block_size + bk
                            b_row = br * block_size + bj
                            b_col = bc * block_size + bk
                            V_s[b_row * p + b_col][col] += b_sign
                        # The W factor: C = A*B, so C[i,j] gets contribution from A[i,*] * B[*,j]
                        # The output block for product prod_idx contributes to C blocks
                        # We need to figure out which C blocks get this product and with what sign
                        # This is where it gets tricky - we need to map back to C blocks
                        
                        # For now, let's just set W to 1 and verify later
                        # Actually, we need to properly account for the C block contributions
                        
                        # Let me reconsider the approach. The W matrix entries should
                        # reflect which C blocks receive contributions from each Strassen product.
                        
                        # C11 = M1 + M4 - M5 + M7
                        # C12 = M3 + M5
                        # C21 = M2 + M4
                        # C22 = M1 - M2 + M3 + M6
                        
                        # So for each Strassen product, we need to know which C blocks it contributes to
                        c_contributions = {
                            0: [(0,0,1), (1,1,1)],   # M1 -> C11(+), C22(+)
                            1: [(1,0,1), (1,1,-1)],   # M2 -> C21(+), C22(-)
                            2: [(0,1,1), (1,1,1)],   # M3 -> C12(+), C22(+)
                            3: [(0,0,1), (1,0,1)],   # M4 -> C11(+), C21(+)
                            4: [(0,0,-1), (0,1,1)],   # M5 -> C11(-), C12(+)
                            5: [(1,1,1)],              # M6 -> C22(+)
                            6: [(0,0,1), (1,1,1)],   # M7 -> C11(+), C22(+)
                        }
                        
                        for (cr, cc, c_sign) in c_contributions[prod_idx]:
                            # The row index in C is cr*block_size + bi, col is cc*block_size + bk
                            c_row = cr * block_size + bi
                            c_col = cc * block_size + bk
                            W_s[c_col * n + c_row][col] += c_sign
                        
                        col += 1
        
        # Verify the Strassen-based decomposition
        if _verify(U_s, V_s, W_s):
            result = {"U": U_s, "V": V_s, "W": W_s}
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
