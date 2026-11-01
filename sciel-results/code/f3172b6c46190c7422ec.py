import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    import random
    random.seed(42)
    
    def verify_strassen():
        """Verify Strassen decomposition with random test cases"""
        for _ in range(20):
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            c = random.randint(-5, 5)
            d = random.randint(-5, 5)
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            u = random.randint(-5, 5)
            v = random.randint(-5, 5)
            
            # Expected result
            C00 = a*x + b*u
            C01 = a*y + b*v
            C10 = c*x + d*u
            C11 = c*y + d*v
            
            # Strassen
            M1 = (a+d)*(x+u)
            M2 = (c+d)*x
            M3 = a*(y-u)
            M4 = d*(v-u)
            M5 = (a+b)*u
            M6 = (c+d)*(v-y)
            M7 = (a+b)*(y-v)
            
            sC00 = M1 - M2 - M4 + M3
            sC01 = M3 + M4 + M7
            sC10 = M2 + M5 + M6
            sC11 = M1 - M5 + M6 + M7
            
            if (sC00, sC01, sC10, sC11) != (C00, C01, C10, C11):
                return False
        return True
    
    strassen_valid = verify_strassen()
    
    if not strassen_valid:
        # Fall back to schoolbook
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
    
    # Correct index mappings:
    # A (2x5): U[i*m + j] = U[i*5 + j], i=0..1, j=0..4
    #   Row 0: U[0],U[1],U[2],U[3],U[4]
    #   Row 1: U[5],U[6],U[7],U[8],U[9]
    # B (5x6): V[j*p + k] = V[j*6 + k], j=0..4, k=0..5
    #   Row 0: V[0]..V[5]
    #   Row 1: V[6]..V[11]
    #   Row 2: V[12]..V[17]
    #   Row 3: V[18]..V[23]
    #   Row 4: V[24]..V[29]
    # C (2x6): W[k*n + i] = W[k*2 + i], k=0..5, i=0..1
    #   k=0: W[0],W[1]
    #   k=1: W[2],W[3]
    #   k=2: W[4],W[5]
    #   k=3: W[6],W[7]
    #   k=4: W[8],W[9]
    #   k=5: W[10],W[11]
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(u_idx, v_idx, w_idx):
        """Add Strassen for 2x2*2x2 with explicit index lists.
        u_idx = [a, b, c, d] indices into U
        v_idx = [x, y, u, v] indices into V (row-major 2x2)
        w_idx = [C00, C01, C10, C11] indices into W
        """
        nonlocal col
        a, b, c, d = u_idx
        x, y, u, v = v_idx
        C00, C01, C10, C11 = w_idx
        
        # M1 = (a+d)(x+u)
        U[a][col] += 1
        U[d][col] += 1
        V[x][col] += 1
        V[u][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M2 = (c+d)x
        U[c][col] += 1
        U[d][col] += 1
        V[x][col] += 1
        W[C00][col] -= 1
        W[C10][col] += 1
        col += 1
        
        # M3 = a(y-u)
        U[a][col] += 1
        V[y][col] += 1
        V[u][col] -= 1
        W[C00][col] += 1
        W[C01][col] += 1
        col += 1
        
        # M4 = d(v-u)
        U[d][col] += 1
        V[v][col] += 1
        V[u][col] -= 1
        W[C00][col] -= 1
        W[C01][col] += 1
        col += 1
        
        # M5 = (a+b)u
        U[a][col] += 1
        U[b][col] += 1
        V[u][col] += 1
        W[C10][col] += 1
        W[C11][col] -= 1
        col += 1
        
        # M6 = (c+d)(v-y)
        U[c][col] += 1
        U[d][col] += 1
        V[v][col] += 1
        V[y][col] -= 1
        W[C10][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (a+b)(y-v)
        U[a][col] += 1
        U[b][col] += 1
        V[y][col] += 1
        V[v][col] -= 1
        W[C01][col] += 1
        W[C11][col] += 1
        col += 1
    
    def add_schoolbook_2x3x2(u_idx, v_idx, w_idx):
        """Add schoolbook for 2x3 * 3x2 -> 2x2.
        u_idx = list of 6 indices for A (row-major 2x3)
        v_idx = list of 6 indices for B (row-major 3x2)
        w_idx = list of 4 indices for C (row-major 2x2)
        """
        nonlocal col
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[u_idx[i*3 + j]][col] += 1
                    V[v_idx[j*2 + k]][col] += 1
                    W[w_idx[i*2 + k]][col] += 1
                    col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A rows 0-1, cols 0-1: U[0],U[1],U[5],U[6]
    # B rows 0-1, cols 0-1: V[0],V[1],V[6],V[7]
    # C cols 0-1, rows 0-1: W[0],W[1],W[2],W[3]
    add_strassen_block(
        u_idx=[0, 1, 5, 6],
        v_idx=[0, 1, 6, 7],
        w_idx=[0, 1, 2, 3]
    )
    
    # Block 2a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # B rows 0-1, cols 2-3: V[2],V[3],V[8],V[9]
    # C cols 2-3, rows 0-1: W[4],W[5],W[6],W[7]
    add_strassen_block(
        u_idx=[0, 1, 5, 6],
        v_idx=[2, 3, 8, 9],
        w_idx=[4, 5, 6, 7]
    )
    
    # Block 2b: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # B rows 0-1, cols 4-5: V[4],V[5],V[10],V[11]
    # C cols 4-5, rows 0-1: W[8],W[9],W[10],W[11]
    add_strassen_block(
        u_idx=[0, 1, 5, 6],
        v_idx=[4, 5, 10, 11],
        w_idx=[8, 9, 10, 11]
    )
    
    # Block 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # A rows 0-1, cols 2-4: U[2],U[3],U[4],U[7],U[8],U[9]
    # B rows 2-4, cols 0-1: V[12],V[13],V[18],V[19],V[24],V[25]
    # C cols 0-1, rows 0-1: W[0],W[1],W[2],W[3]
    add_schoolbook_2x3x2(
        u_idx=[2, 3, 4, 7, 8, 9],
        v_idx=[12, 13, 18, 19, 24, 25],
        w_idx=[0, 1, 2, 3]
    )
    
    # Block 4a: A[0:2, 2:5] * B[2:5, 2:4] -> C[0:2, 2:4]
    # B rows 2-4, cols 2-3: V[14],V[15],V[20],V[21],V[26],V[27]
    # C cols 2-3, rows 0-1: W[4],W[5],W[6],W[7]
    add_schoolbook_2x3x2(
        u_idx=[2, 3, 4, 7, 8, 9],
        v_idx=[14, 15, 20, 21, 26, 27],
        w_idx=[4, 5, 6, 7]
    )
    
    # Block 4b: A[0:2, 2:5] * B[2:5, 4:6] -> C[0:2, 4:6]
    # B rows 2-4, cols 4-5: V[16],V[17],V[22],V[23],V[28],V[29]
    # C cols 4-5, rows 0-1: W[8],W[9],W[10],W[11]
    add_schoolbook_2x3x2(
        u_idx=[2, 3, 4, 7, 8, 9],
        v_idx=[16, 17, 22, 23, 28, 29],
        w_idx=[8, 9, 10, 11]
    )
    
    # Verify correctness with random test
    if time.time() < deadline:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape: U is (n*m, rank) -> (n, m, rank)
        # V is (m*p, rank) -> (m, p, rank)  
        # W is (p*n, rank) -> (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T is the matrix multiplication tensor: T[i,j,k] = delta[j,k] (for A[i,j]*B[j,k])
        # Actually, the multiplication tensor has T[i,j,k] = 1 if j==k (in the standard formulation)
        # But here we're decomposing the bilinear map: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] where the map is (A,B) -> C, C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i,j,k] = delta_{j,k} ... no
        # The tensor T satisfies: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r] ... 
        # Actually the standard form: T is a 3-tensor with T[i,j,k] such that
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * 1
        # So T[i,j,k] = 1 for all i,j,k? No.
        # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = delta_{j,k} in some conventions,
        # but the standard one is: the bilinear map (A,B)->C where C[i,k] = sum_j A[i,j]B[j,k]
        # This corresponds to tensor T with T[i,j,k] = 1 when... 
        # Let me think differently. The tensor decomposition gives:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta_{j,k} ... no
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j A[i,j] B[j,k] * 1
        # In tensor form: C[i,k] = sum_{j,r} A[i,j] B[j,k] (sum over r of U[i,j,r]V[j,k,r]W[k,i,r])
        # So we need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if this equals the identity in j,k coupling
        # The correct identity: for all i,j,k, sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{j,k}? No.
        # 
        # Let me just verify numerically:
        # Construct random A (2x5) and B (5x6), compute C = A @ B
        # Then compute C_decomp[i,k] = sum_{j,r} A[i,j] * B[j,k] * (sum_r' ... )
        # 
        # Actually the decomposition means:
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
        # No. The tensor decomposition of the multiplication tensor T means:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] is the matrix multiplication tensor.
        # For matrix multiplication, T[i,j,k] = 1 if j==k else 0? No.
        # 
        # The standard matrix multiplication tensor: T_{i,j,k} such that
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # This is a bilinear map. The associated tensor T has T[i,j,k] = delta_{j,k} in the sense that
        # sum_j A[i,j] B[j,k] = sum_{j,k} A[i,j] B[j,k] delta_{j,k}... that's not right either.
        # 
        # Let me just do it numerically:
        # The decomposition says: for all A (n x m) and B (m x p),
        # C[i,k] = A @ B [i,k] = sum_{j} A[i,j] B[j,k]
        # = sum_{j,r} A[i,j] * B[j,k] * (coefficient)
        # The coefficient should be 1 for each (i,j,k) triple in the sum, so:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        # 
        # Wait, that can't be right because then we'd just need rank 1 with all ones.
        # 
        # I think the correct interpretation is:
        # The multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # No... Let me look at this more carefully.
        #
        # The bilinear map is: mu: R^{n*m} x R^{m*p} -> R^{n*p}
        # mu(A, B)[i,k] = sum_j A[i,j] B[j,k]
        #
        # The tensor T in R^{n*m} x R^{m*p} x R^{n*p} (dual) is defined by:
        # <T, A ⊗ B ⊗ C> = sum_{i,j,k} C[i,k] * T[i,j,k] = mu(A,B) · C = sum_{i,k,j} A[i,j] B[j,k] C[i,k]
        # So T[i,j,k] = 1 for all i,j,k.
        #
        # And the decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
        # So T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
        # = sum_r U_r[i,j] V_r[j,k] W_r[k,i] should equal 1 for all i,j,k.
        #
        # Let me verify:
        test_pass = True
        for _ in range(5):
            A_test = np.random.randint(-3, 3, (n, m)).astype(np.float64)
            B_test = np.random.randint(-3, 3, (m, p)).astype(np.float64)
            C_expected = A_test @ B_test
            
            # Compute using decomposition
            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (sum_r ... )
            # From T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # So C[i,k] = sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * 1
            # = sum_j A[i,j] B[j,k] * (sum_r U[i,j,r] V[j,k,r] W[k,i,r])
            # = sum_r (sum_j A[i,j] U[i,j,r]) * (sum_j B[j,k] V[j,k,r]) * W[k,i,r]
            # 
            # Hmm, that's not quite right either. Let me think again.
            #
            # Actually, I think the convention might be different. Let me just check if:
            # For each r, define:
            # A_r[i,j] = U[i*m+j, r]
            # B_r[j,k] = V[j*p+k, r]  
            # C_r[k,i] = W[k*n+i, r]
            # Then the claim is that for all A, B:
            # A @ B = sum_r (something involving A_r, B_r, C_r)
            #
            # The standard tensor decomposition of matrix multiplication:
            # C = A @ B where C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor form: sum_{j} A[i,j] B[j,k] = sum_r (sum_j A[i,j] U_r[i,j]) (sum_j B[j,k] V_r[j,k]) W_r[k,i]
            # No, that's not dimensionally consistent.
            #
            # Let me just directly verify: the tensor T[i,j,k] for matrix mult is 1 for all i,j,k.
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Check this equals 1 for all i,j,k.
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        T_check[i,j,k] = s
            if not np.allclose(T_check, 1.0):
                test_pass = False
                break
        
        if not test_pass:
            # Fallback to schoolbook if Strassen blocks are wrong
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
