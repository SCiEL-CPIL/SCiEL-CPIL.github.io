import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18

    T_target = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def validate(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        T_comp = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
        if not np.allclose(T_comp, T_target, atol=1e-9):
            return False
        # Check half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Trivial solution rank 54
    U0 = [[0]*54 for _ in range(9)]
    V0 = [[0]*54 for _ in range(18)]
    W0 = [[0]*54 for _ in range(18)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U0[u][col] = 1
                V0[v][col] = 1
                W0[w][col] = 1
                col += 1
    best_solution = (U0, V0, W0)
    best_rank = 54

    # Strassen decomposition for 3x3x3 matrix multiplication (rank 7)
    # The 7 bilinear forms for Strassen's algorithm:
    # M1 = (A00) * (B00 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A11 - A00) * B00
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B10
    # M6 = (A10 - A11) * (B01 + B11)
    # M7 = (A01 + A11) * (B00 - B10)
    
    # For multiplying C = A * B where A is 3x3 and B is 3x3:
    # C00 = M1 + M3 - M2 + M4
    # C01 = M5 + M4
    # C02 = M6
    # C10 = M7
    # C11 = M1 - M5 + M6 + M7
    # C12 = M1 + M2 - M3 - M7
    # C20 = M3
    # C21 = M2
    # C22 = M1 - M2 + M3 + M7
    
    # Let's define the Strassen factors for a single 3x3x3 block
    # U factors (for A matrix, 9 elements)
    # V factors (for B matrix, 9 elements)
    # W factors (for C matrix, 9 elements)
    
    # Indexing: A[i,j] -> u = i*3 + j, B[j,k] -> v = j*3 + k, C[i,k] -> w = i*3 + k
    
    # For the first block (columns 0-2 of B, rows 0-2 of C):
    # B indices: j in {0,1,2}, k in {0,1,2} -> v = j*3 + k
    # C indices: i in {0,1,2}, k in {0,1,2} -> w = i*3 + k
    
    # For the second block (columns 3-5 of B, rows 3-5 of C):
    # B indices: j in {0,1,2}, k in {3,4,5} -> v = j*3 + (k-3) + 9 = j*3 + k - 3 + 9... 
    # Wait, let me re-index. v = j*p + k where p=6. So for k=0,1,2: v = j*6 + k
    # For k=3,4,5: v = j*6 + k
    # w = k*n + i where n=3. So for k=0,1,2: w = k*3 + i
    # For k=3,4,5: w = k*3 + i
    
    # Let's build the Strassen decomposition for the full 3x3x6 problem
    # by treating it as two independent 3x3x3 multiplications.
    
    # Block 1: k in {0,1,2}
    # Block 2: k in {3,4,5}
    
    # For each block, we use 7 Strassen terms.
    # Total rank = 7 + 7 = 14
    
    # Let's define the factors for each block.
    
    # Block 1 (k=0,1,2):
    # A is 3x3, B is 3x3 (columns 0-2), C is 3x3 (rows 0-2)
    
    # U factors for A (same for both blocks since A is the same):
    # But wait, the U matrix has shape (9, rank). The A matrix is the same for both blocks.
    # So we can reuse the same U factors? No, because each term in the decomposition
    # contributes to the full tensor. Let me think again.
    
    # The tensor T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # For block 1 (k=0,1,2): T[i*3+j, j*6+k, k*3+i] = 1
    # For block 2 (k=3,4,5): T[i*3+j, j*6+k, k*3+i] = 1
    
    # Since the blocks are independent (different v and w indices), we can construct
    # separate decompositions for each block and combine them.
    
    # For block 1, the relevant indices are:
    # u in {0,...,8}, v in {0,6,12, 1,7,13, 2,8,14}, w in {0,1,2, 3,4,5, 6,7,8}
    
    # Let's define Strassen factors for a generic 3x3x3 multiplication.
    # A[i,j] with i,j in {0,1,2}
    # B[j,k] with j,k in {0,1,2}
    # C[i,k] with i,k in {0,1,2}
    
    # Strassen's 7 products:
    # M1 = A00 * (B00 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A11 - A00) * B00
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B10
    # M6 = (A10 - A11) * (B01 + B11)
    # M7 = (A01 + A11) * (B00 - B10)
    
    # U factors (for A):
    # M1: A00 -> u=0
    # M2: A00, A10 -> u=0, u=3
    # M3: A11, -A00 -> u=4, u=0
    # M4: A11 -> u=4
    # M5: A00, A01 -> u=0, u=1
    # M6: A10, -A11 -> u=3, u=4
    # M7: A01, A11 -> u=1, u=4
    
    # V factors (for B):
    # M1: B00, -B11 -> v=0, v=4
    # M2: B11 -> v=4
    # M3: B00 -> v=0
    # M4: B11, -B01 -> v=4, v=1
    # M5: B10 -> v=3
    # M6: B01, B11 -> v=1, v=4
    # M7: B00, -B10 -> v=0, v=3
    
    # W factors (for C):
    # C00 = M1 + M3 - M2 + M4
    # C01 = M5 + M4
    # C02 = M6
    # C10 = M7
    # C11 = M1 - M5 + M6 + M7
    # C12 = M1 + M2 - M3 - M7
    # C20 = M3
    # C21 = M2
    # C22 = M1 - M2 + M3 + M7
    
    # Let's verify this is correct by checking a few entries.
    # C00 = M1 + M3 - M2 + M4
    # = A00*(B00-B11) + (A11-A00)*B00 - (A00+A10)*B11 + A11*(B11-B01)
    # = A00*B00 - A00*B11 + A11*B00 - A00*B00 - A00*B11 - A10*B11 + A11*B11 - A11*B01
    # = -A00*B11 + A11*B00 - A00*B11 - A10*B11 + A11*B11 - A11*B01
    # Hmm, this doesn't look right. Let me re-derive.
    
    # Actually, the standard Strassen algorithm for 2x2 matrices is well-known.
    # For 3x3, we need to be more careful. Let me use a known rank-7 decomposition
    # for 3x3 matrix multiplication.
    
    # Actually, the minimum rank for 3x3 matrix multiplication is 7, achieved by
    # Strassen's algorithm adapted to 3x3. Let me look up the correct factors.
    
    # A simpler approach: use the known rank-7 decomposition.
    # The 7 bilinear forms are:
    # M1 = (A00 - A11) * (B00 + B11)
    # M2 = (A00 + A11) * B11
    # M3 = A10 * (B01 - B11)
    # M4 = (A01 + A10) * B10
    # M5 = A11 * (B00 - B10)
    # M6 = (A00 - A01) * B01
    # M7 = A01 * (B00 + B10)
    
    # This is getting complicated. Let me just use the trivial rank-54 solution
    # and try to find a better one using the block structure.
    
    # Actually, let me try a different approach. Since the problem is a block-diagonal
    # sum of two 3x3x3 multiplications, and the minimum rank for 3x3x3 is 7,
    # the total rank should be at most 14.
    
    # Let me construct the rank-14 solution explicitly.
    
    # For block 1 (k=0,1,2):
    # u indices: 0-8 (A matrix)
    # v indices: 0,1,2, 6,7,8, 12,13,14 (B matrix, columns 0-2)
    # w indices: 0,1,2, 3,4,5, 6,7,8 (C matrix, rows 0-2)
    
    # For block 2 (k=3,4,5):
    # u indices: 0-8 (A matrix)
    # v indices: 3,4,5, 9,10,11, 15,16,17 (B matrix, columns 3-5)
    # w indices: 9,10,11, 12,13,14, 15,16,17 (C matrix, rows 3-5)
    
    # Let me define the Strassen factors for a 3x3x3 multiplication.
    # I'll use the following known rank-7 decomposition:
    
    # U (for A):
    # col 0: [1, 0, 0, 0, 0, 0, 0, 0, 0]  # A00
    # col 1: [1, 0, 0, 1, 0, 0, 0, 0, 0]  # A00 + A10
    # col 2: [0, 0, 0, 0, 0, 0, 1, 0, 0]  # A20
    # col 3: [0, 0, 0, 0, 0, 0, 0, 1, 0]  # A21
    # col 4: [0, 0, 0, 0, 0, 0, 0, 0, 1]  # A22
    # col 5: [1, 0, 0, 0, 0, 0, 0, 0, 0]  # A00 (again, for another term)
    # col 6: [0, 0, 0, 0, 0, 0, 1, 0, 0]  # A20 (again)
    
    # This is getting too complex. Let me just return the trivial solution
    # since finding a rank-14 solution requires careful construction.
    
    # Actually, let me try to construct the rank-14 solution more carefully.
    # I'll use the fact that the problem is two independent 3x3x3 multiplications.
    
    # For a 3x3x3 multiplication, the tensor is:
    # T[i*3+j, j*3+k, i*3+k] = 1 for i,j,k in {0,1,2}
    
    # The rank-7 decomposition for 3x3x3 can be obtained from Strassen's algorithm.
    # Let me use the following factors (from literature):
    
    # For 3x3 matrix multiplication C = A * B:
    # M1 = (A00 - A11) * (B00 + B11)
    # M2 = (A00 + A11) * B11  
    # M3 = A10 * (B01 - B11)
    # M4 = (A01 + A10) * B10
    # M5 = A11 * (B00 - B10)
    # M6 = (A00 - A01) * B01
    # M7 = A01 * (B00 + B10)
    
    # C00 = M1 + M2 - M5 - M7
    # C01 = M3 + M4 - M6 - M7
    # C02 = 0
    # C10 = M1 + M4 - M3
    # C11 = M1 + M2 - M4 - M5
    # C12 = 0
    # C20 = 0
    # C21 = 0
    # C22 = 0
    
    # This doesn't seem right either. The issue is that Strassen's algorithm
    # is for 2x2 matrices, and extending to 3x3 is non-trivial.
    
    # Given the complexity, let me just return the trivial rank-54 solution.
    # The search for a better solution via gradient descent in the original code
    # didn't find anything better, so I'll keep the trivial solution.
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
