"""Matrix multiplication 2x4x7 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 2x4x7 matrix multiplication.
    
    Strategy: Use Strassen-like block decomposition to reduce rank below the schoolbook 56.
    For 2x4x7, we can view this as multiplying a 2x4 matrix by a 4x7 matrix.
    
    The schoolbook rank is 2*4*7 = 56.
    
    We can try to use block structure:
    - Split the 4x7 into 4x4 + 4x3 blocks
    - Use Strassen on the 2x4 x 4x4 part (which can be seen as 2x4 x 4x4)
    
    Actually, let me think about this more carefully.
    
    A 2x4 matrix A times a 4x7 matrix B gives a 2x7 matrix C.
    
    One approach: partition the 4 rows of A and 4 columns of B into 2x2 blocks.
    A = [A11 A12; A21 A22] where each Ai is 1x2
    B = [B11 B12 B13; B21 B22 B23; B31 B32 B33; B41 B42 B43]
    
    Actually, let's use a different partitioning. Let's split the 4 into 2+2.
    
    A (2x4) = [A1 A2] where A1 is 2x2, A2 is 2x2
    B (4x7) = [B1; B2] where B1 is 2x7, B2 is 2x7
    
    C = A1*B1 + A2*B2
    
    Now A1*B1 is 2x2 times 2x7, and A2*B2 is 2x2 times 2x7.
    
    For 2x2 x 2x7: we can use Strassen to compute 2x2 * 2x2 in 7 multiplications,
    but we have 7 columns. So we need 7 * 7 = 49 multiplications for each 2x2*2x7,
    giving 98 total. That's worse than 56.
    
    Let me reconsider. The schoolbook is 56. Can we do better?
    
    For 2x4x7, the tensor rank is known to be at most 56 (schoolbook).
    The lower bound is max(2*4, 4*7, 2*7) = max(8, 28, 14) = 28.
    
    Let me try a block approach:
    Split A (2x4) into two 2x2 blocks: A = [A1, A2]
    Split B (4x7) into two 2x7 blocks: B = [B1; B2]
    C = A1*B1 + A2*B2
    
    Each Ai*Bi is a 2x2 * 2x7 multiplication.
    
    For 2x2 * 2x7: 
    Let A' = [[a, b], [c, d]] (2x2)
    Let B' = [b11 b12 ... b17; b21 b22 ... b27] (2x7)
    C' = A'*B' = [a*b11+b*b21, ..., a*b17+b*b27; c*b11+d*b21, ..., c*b17+d*b27]
    
    Schoolbook for 2x2*2x7: 4*7 = 28 multiplications.
    
    Can we do better? Using Strassen-like ideas:
    We can group columns. If we split the 7 columns into 3+3+1:
    - 2x2 * 2x3: can we do better than 12?
    
    Actually, for 2x2 * 2x2, Strassen gives rank 7 (vs schoolbook 8).
    
    For 2x2 * 2x3, we could use Strassen on a 2x2*2x2 part (rank 7) plus 2x2*2x1 (rank 4), total 11.
    
    So for 2x2 * 2x7: split 7 into 2+2+2+1: 4*7 = 28 schoolbook, or 7+7+7+4 = 25 with Strassen on each 2x2 block and schoolbook on 2x1.
    
    Wait, 2x2*2x1: A is 2x2, B is 2x1. C = A*B is 2x1.
    C1 = a*b1 + b*b2, C2 = c*b1 + d*b2. That's 4 multiplications schoolbook.
    Can we do better? The tensor is 2x2x1. The rank is at least max(2*2, 2*1, 2*1) = 4. So 4 is optimal.
    
    So 2x2*2x7 with Strassen: 7+7+7+4 = 25.
    
    Total for 2x4x7: 25 + 25 = 50.
    
    That's better than 56! Score would be 56/50 = 1.12.
    
    Let me implement this.
    
    A (2x4) = [A1, A2] where A1, A2 are 2x2
    B (4x7) = [B1; B2] where B1, B2 are 2x7
    
    For each Ai (2x2) and Bi (2x7):
    Split Bi into [Bi1, Bi2, Bi3, Bi4] where Bi1, Bi2, Bi3 are 2x2 and Bi4 is 2x1
    
    For Ai * Bi1 (2x2 * 2x2): use Strassen (rank 7)
    For Ai * Bi2 (2x2 * 2x2): use Strassen (rank 7)
    For Ai * Bi3 (2x2 * 2x2): use Strassen (rank 7)
    For Ai * Bi4 (2x2 * 2x1): use schoolbook (rank 4)
    
    Total per (Ai, Bi) pair: 7+7+7+4 = 25
    Total: 25 + 25 = 50
    
    Now let me implement Strassen for 2x2*2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    Strassen:
    m1 = (a+d)*(e+h)
    m2 = (c+d)*e
    m3 = a*(f-h)
    m4 = d*(g-e)
    m5 = (a+b)*h
    m6 = (c-a)*(e+f)
    m7 = (b-d)*(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5],
         [m2+m4, m1-m2+m3+m6]]
    
    Now I need to express this as a tensor decomposition.
    
    For the full 2x4x7 problem:
    U has shape (2*4, rank) = (8, rank)
    V has shape (4*7, rank) = (28, rank)
    W has shape (7*2, rank) = (14, rank)
    
    The convention is W[k*n+i, r] where n=2, so W has 7*2=14 rows.
    Row index: k*n+i where k is column index (0..6) and i is row index (0..1).
    
    Let me carefully index everything.
    
    A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    B is 4x7: B[j,k] where j in {0,1,2,3}, k in {0,...,6}
    C is 2x7: C[i,k] where i in {0,1}, k in {0,...,6}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Partition:
    A1 = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    A2 = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    B1 = B[0:2, :] = [[B[0,0], ..., B[0,6]], [B[1,0], ..., B[1,6]]]
    B2 = B[2:4, :] = [[B[2,0], ..., B[2,6]], [B[3,0], ..., B[3,6]]]
    
    C = A1*B1 + A2*B2
    
    For A1*B1:
    A1 = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    B1 = 2x7, split into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)
    B1a = B[0:2, 0:2], B1b = B[0:2, 2:4], B1c = B[0:2, 4:6], B1d = B[0:2, 6:7]
    
    For A2*B2:
    A2 = [[e, f], [g, h]] where e=A[0,2], f=A[0,3], g=A[1,2], h=A[1,3]
    B2 = 2x7, split into B2a (2x2), B2b (2x2), B2c (2x2), B2d (2x1)
    B2a = B[2:4, 0:2], B2b = B[2:4, 2:4], B2c = B[2:4, 4:6], B2d = B[2:4, 6:7]
    
    Now let's build the decomposition.
    
    For each Strassen multiplication, we have 7 terms.
    For each schoolbook 2x2*2x1, we have 4 terms.
    
    Total: 2*(7+7+7+4) = 50 terms.
    
    Let me code this up.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # (8, 50)
    V = [[0] * rank for _ in range(m * p)]  # (28, 50)
    W = [[0] * rank for _ in range(p * n)]  # (14, 50)
    
    col = 0
    
    def add_strassen_2x2x2(A_base, B_base, C_base, col_start):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        A_base: (i, j) offset in A for the 2x2 block
        B_base: (j, k) offset in B for the 2x2 block  
        C_base: (i, k) offset in C for the 2x2 block
        
        A elements: A[i_base, j_base], A[i_base, j_base+1], A[i_base+1, j_base], A[i_base+1, j_base+1]
        B elements: B[j_base, k_base], B[j_base, k_base+1], B[j_base+1, k_base], B[j_base+1, k_base+1]
        C elements: C[i_base, k_base], C[i_base, k_base+1], C[i_base+1, k_base], C[i_base+1, k_base+1]
        """
        # a=A[ib,jb], b=A[ib,jb+1], c=A[ib+1,jb], d=A[ib+1,jb+1]
        # e=B[jb,kb], f=B[jb,kb+1], g=B[jb+1,kb], h=B[jb+1,kb+1]
        
        # m1 = (a+d)*(e+h)
        # m2 = (c+d)*e
        # m3 = a*(f-h)
        # m4 = d*(g-e)
        # m5 = (a+b)*h
        # m6 = (c-a)*(e+f)
        # m7 = (b-d)*(g+h)
        
        # C[0,0] = m1+m4-m5+m7
        # C[0,1] = m3+m5
        # C[1,0] = m2+m4
        # C[1,1] = m1-m2+m3+m6
        
        # U row for A[i,j] is i*m+j
        # V row for B[j,k] is j*p+k
        # W row for C[i,k] is k*n+i
        
        # m1 = (a+d)*(e+h)
        # U: a and d, V: e and h
        U[A_base[0]*m + A_base[1]][col_start] = 1
        U[(A_base[0]+1)*m + (A_base[1]+1)][col_start] = 1
        V[B_base[0]*p + B_base[1]][col_start] = 1
        V[(B_base[0]+1)*p + (B_base[1]+1)][col_start] = 1
        W[C_base[1]*n + C_base[0]][col_start] = 1
        W[(C_base[1]+1)*n + (C_base[0]+1)][col_start] = 1
        col_start += 1
        
        # m2 = (c+d)*e
        U[(A_base[0]+1)*m + A_base[1]][col_start] = 1
        U[(A_base[0]+1)*m + (A_base[1]+1)][col_start] = 1
        V[B_base[0]*p + B_base[1]][col_start] = 1
        W[C_base[1]*n + (C_base[0]+1)][col_start] = 1
        col_start += 1
        
        # m3 = a*(f-h)
        U[A_base[0]*m + A_base[1]][col_start] = 1
        V[B_base[0]*p + (B_base[1]+1)][col_start] = 1
        V[(B_base[0]+1)*p + B_base[1]][col_start] = -1
        W[(C_base[1]+1)*n + C_base[0]][col_start] = 1
        col_start += 1
        
        # m4 = d*(g-e)
        U[(A_base[0]+1)*m + (A_base[1]+1)][col_start] = 1
        V[(B_base[0]+1)*p + B_base[1]][col_start] = 1
        V[B_base[0]*p + B_base[1]][col_start] = -1
        W[C_base[1]*n + C_base[0]][col_start] = 1
        W[(C_base[1]+1)*n + (C_base[0]+1)][col_start] = 1
        col_start += 1
        
        # m5 = (a+b)*h
        U[A_base[0]*m + A_base[1]][col_start] = 1
        U[A_base[0]*m + (A_base[1]+1)][col_start] = 1
        V[(B_base[0]+1)*p + (B_base[1]+1)][col_start] = 1
        W[C_base[1]*n + C_base[0]][col_start] = -1
        W[(C_base[1]+1)*n + (C_base[0]+1)][col_start] = 1
        col_start += 1
        
        # m6 = (c-a)*(e+f)
        U[(A_base[0]+1)*m + A_base[1]][col_start] = 1
        U[A_base[0]*m + A_base[1]][col_start] = -1
        V[B_base[0]*p + B_base[1]][col_start] = 1
        V[B_base[0]*p + (B_base[1]+1)][col_start] = 1
        W[(C_base[1]+1)*n + (C_base[0]+1)][col_start] = 1
        col_start += 1
        
        # m7 = (b-d)*(g+h)
        U[A_base[0]*m + (A_base[1]+1)][col_start] = 1
        U[(A_base[0]+1)*m + (A_base[1]+1)][col_start] = -1
        V[(B_base[0]+1)*p + B_base[1]][col_start] = 1
        V[(B_base[0]+1)*p + (B_base[1]+1)][col_start] = 1
        W[C_base[1]*n + C_base[0]][col_start] = 1
        col_start += 1
        
        return col_start
    
    def add_schoolbook_2x2x1(A_base, B_base, C_base, col_start):
        """
        Add schoolbook decomposition for 2x2 * 2x1.
        A is 2x2, B is 2x1, C is 2x1.
        """
        # C[0,0] = a*e + b*g
        # C[1,0] = c*e + d*g
        
        # a*e
        U[A_base[0]*m + A_base[1]][col_start] = 1
        V[B_base[0]*p + B_base[1]][col_start] = 1
        W[B_base[1]*n + A_base[0]][col_start] = 1
        col_start += 1
        
        # b*g
        U[A_base[0]*m + (A_base[1]+1)][col_start] = 1
        V[(B_base[0]+1)*p + B_base[1]][col_start] = 1
        W[B_base[1]*n + A_base[0]][col_start] = 1
        col_start += 1
        
        # c*e
        U[(A_base[0]+1)*m + A_base[1]][col_start] = 1
        V[B_base[0]*p + B_base[1]][col_start] = 1
        W[B_base[1]*n + (A_base[0]+1)][col_start] = 1
        col_start += 1
        
        # d*g
        U[(A_base[0]+1)*m + (A_base[1]+1)][col_start] = 1
        V[(B_base[0]+1)*p + B_base[1]][col_start] = 1
        W[B_base[1]*n + (A_base[0]+1)][col_start] = 1
        col_start += 1
        
        return col_start
    
    # Process A1*B1
    # A1 is at rows 0-1, cols 0-1 of A
    # B1 is at rows 0-1, cols 0-6 of B
    # C1 is at rows 0-1, cols 0-6 of C
    
    # B1 split: B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5), B1d (col 6)
    col = add_strassen_2x2x2((0, 0), (0, 0), (0, 0), col)
    col = add_strassen_2x2x2((0, 0), (0, 2), (0, 2), col)
    col = add_strassen_2x2x2((0, 0), (0, 4), (0, 4), col)
    col = add_schoolbook_2x2x1((0, 0), (0, 6), (0, 6), col)
    
    # Process A2*B2
    # A2 is at rows 0-1, cols 2-3 of A
    # B2 is at rows 2-3, cols 0-6 of B
    # C2 is at rows 0-1, cols 0-6 of C
    
    # B2 split: B2a (cols 0-1), B2b (cols 2-3), B2c (cols 4-5), B2d (col 6)
    col = add_strassen_2x2x2((0, 2), (2, 0), (0, 0), col)
    col = add_strassen_2x2x2((0, 2), (2, 2), (0, 2), col)
    col = add_strassen_2x2x2((0, 2), (2, 4), (0, 4), col)
    col = add_schoolbook_2x2x1((0, 2), (2, 6), (0, 6), col)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 120:
        return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    def verify():
        """Verify that U @ diag @ V^T reconstructs the multiplication tensor."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = delta_{i,k} (for matrix mult, T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k])
        # Actually, the decomposition says: for each column c, U[:,c] * V[:,c] * W[:,c] contributes to the tensor
        # T = sum_c outer(U[:,c], V[:,c], W[:,c])
        # And T should equal the matrix multiplication tensor.
        
        # Matrix multiplication tensor: T[i, j, k] = 1 if i corresponds to row of A, j to col of A / row of B, k to col of B / row of C
        # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the j index in some sense)
        # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Let's compute the decomposed tensor
        # T_decomp[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        
        T_decomp = np.zeros((n, m, p))
        for c in range(rank):
            uc = U_np[:, c]
            vc = V_np[:, c]
            wc = W_np[:, c]
            T_decomp += np.einsum('i,j,k->ijk', uc, vc, wc)
        
        # The target tensor: T[i,j,k] = 1 for all i,j,k
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_decomp, T_target, atol=1e-10)
    
    try:
        if not verify():
            # Fall back to schoolbook if verification fails
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
    except Exception:
        # If verification fails, fall back to schoolbook
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
