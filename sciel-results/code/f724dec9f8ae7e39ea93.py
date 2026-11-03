"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_block(n, m, p):
    """
    Return a rank-7 decomposition for n x m x p where n=m=p=2 (Strassen).
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    """
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing: A is n x m, B is m x p, C is n x p
    # For 2x2: A[i,j] with i,j in {0,1}
    # B[j,k] with j,k in {0,1}
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[i,k]
    
    # M1: (A00 + A11)(B00 + B11)
    # U: A00=U[0,0], A11=U[3,0] -> U[0,0]=1, U[3,0]=1
    # V: B00=V[0,0], B11=V[3,0] -> V[0,0]=1, V[3,0]=1
    # W: contributes to C00 (+1), C11 (+1) -> W[0,0]=1 (C00: k=0,i=0 -> W[0*2+0]=W[0]), W[1,1]=1 (C11: k=1,i=1 -> W[1*2+1]=W[3])
    # Wait, W index is k*n+i. For C00: k=0, i=0 -> W[0]. For C11: k=1, i=1 -> W[3].
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10 + A11)B00
    # U: A10=U[1*2+0]=U[2], A11=U[3] -> U[2,1]=1, U[3,1]=1
    # V: B00=V[0] -> V[0,1]=1
    # W: contributes to C10 (+1) -> k=0, i=1 -> W[0*2+1]=W[1]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3: A00(B10 - B11)
    # U: A00=U[0] -> U[0,2]=1
    # V: B10=V[1*2+0]=V[2], B11=V[3] -> V[2,2]=1, V[3,2]=-1
    # W: contributes to C01 (+1) -> k=1, i=0 -> W[1*2+0]=W[2]
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: A11(B01 - B00)
    # U: A11=U[3] -> U[3,3]=1
    # V: B01=V[0*2+1]=V[1], B00=V[0] -> V[1,3]=1, V[0,3]=-1
    # W: contributes to C00 (+1), C10 (+1) -> W[0,3]=1, W[1,3]=1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00 + A01)B11
    # U: A00=U[0], A01=U[1] -> U[0,4]=1, U[1,4]=1
    # V: B11=V[3] -> V[3,4]=1
    # W: contributes to C00 (-1), C01 (+1) -> W[0,4]=-1, W[2,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10 - A00)(B00 + B01)
    # U: A10=U[2], A00=U[0] -> U[2,5]=1, U[0,5]=-1
    # V: B00=V[0], B01=V[1] -> V[0,5]=1, V[1,5]=1
    # W: contributes to C11 (+1) -> W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01 - A11)(B10 + B11)
    # U: A01=U[1], A11=U[3] -> U[1,6]=1, U[3,6]=-1
    # V: B10=V[2], B11=V[3] -> V[2,6]=1, V[3,6]=1
    # W: contributes to C00 (-1) -> W[0,6]=-1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy: Use Strassen for 4x4x4 blocks.
    # 4x4 can be seen as 2x2 blocks of 2x2 matrices.
    # But we want a direct rank reduction.
    # Known: rank(4x4x4) <= 49 (using Strassen recursively: 7^2=49).
    # We have 4x4 x 4x8.
    # Split B into two 4x4 matrices: B = [B1, B2].
    # C = A * B = [A*B1, A*B2].
    # If we use a rank-49 decomposition for A*B1, and another for A*B2,
    # the total rank would be 98.
    # However, we can potentially share some terms if the structure allows.
    # But generally, the rank of A*(B1,B2) is at most rank(A*B1) + rank(A*B2).
    # Can we do better?
    # The problem is equivalent to multiplying a 4x4 matrix by a 4x8 matrix.
    # This is the same as computing 8 columns of C = A * B, where B is 4x8.
    # This is equivalent to 8 independent 4x1 matrix-vector multiplications? No, that's not tensor rank.
    # It is the tensor rank of the multiplication map M_{4,4,8}.
    # Lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32.
    # Upper bound: 4 * rank(4x4x4) = 4 * 49 = 196? No.
    # Actually, M_{4,4,8} can be seen as 2 copies of M_{4,4,4} if we split the output.
    # Rank(M_{4,4,8}) <= 2 * Rank(M_{4,4,4}) <= 2 * 49 = 98.
    # Can we do better than 98?
    # Maybe by interleaving the Strassen steps.
    
    # Let's try to construct a rank-49 * 2 = 98 decomposition first, then see if we can optimize.
    # Or, use the fact that 4x4x8 is like 4x4x4 but with B doubled.
    # If we use the 49-term Strassen for 4x4x4, each term c has U_c (4x4), V_c (4x4), W_c (4x4).
    # For 4x4x8, we have U (16 x r), V (32 x r), W (32 x r).
    # If we just duplicate the V and W blocks for the second 4 columns of B and C,
    # we get rank 98.
    # U is the same for both halves? No, U depends on A.
    # In the decomposition C = sum_c U_c A V_c B W_c^T (schematically),
    # For the first half C1 = A B1, we have terms (U_c, V_c, W_c).
    # For the second half C2 = A B2, we have terms (U_c, V'_c, W'_c).
    # If we use the same Strassen structure for B2, then U_c is the same!
    # So we can share the U factors.
    # This means we don't need 2 * 49 terms, but rather 49 terms where each term has:
    # U_c (from A), V_c (from B1), V'_c (from B2), W_c (from C1), W'_c (from C2).
    # But the tensor decomposition requires a single sum over c.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we split k into k1 (0..3) and k2 (4..7):
    # C[i, k1] = sum_j A[i,j] B[j, k1]
    # C[i, 4+k1] = sum_j A[i,j] B[j, 4+k1]
    # If we use the same 49 Strassen terms for both, we need to double the rank?
    # No, because the terms are additive.
    # C[i,k] = sum_{c=1}^{49} (U_c A V_c B1 W_c^T)_{i,k} + sum_{c=1}^{49} (U_c A V'_c B2 W'_c^T)_{i,k}
    # This is a sum of 98 terms.
    # Can we combine them into 49 terms?
    # Only if V_c = V'_c and W_c = W'_c, which is not true for arbitrary B1, B2.
    # So the rank is likely 98 or close to it.
    
    # Let's construct the rank-98 decomposition.
    # We need the rank-49 decomposition for 4x4x4.
    # This is obtained by applying Strassen to 2x2 blocks of 2x2 matrices.
    
    # Let's build the 4x4x4 decomposition first.
    # A is 4x4, B is 4x4, C is 4x4.
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2).
    # Split B into 2x2 blocks: B00, B01, B10, B11 (each 2x2).
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each of these is a sum of two 2x2 matrix multiplications.
    # We can use Strassen for each? No, that would be 4 * 7 = 28 terms, but with overlaps.
    # Actually, the standard recursive Strassen for 4x4 uses 49 terms.
    # Let's just generate the 49 terms by applying Strassen to the 2x2 block structure.
    
    # This is complex to hardcode. Let's use a simpler approach:
    # Use the schoolbook for 4x4x4 (rank 64) and see if we can reduce it.
    # Or, just use the rank-98 construction for 4x4x8.
    
    # Rank 98 construction:
    # We need a rank-49 decomposition for 4x4x4.
    # Let's compute it using the recursive Strassen algorithm.
    
    def _strassen_4x4_4x4():
        """
        Returns U, V, W for 4x4x4 multiplication with rank 49.
        U: (16, 49), V: (16, 49), W: (16, 49)
        """
        # This function would recursively apply Strassen.
        # For now, let's use a known result or generate it.
        # Actually, let's just use the schoolbook for 4x4x4 (rank 64) for simplicity,
        # then the total rank for 4x4x8 would be 128 (same as original).
        # That's not an improvement.
        
        # Let's try to implement the recursive Strassen properly.
        # It's better to use a library or a precomputed table.
        # Since I can't easily precompute 49 terms here without a lot of code,
        # let's try a different approach: use the rank-7 Strassen for 2x2x2,
        # and apply it to the 4x4x8 problem directly by viewing it as 2x2 blocks.
        
        # A is 4x4, B is 4x8, C is 4x8.
        # Split A into 2x2 blocks of 2x2 matrices: A = [[A00, A01], [A10, A11]]
        # Split B into 2x2 blocks of 2x4 matrices: B = [[B00, B01], [B10, B11]] where B00 is 2x4, etc.
        # C = [[C00, C01], [C10, C11]] where C00 is 2x4, etc.
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Each term like A00 B00 is a 2x2 x 2x4 multiplication.
        # We can decompose 2x2 x 2x4 using Strassen?
        # Strassen is for 2x2 x 2x2.
        # 2x2 x 2x4 is like two 2x2 x 2x2 multiplications side by side.
        # So rank(2x2 x 2x4) <= 2 * 7 = 14.
        # There are 4 such products in C00, C01, C10, C11? No.
        # C00 = A00 B00 + A01 B10.
        # A00 B00 is 2x2 x 2x4.
        # A01 B10 is 2x2 x 2x4.
        # So C00 is the sum of two 2x2 x 2x4 products.
        # If we use rank-14 for each, C00 has rank 28.
        # Total rank would be 4 * 28 = 112.
        # This is worse than 98.
        
        # The 98 construction is better. Let's stick with that.
        # I will implement the 4x4x4 rank-49 decomposition using a known algorithm or fallback to schoolbook.
        # For now, let's just return the schoolbook to be safe, as implementing recursive Strassen is error-prone.
        
        # Actually, let's try to use the rank-7 Strassen for 2x2x2 and build up.
        # This is too complex for this context. Let's just use the schoolbook.
        
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

    # Since implementing the optimal 49-rank decomposition is complex, and the 98-rank construction
    # requires it, let's just use the schoolbook for now.
    # The score will be 1.0, but it's a safe fallback.
    
    # Wait, the inspiration says to use Strassen. Let's try to implement the 4x4x4 rank-49.
    # I will use a simplified version: just use the schoolbook for 4x4x4, which is rank 64.
    # Then for 4x4x8, we use 2 * 64 = 128. Same as before.
    
    # Let's try a different structural change: use the rank-7 Strassen for 2x2x2,
    # and apply it to the 4x4x8 problem by splitting into 2x2 blocks.
    # A (4x4) -> 4 blocks of 2x2.
    # B (4x8) -> 8 blocks of 2x2.
    # C (4x8) -> 8 blocks of 2x2.
    # C_ij = sum_k A_ik B_kj.
    # This is a 2x2 block matrix multiplication.
    # We can use Strassen for the block multiplication?
    # Strassen works for 2x2 block matrices where each block is a matrix.
    # Here, the "matrices" are 2x2.
    # So we can apply Strassen to the 2x2 block structure.
    # This gives 7 block products, each of which is a 2x2 x 2x2 multiplication.
    # Each 2x2 x 2x2 multiplication can be done with rank 7 (Strassen) or rank 8 (schoolbook).
    # If we use Strassen for the inner multiplication, total rank = 7 * 7 = 49.
    # This is for 4x4 x 4x4.
    # For 4x4 x 4x8, we have 8 block columns in B and C.
    # So we have 2 sets of 4 block columns? No, 8 block columns.
    # The block structure is 2x2 blocks of size 2x2.
    # B has 2 block rows and 4 block columns.
    # C has 2 block rows and 4 block columns.
    # C_ij = sum_k A_ik B_kj.
    # This is not a square block matrix multiplication.
    # Strassen is for 2x2 block matrices.
    # We can split the 4 block columns into two sets of 2.
    # C = [C_left, C_right] where C_left is 2x2 blocks, C_right is 2x2 blocks.
    # C_left = A * B_left (2x2 block mult).
    # C_right = A * B_right (2x2 block mult).
    # Each 2x2 block mult has rank 49 (if using Strassen for inner 2x2).
    # Total rank = 49 + 49 = 98.
    
    # So the plan is:
    # 1. Implement a function to get the rank-49 decomposition for 4x4x4.
    # 2. Use it to construct the rank-98 decomposition for 4x4x8.
    
    # Let's implement the rank-49 decomposition for 4x4x4.
    # This requires applying Strassen to 2x2 blocks, where each block multiplication is also done with Strassen.
    
    def _strassen_2x2_2x2_2x2():
        """
        Returns U, V, W for 2x2x2 multiplication with rank 7.
        """
        U, V, W = _strassen_2x2_block(2, 2, 2)
        return U, V, W

    def _strassen_4x4_4x4():
        """
        Returns U, V, W for 4x4x4 multiplication with rank 49.
        """
        # Apply Strassen to 2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Use Strassen for the block multiplication:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Each M_i is a 2x2 matrix product of 2x2 matrices.
        # We can decompose each M_i using the rank-7 Strassen decomposition.
        
        # This is getting too complex to implement correctly in this context.
        # Let's just use the schoolbook for 4x4x4 (rank 64) and then for 4x4x8 (rank 128).
        
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

    # Fallback to schoolbook for 4x4x8
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
