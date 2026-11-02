import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Tensor T_{(i,j), (j',k), (k',i')} = 1 if i==i', j==j', k==k'
        """
        r = U.shape[1]
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        # Check all tensor entries
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    i1 = u // m
                    j1 = u % m
                    j2 = v // p
                    k1 = v % p
                    k2 = w // n
                    i2 = w % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    s = float(np.dot(U[u, :], V[v, :] * W[w, :]))
                    if abs(s - target) > 1e-9:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strategy: Use Strassen's algorithm for 2x2x2 blocks
    # 2x3x6 = 2x3x(3*2)
    # Split B into 3 blocks of 3x2
    # For each block, use a rank-7 Strassen decomposition for 2x2x2
    # But we need 2x3x2, not 2x2x2.
    # 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # C = A * B = [A[:,0], A[:,1], A[:,2]] * [[B[0,0], B[0,1]], [B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # C = A[:,0] @ [[B[0,0], B[0,1]]] + A[:,1] @ [[B[1,0], B[1,1]]] + A[:,2] @ [[B[2,0], B[2,1]]]
    # This is a 3-term sum of 2x1 @ 1x2 outer products.
    # Each term A[:,j] @ B[j,:] is rank-1 in the bilinear sense.
    # So 2x3x2 has rank at most 3 (the schoolbook rank is 3*2*2=12, but we can do better).
    # Actually, 2x3x2: A is 2x3, B is 3x2.
    # C_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}
    # This is just 3 matrix products of 2x1 and 1x2, each contributing to C.
    # The tensor T_{(i,j), (j',k), (k',i')} = 1 if i==i', j==j', k==k'.
    # For fixed j, the contribution is A_{i,j} * B_{j,k} -> C_{i,k}.
    # So for each j, we have a rank-1 term: U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    # That gives rank 3 for 2x3x2!
    # Wait, that's the schoolbook for 2x3x2 which has rank 2*3*2=12.
    # No, the schoolbook has one term for each (i,j,k). For 2x3x2, that's 2*3*2=12 terms.
    # But C_{i,k} = sum_j A_{i,j} B_{j,k}. Each term A_{i,j}B_{j,k} is a separate product.
    # So the schoolbook rank is indeed 12 for 2x3x2.
    # Can we do better?
    # The rank of matrix multiplication M_{n,m,p} is known.
    # For M_{2,3,2}, the rank is at most 12 (schoolbook) but might be lower.
    # Actually, I think the rank of M_{2,3,2} is 12 or close to it.
    # Let me think differently.
    # 2x3x6 = (2x3) x (3x6)
    # We can write this as 2x3x(3*2) = 3 blocks of 2x3x2.
    # If we use the schoolbook for each 2x3x2 block, we get 3*12=36.
    # If we can find a rank-r decomposition for 2x3x2, we get 3*r for 2x3x6.
    # The best known rank for 2x3x2... let me just use the schoolbook and see if I can improve.
    # Actually, let me try to use Strassen for the 2x2 parts.
    # 2x3x2: A is 2x3 = [A_left | A_right] where A_left is 2x1, A_right is 2x2.
    # B is 3x2 = [[B_top], [B_bottom]] where B_top is 1x2, B_bottom is 2x2.
    # C = A_left @ B_top + A_right @ B_bottom.
    # A_right @ B_bottom is 2x2x2 matrix multiplication, which has rank 7 (Strassen).
    # A_left @ B_top is 2x1x2, which has rank 2 (schoolbook: 2*1*2=2, and it's just two outer products).
    # So total rank is 7 + 2 = 9 for 2x3x2? No, that's not quite right.
    # The tensors don't add like that because the indices overlap.
    # Let me just construct it directly.
    # For 2x3x2, let me use the schoolbook (rank 12) and see.
    # Then 2x3x6 would be 3*12 = 36.
    # But the inspiration says rank 34 is possible. Let me try to find a better decomposition.
    # Actually, the measured finding says "Using block decomposition with Strassen's algorithm for 2x2x2 sub-problems reduces the rank from 36 to 33 by exploiting the structure of the 2x3x2 multiplication (rank 11 vs 12)."
    # So 2x3x2 has rank 11, not 12!
    # Let me try to construct a rank-11 decomposition for 2x3x2.
    # Then 2x3x6 would be 3*11 = 33.
    
    # Let me try to construct 2x3x2 with rank 11.
    # A is 2x3, B is 3x2, C is 2x2.
    # Schoolbook: 12 terms.
    # Can we save one term?
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0} + A_{0,2}B_{2,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1} + A_{0,2}B_{2,1}
    # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0} + A_{1,2}B_{2,0}
    # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1} + A_{1,2}B_{2,1}
    # That's 12 terms. To get rank 11, we need to combine two terms.
    # Using Strassen-like tricks for the 2x2 part.
    # Let A_right = A[:,1:3] (2x2), B_bottom = B[1:3,:] (2x2).
    # C_bottom = A_right @ B_bottom (2x2 result, but it adds to C).
    # Actually, C = A[:,0] @ B[0,:] + A[:,1:] @ B[1:,:]
    # A[:,0] @ B[0,:] is 2x1 @ 1x2 = 2x2 matrix.
    # A[:,1:] @ B[1:,:] is 2x2 @ 2x2 = 2x2 matrix.
    # So C = outer(A[:,0], B[0,:]) + A[:,1:] @ B[1:,:]
    # The outer product has rank 2 (two bilinear terms: A_{0,0}B_{0,0}->C_{0,0}, A_{0,0}B_{0,1}->C_{0,1}, A_{1,0}B_{0,0}->C_{1,0}, A_{1,0}B_{0,1}->C_{1,1}).
    # Wait, that's 4 terms for the outer product alone!
    # A_{0,0}*B_{0,0} -> C_{0,0}
    # A_{0,0}*B_{0,1} -> C_{0,1}
    # A_{1,0}*B_{0,0} -> C_{1,0}
    # A_{1,0}*B_{0,1} -> C_{1,1}
    # And A[:,1:] @ B[1:,:] is 2x2x2 which has 4 terms in schoolbook, but rank 7 with Strassen.
    # So total is 4 + 7 = 11!
    # Let me construct this.
    
    def construct_2x3x2_rank11():
        """Construct rank-11 decomposition for 2x3x2.
        A: 2x3, B: 3x2, C: 2x2
        C = A[:,0] @ B[0,:] + A[:,1:] @ B[1:,:]
        """
        # U: (2*3, 11) = (6, 11)
        # V: (3*2, 11) = (6, 11)
        # W: (2*2, 11) = (4, 11)
        rank = 11
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        col = 0
        
        # Part 1: A[:,0] @ B[0,:] (4 terms)
        # A_{0,0} * B_{0,0} -> C_{0,0}
        U[0*3+0, col] = 1  # A[0,0]
        V[0*2+0, col] = 1  # B[0,0]
        W[0*2+0, col] = 1  # C[0,0]
        col += 1
        # A_{0,0} * B_{0,1} -> C_{0,1}
        U[0*3+0, col] = 1  # A[0,0]
        V[0*2+1, col] = 1  # B[0,1]
        W[0*2+1, col] = 1  # C[0,1]
        col += 1
        # A_{1,0} * B_{0,0} -> C_{1,0}
        U[1*3+0, col] = 1  # A[1,0]
        V[0*2+0, col] = 1  # B[0,0]
        W[1*2+0, col] = 1  # C[1,0]
        col += 1
        # A_{1,0} * B_{0,1} -> C_{1,1}
        U[1*3+0, col] = 1  # A[1,0]
        V[0*2+1, col] = 1  # B[0,1]
        W[1*2+1, col] = 1  # C[1,1]
        col += 1
        
        # Part 2: A[:,1:] @ B[1:,:] (2x2x2, Strassen rank 7)
        # A_sub = A[:,1:3], indices (i, j+1) for i in {0,1}, j in {0,1}
        # B_sub = B[1:3,:], indices (j+1, k) for j in {0,1}, k in {0,1}
        # C_sub = A_sub @ B_sub, adds to C
        # Strassen:
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B01-B11)
        # P4 = A11*(B10-B00)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # A00 = A[0,1], A01 = A[0,2], A10 = A[1,1], A11 = A[1,2]
        # B00 = B[1,0], B01 = B[1,1], B10 = B[2,0], B11 = B[2,1]
        
        # P1 = (A[0,1]+A[1,2])*(B[1,0]+B[2,1])
        U[0*3+1, col] = 1  # A[0,1]
        U[1*3+2, col] = 1  # A[1,2]
        V[1*2+0, col] = 1  # B[1,0]
        V[2*2+1, col] = 1  # B[2,1]
        W[0*2+0, col] = 1  # C[0,0] += P1
        W[1*2+1, col] = 1  # C[1,1] += P1
        col += 1
        
        # P2 = (A[1,1]+A[1,2])*B[1,0]
        U[1*3+1, col] = 1  # A[1,1]
        U[1*3+2, col] = 1  # A[1,2]
        V[1*2+0, col] = 1  # B[1,0]
        W[1*2+0, col] = 1  # C[1,0] += P2
        W[1*2+1, col] = -1 # C[1,1] -= P2
        col += 1
        
        # P3 = A[0,1]*(B[1,1]-B[2,1])
        U[0*3+1, col] = 1  # A[0,1]
        V[1*2+1, col] = 1  # B[1,1]
        V[2*2+1, col] = -1 # -B[2,1]
        W[0*2+1, col] = 1  # C[0,1] += P3
        W[1*2+1, col] = 1  # C[1,1] += P3
        col += 1
        
        # P4 = A[1,2]*(B[2,0]-B[1,0])
        U[1*3+2, col] = 1  # A[1,2]
        V[2*2+0, col] = 1  # B[2,0]
        V[1*2+0, col] = -1 # -B[1,0]
        W[0*2+0, col] = 1  # C[0,0] += P4
        W[1*2+0, col] = 1  # C[1,0] += P4
        col += 1
        
        # P5 = (A[0,1]+A[0,2])*B[2,1]
        U[0*3+1, col] = 1  # A[0,1]
        U[0*3+2, col] = 1  # A[0,2]
        V[2*2+1, col] = 1  # B[2,1]
        W[0*2+0, col] = -1 # C[0,0] -= P5
        W[0*2+1, col] = 1  # C[0,1] += P5
        col += 1
        
        # P6 = (A[1,1]-A[0,1])*(B[1,0]+B[1,1])
        U[1*3+1, col] = 1  # A[1,1]
        U[0*3+1, col] = -1 # -A[0,1]
        V[1*2+0, col] = 1  # B[1,0]
        V[1*2+1, col] = 1  # B[1,1]
        W[1*2+1, col] = 1  # C[1,1] += P6
        col += 1
        
        # P7 = (A[0,2]-A[1,2])*(B[2,0]+B[2,1])
        U[0*3+2, col] = 1  # A[0,2]
        U[1*3+2, col] = -1 # -A[1,2]
        V[2*2+0, col] = 1  # B[2,0]
        V[2*2+1, col] = 1  # B[2,1]
        W[0*2+0, col] = 1  # C[0,0] += P7
        col += 1
        
        assert col == 11, f"Expected 11 terms, got {col}"
        return U, V, W
    
    # Verify the rank-11 2x3x2 decomposition
    U2, V2, W2 = construct_2x3x2_rank11()
    if not verify(U2, V2, W2, 2, 3, 2):
        # Fall back to schoolbook for 2x3x2
        U2, V2, W2 = schoolbook(2, 3, 2)
        r2 = 12
    else:
        r2 = 11
    
    # Now assemble 2x3x6 from three 2x3x2 blocks
    # B is 3x6 = [B0, B1, B2] where each Bj is 3x2
    # C is 2x6 = [C0, C1, C2] where each Cj is 2x2
    # Cj = A @ Bj
    # U is the same for all blocks: A
    # V for block j is Bj
    # W for block j is Cj
    
    total_rank = 3 * r2
    U_comb = np.zeros((2*3, total_rank))
    V_comb = np.zeros((3*6, total_rank))
    W_comb = np.zeros((6*2, total_rank))
    
    U_comb[:, :] = np.tile(U2, (1, 3))
    
    for j in range(3):
        V_comb[:, j*r2:(j+1)*r2] = V2
        W_comb[:, j*r2:(j+1)*r2] = W2
    
    if verify(U_comb, V_comb, W_comb, 2, 3, 6):
        return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
    else:
        # Final fallback: schoolbook
        U, V, W = schoolbook(2, 3, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
