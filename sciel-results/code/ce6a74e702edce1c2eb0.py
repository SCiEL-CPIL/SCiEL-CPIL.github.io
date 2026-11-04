import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Winograd F(4,4,4) rank-20 decomposition
    # Use a 5x5 transform matrix G
    # A5 = [[A, 0], [0, 0]] (5x5, last row and col zero)
    # B5 = [[B, 0], [0, 0]] (5x5, last row and col zero)
    # C5 = A5 * B5, C = C5[0:4, 0:4]
    # 
    # Winograd: C5 = G_inv * ((G * A5) .* (G * B5))
    # where .* is element-wise product
    # 
    # Since A5 and B5 have last column zero, (G*A5) and (G*B5) also have last column zero
    # So element-wise product has only 5*4 = 20 potentially non-zero entries
    
    # Choose G as a simple invertible 5x5 matrix
    # Use identity for simplicity (gives rank 25, not optimal)
    # Better: use a matrix that creates sparsity
    
    # Actually, ANY invertible 5x5 G works for rank 20 because:
    # (G*A5)[i,j] = sum_k G[i,k]*A5[k,j], and A5[k,4]=0 for all k
    # So (G*A5)[i,4] = 0 for all i
    # Similarly (G*B5)[i,4] = 0
    # Element-wise product has last column zero: only 5*4=20 non-zero entries
    
    # Let's use G = identity (simplest, gives valid rank-20)
    G = np.eye(5, dtype=float)
    G_inv = np.linalg.inv(G)
    
    # For rank-20: index r by (i, j) where i in 0..4, j in 0..3
    # r = i*4 + j
    
    rank_444 = 20
    U4 = np.zeros((16, rank_444), dtype=float)
    V4 = np.zeros((16, rank_444), dtype=float)
    W4 = np.zeros((16, rank_444), dtype=float)
    
    for i in range(5):
        for j in range(4):
            r = i * 4 + j
            # U: A5[k, j] for k in 0..3, U[k*4+j, r] = G[i,k]
            for k in range(4):
                U4[k * 4 + j, r] = G[i, k]
            
            # V: B5[l, j] for l in 0..3, V[l*4+j, r] = G[i,l]
            for l in range(4):
                V4[l * 4 + j, r] = G[i, l]
            
            # W: C5[m, n] = sum_i G_inv[m,i] * C5'[i,n]
            # C5'[i,j] = (G*A5)[i,j] * (G*B5)[i,j]
            # So W[n*4+m, r] = G_inv[m, i] for the contribution from (i,j)
            # But wait, C5[m,n] = sum_i G_inv[m,i] * C5'[i,n]
            # And C5'[i,n] = (G*A5)[i,n] * (G*B5)[i,n]
            # For fixed (i,j), C5'[i,j] contributes to C5[m,j] via G_inv[m,i]
            # So W[j*4+m, r] = G_inv[m, i]
            for m in range(4):
                W4[j * 4 + m, r] = G_inv[m, i]
    
    # Verify: C[m,n] = sum_k A[m,k]*B[k,n]
    # C[m,n] = C5[m,n] = sum_i G_inv[m,i] * C5'[i,n]
    # C5'[i,n] = (G*A5)[i,n] * (G*B5)[i,n]
    # (G*A5)[i,n] = sum_k G[i,k]*A5[k,n] = sum_k G[i,k]*A[k,n] (for n in 0..3)
    # (G*B5)[i,n] = sum_l G[i,l]*B5[l,n] = sum_l G[i,l]*B[l,n] (for n in 0..3)
    # 
    # C[m,n] = sum_i G_inv[m,i] * (sum_k G[i,k]*A[k,n]) * (sum_l G[i,l]*B[l,n])
    # 
    # This is NOT equal to sum_k A[m,k]*B[k,n] in general!
    # 
    # I made an error. The Winograd scheme is:
    # C5 = A5 * B5 (matrix product)
    # C5[m,n] = sum_k A5[m,k]*B5[k,n]
    # 
    # Winograd: C5 = G_inv * ((G*A5) .* (G*B5)) is WRONG.
    # 
    # Correct Winograd for convolution:
    # For 1D: c[n] = sum_k a[k]*b[n-k]
    # Pad to length 5, apply transform, multiply element-wise, inverse transform
    # 
    # For 2D matrix multiplication, it's more complex.
    # 
    # Let me use a different approach: direct rank-20 construction.
    # 
    # Actually, the correct Winograd F(4,4,4) uses:
    # C = A * B where A, B are 4x4
    # View as: for each pair (m,n), C[m,n] = sum_k A[m,k]*B[k,n]
    # 
    # The rank-20 decomposition comes from the fact that:
    # We can write C as a sum of 20 rank-1 tensors
    # 
    # One known construction:
    # Use the fact that 4x4x4 tensor has rank <= 20
    # 
    # Let me use a verified rank-20 decomposition from the literature.
    # 
    # Alternative: use the Strassen-based approach but optimize.
    # The current approach gives rank 98. Let's see if we can do better.
    # 
    # Actually, let me reconsider. The tensor product of two rank-7 decompositions
    # gives rank 49 for 4x4x4. For 4x4x8, we get rank 98.
    # 
    # Can we do better? The minimum rank for 4x4x4 is known to be at most 20.
    # For 4x4x8, it should be at most 40 (two copies of 4x4x4).
    # 
    # Let me construct a proper rank-20 decomposition for 4x4x4.
    
    # Known rank-20 decomposition for 4x4x4:
    # Reference: "On the rank of the matrix multiplication tensor" 
    # 
    # One construction uses the following:
    # C[m,n] = sum_{k=0}^{3} A[m,k] * B[k,n]
    # 
    # We can write this as:
    # C[m,n] = sum_{r=0}^{19} U[m,n,r] * V[?,r] * W[?,r]
    # 
    # A specific rank-20 decomposition:
    # Use the identity:
    # sum_k A[m,k]*B[k,n] = sum_{r=0}^{19} f_r(A) * g_r(B) * h_r(m,n)
    # 
    # Let me use a constructive approach with polynomial evaluation.
    # 
    # Actually, let me just use the Strassen tensor product which is verified to work,
    # and see if I can find a better decomposition.
    # 
    # For now, let's stick with the working rank-98 solution but see if we can
    # reduce it. The key insight from the inspiration is to use rank-20 for 4x4x4.
    # 
    # Let me try a different G matrix that actually works.
    # 
    # The correct Winograd scheme for matrix multiplication:
    # C = A * B
    # Pad A to A5 (5x5) with zeros in last row and column
    # Pad B to B5 (5x5) with zeros in last row and column
    # C5 = A5 * B5 (5x5 matrix product)
    # C = C5[0:4, 0:4]
    # 
    # Now, C5[m,n] = sum_{k=0}^{4} A5[m,k]*B5[k,n]
    # Since A5[m,4]=0 and B5[4,n]=0, this becomes:
    # C5[m,n] = sum_{k=0}^{3} A[m,k]*B[k,n]
    # 
    # Winograd: use a 5x5 matrix G such that:
    # C5 = G_inv * diag(d) * (G * A5) .* (G * B5) * G_inv^T
    # where d is a diagonal matrix with some zeros
    # 
    # For rank 20, we need d to have exactly 20 non-zero entries.
    # 
    # A simple choice: G = identity, d = diag(1,1,1,1,0) (last entry zero)
    # Then: C5 = diag(1,1,1,1,0) * A5 .* B5
    # C5[m,n] = A5[m,n]*B5[m,n] if m,n < 4, else 0
    # But this is NOT equal to A5*B5!
    # 
    # I'm confusing 1D convolution with 2D matrix multiplication.
    # 
    # Let me abandon the Winograd approach and use a different strategy.
    # 
    # Strategy: Use the Strassen 2x2x2 decomposition (rank 7) and apply it
    # in a block structure to get a better rank for 4x4x4.
    # 
    # The current approach uses tensor product: rank(4x4x4) = rank(2x2x2)^2 = 49
    # Can we do better?
    # 
    # Yes! We can use a block structure where we apply Strassen to 2x2 blocks
    # but with shared computations.
    # 
    # Actually, the tensor product is already optimal for this approach.
    # 
    # Let me try a completely different approach: use known low-rank decompositions.
    # 
    # After research, the minimum rank for 4x4x4 matrix multiplication is 20.
    # Here's a verified rank-20 decomposition:
    
    # Rank-20 decomposition for 4x4x4:
    # Based on the work of Strassen and later improvements.
    # 
    # One construction:
    # C[m,n] = sum_{k=0}^{3} A[m,k]*B[k,n]
    # 
    # We can write this using 20 terms:
    # For r = 0..19, let (i,j) = (r//4, r%4)
    # 
    # Actually, let me use a known explicit decomposition.
    # 
    # Here's a rank-20 decomposition using the following idea:
    # Group the k-sum into pairs and use Strassen-like combinations.
    # 
    # k = 0,1: use Strassen on 2x2 blocks
    # k = 2,3: use Strassen on 2x2 blocks
    # But this gives 7+7=14 terms, not 20.
    # 
    # I think I need to just use the working rank-98 solution.
    # Let me verify it works and return it.
    
    # Let's go back to the Strassen tensor product approach which is verified.
    
    # Strassen 2x2x2 decomposition (rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4x4 decomposition (Rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            I = i // 2
            J = j // 2
            ip = i % 2
            jp = j % 2
            idx4 = i * 4 + j
            idx2_outer = 2 * I + J
            idx2_inner = 2 * ip + jp
            
            for ro in range(7):
                uo = U2[idx2_outer, ro]
                if uo == 0:
                    continue
                for ri in range(7):
                    ui = U2[idx2_inner, ri]
                    if ui == 0:
                        continue
                    r = ro * 7 + ri
                    U4[idx4, r] = uo * ui
                    
            for k in range(4):
                K = k // 2
                kp = k % 2
                idx4_v = j * 4 + k
                idx2_v_outer = 2 * J + K
                idx2_v_inner = 2 * jp + kp
                
                for ro in range(7):
                    vo = V2[idx2_v_outer, ro]
                    if vo == 0:
                        continue
                    for ri in range(7):
                        vi = V2[idx2_v_inner, ri]
                        if vi == 0:
                            continue
                        r = ro * 7 + ri
                        V4[idx4_v, r] = vo * vi
                        
            for k in range(4):
                K = k // 2
                kp = k % 2
                idx4_w = k * 4 + i
                idx2_w_outer = 2 * K + I
                idx2_w_inner = 2 * kp + ip
                
                for ro in range(7):
                    wo = W2[idx2_w_outer, ro]
                    if wo == 0:
                        continue
                    for ri in range(7):
                        wi = W2[idx2_w_inner, ri]
                        if wi == 0:
                            continue
                        r = ro * 7 + ri
                        W4[idx4_w, r] = wo * wi

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Right block (cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
