"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's 7-rank decomposition for 2x2 matrix multiplication
    # For 2x2: A*B = C where A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # 
    # The 7 intermediate products are:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    #
    # Then:
    # i = m1 + m4 - m5 + m7
    # j = m3 + m5
    # k = m2 + m4
    # l = m1 - m2 + m3 + m6
    #
    # U maps (a,b,c,d) to the 7 intermediates
    # V maps (e,f,g,h) to the 7 intermediates  
    # W maps the 7 intermediates to (i,j,k,l)
    #
    # U (4x7): row for each input element [a,b,c,d]
    #   a: [1, 0, 1, 0, 1, -1, 0]
    #   b: [0, 0, 0, 0, 1, 0, 1]
    #   c: [0, 1, 0, 0, 0, 1, 0]
    #   d: [1, 1, 0, 1, 0, 0, -1]
    #
    # V (4x7): row for each input element [e,f,g,h]
    #   e: [1, 1, 0, -1, 0, 1, 0]
    #   f: [0, 0, 1, 0, 0, 1, 0]
    #   g: [0, 0, 0, 1, 0, 0, 1]
    #   h: [1, 0, -1, 0, 1, 0, 1]
    #
    # W (4x7): row for each output element [i,j,k,l]
    #   i: [1, 0, 0, 1, -1, 0, 1]
    #   j: [0, 0, 1, 0, 1, 0, 0]
    #   k: [0, 1, 0, 1, 0, 0, 0]
    #   l: [1, -1, 1, 0, 0, 1, 0]
    
    U2 = [[1, 0, 1, 0, 1, -1, 0],
          [0, 0, 0, 0, 1, 0, 1],
          [0, 1, 0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0, 0, -1]]
    
    V2 = [[1, 1, 0, -1, 0, 1, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 0, 1, 0, 0, 1],
          [1, 0, -1, 0, 1, 0, 1]]
    
    W2 = [[1, 0, 0, 1, -1, 0, 1],
          [0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [1, -1, 1, 0, 0, 1, 0]]
    
    # Verify 2x2 decomposition works
    # For 2x2: U shape (n*m, rank) = (4, 7), V shape (m*p, rank) = (4, 7), W shape (p*n, rank) = (4, 7)
    # U[i*m+j, c] where i,j in 0..1
    # V[j*p+k, c] where j,k in 0..1
    # W[k*n+i, c] where k,i in 0..1
    
    U2_arr = np.array(U2, dtype=float)
    V2_arr = np.array(V2, dtype=float)
    W2_arr = np.array(W2, dtype=float)
    
    # Test 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # A_flat = [a,b,c,d] (row-major: a=idx0, b=idx1, c=idx2, d=idx3)
    # B_flat = [e,f,g,h] (row-major: e=idx0, f=idx1, g=idx2, h=idx3)
    # C_flat should be [i,j,k,l] where C = [[i,j],[k,l]]
    # W[k*n+i, r] means output index = k*2+i
    # C[0,0]=i -> W[0*2+0]=W[0]
    # C[0,1]=j -> W[0*2+1]=W[1]
    # C[1,0]=k -> W[1*2+0]=W[2]
    # C[1,1]=l -> W[1*2+1]=W[3]
    
    # Check: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should give C[i,k]
    # Let's verify with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, (2, 2)).astype(float)
    B_test = np.random.randint(-5, 6, (2, 2)).astype(float)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()  # [a, b, c, d]
    B_flat = B_test.flatten()  # [e, f, g, h]
    
    C_reconstructed = np.zeros((2, 2))
    for i in range(2):
        for k in range(2):
            total = 0
            for r in range(7):
                for j in range(2):
                    total += U2_arr[i*2+j, r] * V2_arr[j*2+k, r] * W2_arr[k*2+i, r]
            C_reconstructed[i, k] = total
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Try alternative indexing
        pass
    
    # The Kronecker product approach:
    # For 4x4, we partition into 2x2 blocks
    # A (4x4) has blocks: A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # In flat row-major: A_flat[i*4+j]
    # Block (bi, bj) where bi, bj in {0,1}: A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # Element within block at position (pi, pj): A_flat[(bi*2+pi)*4 + (bj*2+pj)]
    
    # For the Kronecker product decomposition:
    # U4[i*4+j, r1*7+r2] = U2[bi*2+pi, r1] * U2[bj*2+pj, r2]
    # where i = bi*2+pi, j = bj*2+pj, and r1,r2 in 0..6
    
    # Similarly for V and W
    
    # Let's build the 4x4 decomposition
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    # U4[i*4+j, r1*7+r2] = U2[bi*2+pi, r1] * U2[bj*2+pj, r2]
    # where i = bi*2+pi, j = bj*2+pj
    for i in range(4):
        for j in range(4):
            bi = i // 2
            pi = i % 2
            bj = j // 2
            pj = j % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U4[i*4+j, r] = U2_arr[bi*2+pi, r1] * U2_arr[bj*2+pj, r2]
    
    # V4[j*4+k, r1*7+r2] = V2[bj*2+pj, r1] * V2[bk*2+pk, r2]
    # where j = bj*2+pj, k = bk*2+pk
    for j in range(4):
        for k in range(4):
            bj = j // 2
            pj = j % 2
            bk = k // 2
            pk = k % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V4[j*4+k, r] = V2_arr[bj*2+pj, r1] * V2_arr[bk*2+pk, r2]
    
    # W4[k*4+i, r1*7+r2] = W2[bk*2+pk, r1] * W2[bi*2+pi, r2]
    # where k = bk*2+pk, i = bi*2+pi
    for k in range(4):
        for i in range(4):
            bk = k // 2
            pk = k % 2
            bi = i // 2
            pi = i % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W4[k*4+i, r] = W2_arr[bk*2+pk, r1] * W2_arr[bi*2+pi, r2]
    
    # Verify the 4x4 decomposition
    np.random.seed(123)
    A_test4 = np.random.randint(-5, 6, (4, 4)).astype(float)
    B_test4 = np.random.randint(-5, 6, (4, 4)).astype(float)
    C_expected4 = A_test4 @ B_test4
    
    A_flat4 = A_test4.flatten()
    B_flat4 = B_test4.flatten()
    
    C_reconstructed4 = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            total = 0
            for r in range(49):
                for j in range(4):
                    total += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
            C_reconstructed4[i, k] = total
    
    if not np.allclose(C_reconstructed4, C_expected4, atol=1e-10):
        # Something is wrong with the indexing. Let me debug.
        # The issue might be in how I'm mapping the Kronecker product indices.
        # Let me try a different approach.
        
        # Actually, let me reconsider. The Kronecker product of two bilinear forms:
        # If T1(a,b) = sum_r U1[a,r]*V1[b,r]*W1[out,r] and T2 is similar,
        # then T1 ⊗ T2 acts on (a1⊗a2, b1⊗b2) -> out1⊗out2
        # with rank = rank1 * rank2
        #
        # For matrix multiplication, the tensor is T[i,j,k] = delta_{i,k} for A[i,j]*B[j,k]
        # The Kronecker product T ⊗ T has T[i1,j1,k1]*T[i2,j2,k2]
        # which corresponds to 4x4 multiplication if we index properly.
        
        # Let me try: the 4x4 multiplication tensor is the Kronecker product of two 2x2 tensors
        # T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2] where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
        #
        # For a decomposition of T2: T2[i,j,k] = sum_r U2[i,j,r]*V2[j,k,r]*W2[k,i,r]
        # Then T4[i,j,k] = sum_{r1,r2} U2[i1,j1,r1]*U2[i2,j2,r2] * V2[j1,k1,r1]*V2[j2,k2,r2] * W2[k1,i1,r1]*W2[k2,i2,r2]
        #
        # So U4[i,j,(r1,r2)] = U2[i1,j1,r1]*U2[i2,j2,r2]
        #    V4[j,k,(r1,r2)] = V2[j1,k1,r1]*V2[j2,k2,r2]
        #    W4[k,i,(r1,r2)] = W2[k1,i1,r1]*W2[k2,i2,r2]
        #
        # This is what I had. Let me check the 2x2 verification first.
        
        # Debug 2x2
        print("Debug 2x2:")
        print("A:", A_test)
        print("B:", B_test)
        print("Expected C:", C_expected)
        print("Reconstructed C:", C_reconstructed)
        
        # Maybe my Strassen matrices are wrong. Let me verify manually.
        # m1 = (a+d)*(e+h): U2 row for a has 1 at pos 0, row for d has 1 at pos 0. 
        # But U2 is (4x7) where rows are [a,b,c,d]. 
        # m1 needs (a+d) from A side and (e+h) from B side.
        # U2[0,:] (for a) should have 1 where m1 is, U2[3,:] (for d) should have 1 where m1 is.
        # V2[0,:] (for e) should have 1 where m1 is, V2[3,:] (for h) should have 1 where m1 is.
        # 
        # m1 is at position 0 (r=0).
        # U2[0,0]=1 (a), U2[3,0]=1 (d) -- OK
        # V2[0,0]=1 (e), V2[3,0]=1 (h) -- OK
        # 
        # m2 = (c+d)*e: 
        # U2[2,1]=1 (c), U2[3,1]=1 (d) -- OK
        # V2[0,1]=1 (e) -- OK
        # 
        # m3 = a*(f-h):
        # U2[0,2]=1 (a) -- OK
        # V2[1,2]=1 (f), V2[3,2]=-1 (h) -- OK
        # 
        # m4 = d*(g-e):
        # U2[3,3]=1 (d) -- OK
        # V2[2,3]=1 (g), V2[0,3]=-1 (e) -- OK
        # 
        # m5 = (a+b)*h:
        # U2[0,4]=1 (a), U2[1,4]=1 (b) -- OK
        # V2[3,4]=1 (h) -- OK
        # 
        # m6 = (c-a)*(e+f):
        # U2[2,5]=1 (c), U2[0,5]=-1 (a) -- OK
        # V2[0,5]=1 (e), V2[1,5]=1 (f) -- OK
        # 
        # m7 = (b-d)*(g+h):
        # U2[1,6]=1 (b), U2[3,6]=-1 (d) -- OK
        # V2[2,6]=1 (g), V2[3,6]=1 (h) -- OK
        # 
        # W: 
        # i = m1 + m4 - m5 + m7: W[0,:] = [1,0,0,1,-1,0,1] -- OK
        # j = m3 + m5: W[1,:] = [0,0,1,0,1,0,0] -- OK
        # k = m2 + m4: W[2,:] = [0,1,0,1,0,0,0] -- OK
        # l = m1 - m2 + m3 + m6: W[3,:] = [1,-1,1,0,0,1,0] -- OK
        #
        # So the matrices look correct. The issue must be in the 4x4 construction.
        
        # Let me check: maybe the Kronecker product needs different index mapping.
        # Actually, I think the issue is that for the Kronecker product to work as 4x4 matmul,
        # the block structure needs to be consistent.
        
        # Let me try a completely different approach: just use the Kronecker product 
        # of the full 2x2 multiplication tensors directly.
        
        # The 2x2 multiplication tensor T2[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # Actually T2 is defined by: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T2[i,j,k] which is 1 for all valid (i,j,k) in the bilinear form.
        #
        # A rank-r decomposition means: T2[i,j,k] = sum_{r=1}^r u_r[i,j] * v_r[j,k] * w_r[k,i]
        #
        # For the Kronecker product: T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2]
        # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
        #
        # But wait, is T4 really the Kronecker product? Let's think...
        # 4x4 matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # If we write i = i1*2+i2, j = j1*2+j2, k = k1*2+k2, then:
        # C[i1,i2, k1,k2] = sum_{j1,j2} A[i1,i2, j1,j2] * B[j1,j2, k1,k2]
        # = sum_{j1,j2} A[i1,i2,j1,j2] * B[j1,j2,k1,k2]
        #
        # This is NOT simply the Kronecker product of two 2x2 matmuls because the sum 
        # is over both j1 and j2 simultaneously, not separately.
        #
        # The Kronecker product of two 2x2 matmul tensors would give:
        # (T2 ⊗ T2)[i1,i2,j1,j2,k1,k2] = T2[i1,j1,k1] * T2[i2,j2,k2]
        # which corresponds to: C[i1,i2,k1,k2] = sum_{j1,j2} A[i1,j1]*A[i2,j2] * B[j1,k1]*B[j2,k2]
        # That's not the same as 4x4 matmul!
        #
        # So the Kronecker product approach doesn't directly work for 4x4 matmul.
        
        # I need to use a different approach. Let me use the fact that 4x4 matmul 
        # can be done by treating it as 2x2 block matmul where each block is 2x2.
        # C = [[C11, C12], [C21, C22]] where Cij = sum_k Aik * Bkj (block products)
        # Each block product Aik * Bkj is a 2x2 matmul, which has rank 7.
        # There are 4 blocks in A, 4 in B, and for each output block Cij we sum over 2 values of k.
        # Total rank = 4 * 2 * 7 = 56? No, that's not right either.
        #
        # Actually, for block matmul: C = A*B where A,B are 2x2 blocks of 2x2 matrices.
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        #
        # Each block product has rank 7, and we have 8 block products total.
        # But we can share the intermediate computations.
        #
        # This is getting complex. Let me just use the schoolbook decomposition 
        # for now and see if I can improve it.
        
        # Actually, let me reconsider. The known result is that the Kronecker product 
        # of Strassen's algorithm does give a valid decomposition for 4x4 matmul with rank 49.
        # The key is that we're not doing block matmul, but rather using the algebraic 
        # structure of the tensor.
        
        # Let me look at this differently. The 4x4 multiplication tensor can be 
        # decomposed using the "subcube" technique. But for now, let me just verify 
        # if my Kronecker product construction is correct by checking the tensor 
        # directly.
        
        # Build the 4x4 multiplication tensor T4[i,j,k] = 1 (it's always 1 for valid indices)
        # Actually no, the tensor is defined by the bilinear form. Let me just verify 
        # that my decomposition gives the correct result.
        
        # Let me try with simple test matrices.
        A_test4 = np.eye(4)
        B_test4 = np.eye(4)
        C_expected4 = A_test4 @ B_test4
        
        C_reconstructed4 = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                total = 0
                for r in range(49):
                    for j in range(4):
                        total += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                C_reconstructed4[i, k] = total
        
        if not np.allclose(C_reconstructed4, C_expected4, atol=1e-10):
            # Still wrong. Let me try a completely different construction.
            # 
            # Maybe I should just use the schoolbook decomposition and return that.
            # It's rank 64, which gives score 64/64 = 1.0.
            # 
            # But the Kronecker product should give 49. Let me think more carefully.
            #
            # Actually, I realize the issue. The Kronecker product of two bilinear forms 
            # (T1, T2) gives a bilinear form on the tensor product spaces. For matrix 
            # multiplication, this corresponds to multiplying Kronecker products of matrices:
            # (A1 ⊗ A2) * (B1 ⊗ B2) = (A1*B1) ⊗ (A2*B2)
            #
            # So the Kronecker product of two 2x2 matmul tensors gives the tensor for 
            # multiplying 4x4 matrices that are Kronecker products of 2x2 matrices.
            # This is NOT the full 4x4 multiplication tensor!
            #
            # So the rank-49 decomposition only works for a subset of 4x4 matrices 
            # (those that are Kronecker products). It does NOT work for general 4x4 matrices.
            #
            # This means I cannot use the simple Kronecker product approach.
            #
            # I need to find a proper rank-49 or lower decomposition for 4x4 matmul.
            # The best known upper bound is 49, achieved by more sophisticated methods.
            #
            # For now, let me just return the schoolbook decomposition (rank 64).
            
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
    
    # Convert to lists
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
