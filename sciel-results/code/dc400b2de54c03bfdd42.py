"""Optimized matrix multiplication tensor decomposition for 4x4x8."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 128
    
    # Construct the schoolbook decomposition (always valid, rank 128)
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Use the block structure.
    # 4x4x8 multiplication can be viewed as two independent 4x4x4 multiplications
    # if we split the output matrix C (4x8) into two 4x4 blocks C_left and C_right.
    # C_left = A * B_left, C_right = A * B_right.
    # This is equivalent to multiplying a 4x4 matrix A by a 4x8 matrix B.
    # The tensor is T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,8).
    # This tensor is the Kronecker product of the 4x4x4 multiplication tensor with a 1x1x2 tensor?
    # No, the 4x4x4 tensor is M4[i,j,k] = 1.
    # The 4x4x8 tensor is M8[i,j,k] = 1.
    # M8 is NOT simply M4 ⊗ M2.
    # However, if we have a rank-R decomposition for 4x4x4, can we use it for 4x4x8?
    # Let's look at the indices.
    # U has shape (16, R), V has shape (32, R), W has shape (32, R).
    # For 4x4x4: U4 (16, R4), V4 (16, R4), W4 (16, R4).
    # If we duplicate the structure:
    # U = [U4, U4] (16, 2*R4) -- No, U must be (16, R_total).
    # Let's try:
    # U_total (16, 2*R4) = [U4, U4] horizontally?
    # V_total (32, 2*R4) = [V4, V4] vertically? i.e., block_diag(V4, V4) is (32, 2*R4) if we interleave?
    # Let's check the tensor equation:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # If we set r = r1 + r2 where r1 in [0, R4) and r2 in [0, R4):
    # Term 1 (r in first block): U4[i,j,r1] * V4[j,k1,r1] * W4[k1,i,r1] where k1 = k mod 4?
    # This requires V and W to be structured such that they select the right half.
    
    # Let's construct it explicitly.
    # Suppose we have a valid decomposition for 4x4x4 with rank R4:
    # sum_{r=0}^{R4-1} U4[i*m+j, r] V4[j*p4+k, r] W4[k*n+i, r] = 1 for all i,j,k in 0..3.
    # Here p4=4, n=4, m=4.
    
    # Now for 4x4x8, we want:
    # sum_{r=0}^{R_total-1} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j in 0..3, k in 0..7.
    
    # Let R_total = 2 * R4.
    # Let r = 2*s + t, where s in [0, R4), t in {0, 1}.
    # We want to define U, V, W in terms of U4, V4, W4.
    # Let U[i*m+j, 2*s+t] = U4[i*m+j, s]. (U is duplicated for t=0,1)
    # Let V[j*p+k, 2*s+t]:
    #   If t=0 (corresponds to first 4 columns of B, i.e., k in 0..3):
    #     We want this to contribute to the first half of the output.
    #     V[j*8+k, 2*s] = V4[j*4+k, s] if k < 4, else 0.
    #   If t=1 (corresponds to second 4 columns of B, i.e., k in 4..7):
    #     V[j*8+k, 2*s+1] = V4[j*4+(k-4), s] if k >= 4, else 0.
    # Let W[k*n+i, 2*s+t]:
    #   The output index is k (0..7) and i (0..3).
    #   If t=0, we want to write to C[i, k] for k in 0..3.
    #     W[k*4+i, 2*s] = W4[k*4+i, s] if k < 4, else 0.
    #   If t=1, we want to write to C[i, k] for k in 4..7.
    #     W[k*4+i, 2*s+1] = W4[(k-4)*4+i, s] if k >= 4, else 0.
    
    # Let's verify the sum:
    # For a fixed i, j, k:
    # Sum over r = sum_{s} (U[i,j,2s] V[j,k,2s] W[k,i,2s] + U[i,j,2s+1] V[j,k,2s+1] W[k,i,2s+1])
    # = sum_{s} (U4[i,j,s] V[j,k,2s] W[k,i,2s] + U4[i,j,s] V[j,k,2s+1] W[k,i,2s+1])
    
    # Case 1: k < 4.
    #   V[j,k,2s] = V4[j,k,s], W[k,i,2s] = W4[k,i,s].
    #   V[j,k,2s+1] = 0, W[k,i,2s+1] = 0.
    #   Sum = sum_s U4[i,j,s] V4[j,k,s] W4[k,i,s] = 1. (Correct)
    
    # Case 2: k >= 4. Let k' = k - 4.
    #   V[j,k,2s] = 0, W[k,i,2s] = 0.
    #   V[j,k,2s+1] = V4[j,k',s], W[k,i,2s+1] = W4[k',i,s].
    #   Sum = sum_s U4[i,j,s] V4[j,k',s] W4[k',i,s] = 1. (Correct)
    
    # So if we have a rank-R4 decomposition for 4x4x4, we can construct a rank-2*R4 decomposition for 4x4x8.
    
    # Now, what is the best known rank for 4x4x4?
    # Schoolbook: 64.
    # Strassen-like: 
    #   2x2x2 rank is 7.
    #   4x4x4 can be built from 2x2x2 blocks.
    #   Standard block construction:
    #   A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    #   C = A B.
    #   Using Strassen for 2x2 blocks:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A10)B11
    #   M6 = (A11-A10)(B00+B10)
    #   M7 = (A01-A11)(B10+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #   
    #   Each Mx is a 2x2x2 multiplication tensor contraction.
    #   The rank of the 4x4x4 tensor is at most 7 * (rank of 2x2x2 decomposition used for each Mx)?
    #   No, the Mx are linear combinations of the 2x2 matrices.
    #   Actually, the rank of the 4x4x4 tensor is bounded by 7 * 4 = 28? No.
    #   The standard Strassen algorithm for 4x4x4 uses 7 multiplications of 2x2 matrices.
    #   Each 2x2 multiplication can be done with rank 4 (schoolbook) or 7 (Strassen).
    #   If we use schoolbook for the inner 2x2, the total rank is 7 * 4 = 28.
    #   Wait, is 28 the rank?
    #   Let's check the tensor structure.
    #   The 4x4x4 tensor is the Kronecker product of the 2x2x2 tensor with itself?
    #   T4 = T2 ⊗ T2.
    #   Rank(T ⊗ S) <= Rank(T) * Rank(S).
    #   Rank(T2) = 7.
    #   So Rank(T4) <= 7 * 7 = 49.
    #   Can we do better?
    #   The block construction above (7 multiplications of 2x2) corresponds to:
    #   Each "multiplication" is a 2x2x2 tensor.
    #   If we decompose each 2x2x2 tensor with rank 7, we get 7 * 7 = 49 terms.
    #   If we decompose each 2x2x2 tensor with rank 4 (schoolbook), we get 7 * 4 = 28 terms.
    #   BUT, the 7 multiplications M1..M7 are not independent 2x2x2 tensors in the standard sense?
    #   M1 = (A00+A11)(B00+B11). This is a bilinear form.
    #   The tensor for M1 is (e_A00 + e_A11) ⊗ (e_B00 + e_B11) ⊗ e_C00? No.
    #   M1 contributes to C00 and C11.
    #   M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11.
    #   This is a sum of 4 rank-1 tensors for the 2x2 block multiplication.
    #   So the rank of the 4x4x4 tensor using this specific block decomposition is 7 * 4 = 28.
    #   Is 28 a valid upper bound?
    #   Let's verify.
    #   We need to express the 4x4x4 multiplication tensor as a sum of 28 rank-1 tensors.
    #   The 28 terms come from expanding the 7 Strassen multiplications into 4 schoolbook terms each.
    #   Total terms: 28.
    #   So Rank(4x4x4) <= 28.
    #   Then Rank(4x4x8) <= 2 * 28 = 56.
    
    # Let's try to construct the rank-28 decomposition for 4x4x4 explicitly.
    # Then double it for 4x4x8 to get rank 56.
    # This is significantly better than 128.
    
    # Construction of 4x4x4 rank-28 decomposition:
    # We define 28 triples (u, v, w) of vectors in R^16 (or C^16).
    # Actually, we can just build the matrices U4, V4, W4 of shape (16, 28).
    
    # Indices for 4x4 matrices:
    # A = [[a00, a01], [a10, a11]] where aij are 2x2 matrices.
    # Flat index for A: 
    #   A[0,0]=a00[0,0], A[0,1]=a00[0,1], A[0,2]=a00[1,0], A[0,3]=a00[1,1]
    #   A[1,0]=a01[0,0], ...
    #   Wait, standard row-major: A[i,j] with i in 0..3, j in 0..3.
    #   Block (0,0) is A[0:2, 0:2].
    #   Block (0,1) is A[0:2, 2:4].
    #   Block (1,0) is A[2:4, 0:2].
    #   Block (1,1) is A[2:4, 2:4].
    
    # Let's define the 7 Strassen multiplications in terms of flat indices.
    # M1 = (A00 + A11) * (B00 + B11)
    # A00: indices (0,0), (0,1), (1,0), (1,1) -> flat 0,1,4,5
    # A11: indices (2,2), (2,3), (3,2), (3,3) -> flat 10,11,14,15
    # B00: flat 0,1,4,5
    # B11: flat 10,11,14,15
    # C00: flat 0,1,4,5
    # C11: flat 10,11,14,15
    
    # M1 contributes to C00 and C11.
    # M1 = sum_{i,j} (A00[i,j] + A11[i,j]) (B00[i,j] + B11[i,j]) ... wait, matrix multiplication is sum_k A[i,k]B[k,j].
    # So M1 is a 2x2 matrix: M1 = (A00+A11)(B00+B11).
    # M1[i,j] = sum_k (A00[i,k]+A11[i,k]) (B00[k,j]+B11[k,j]).
    # This is a bilinear form in A and B, linear in C.
    # We can decompose M1 into 4 rank-1 tensors (schoolbook for 2x2x2).
    # M1 = sum_{k=0}^1 (A00[:,k]+A11[:,k]) (B00[k,:]+B11[k,:])
    # Term k=0: (A00[:,0]+A11[:,0]) (B00[0,:]+B11[0,:])
    # Term k=1: (A00[:,1]+A11[:,1]) (B00[1,:]+B11[1,:])
    # Each term is a rank-1 tensor?
    # T = u ⊗ v ⊗ w.
    # For term k=0:
    #   u has 1s at A00[:,0] and A11[:,0] positions.
    #   v has 1s at B00[0,:] and B11[0,:] positions.
    #   w has 1s at C00[:,0] and C11[:,0]? No.
    #   M1 contributes to C00 and C11 with coefficients.
    #   C00 = M1 + M4 - M5 + M7
    #   C11 = M1 - M2 + M3 + M6
    #   So M1 contributes +1 to C00 and +1 to C11.
    #   So w has 1s at C00 and C11 positions corresponding to the output indices.
    #   Specifically, for the term (A00[:,0]+A11[:,0]) (B00[0,:]+B11[0,:]):
    #   This computes (A00+A11)[i,0] * (B00+B11)[0,j] for i,j in 0..1.
    #   This contributes to C00[i,j] and C11[i,j].
    #   So u has 1s at A00[i,0] and A11[i,0] for i=0,1.
    #   v has 1s at B00[0,j] and B11[0,j] for j=0,1.
    #   w has 1s at C00[i,j] and C11[i,j] for i,j=0,1.
    
    # This is getting complicated to hardcode.
    # Let's use a simpler approach:
    # We know Rank(4x4x4) <= 49 (using Strassen for inner 2x2).
    # We know Rank(4x4x4) <= 28 (using schoolbook for inner 2x2 in Strassen blocks).
    # Let's just implement the rank-28 decomposition for 4x4x4 and double it.
    
    # Actually, there is an even simpler decomposition.
    # The 4x4x4 tensor is the Kronecker product of 2x2x2 tensor with itself.
    # T4 = T2 ⊗ T2.
    # If we have a rank-7 decomposition for T2:
    # T2 = sum_{r=0}^6 u_r ⊗ v_r ⊗ w_r.
    # Then T4 = sum_{r1=0}^6 sum_{r2=0}^6 (u_{r1} ⊗ u_{r2}) ⊗ (v_{r1} ⊗ v_{r2}) ⊗ (w_{r1} ⊗ w_{r2}).
    # This gives rank 49.
    # This is easier to construct if we have the rank-7 decomposition for 2x2x2.
    
    # Rank-7 decomposition for 2x2x2 (Strassen):
    # u_1 = [1, 0, 0, 0] (A00)
    # v_1 = [1, 0, 0, 0] (B00)
    # w_1 = [1, 0, 0, 0] (C00) ... wait, Strassen is different.
    
    # Let's use the standard Strassen decomposition for 2x2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # m1 = (a+d)(e+h) -> u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (contributes to i and l)
    # m2 = (c+d)e -> u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (contributes to k)
    # m3 = a(f-h) -> u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] (contributes to j)
    # m4 = d(g-e) -> u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,0,1] (contributes to l)
    # m5 = (a+c)h -> u=[1,0,1,0], v=[0,0,0,1], w=[0,0,1,0] (contributes to k)
    # m6 = (d-c)(e+g) -> u=[0,0,-1,1], v=[1,0,1,0], w=[1,0,0,0] (contributes to i)
    # m7 = (b-d)(g+h) -> u=[0,1,0,-1], v=[0,1,1,1], w=[1,0,0,1] (contributes to i and l)
    
    # Let's verify m1:
    # (a+d)(e+h) = ae + ah + de + dh.
    # C00 = i = ae + bg + de + fh? No, C00 = ae + bg.
    # Strassen:
    # i = m1 + m4 - m5 + m7
    # l = m1 - m2 + m3 + m6
    # j = m3 + m5
    # k = m2 + m4
    
    # Let's check m1 contribution to i and l.
    # m1 = (a+d)(e+h).
    # In i: +m1.
    # In l: +m1.
    # So w_1 has 1 at i and 1 at l.
    # u_1 has 1 at a and d.
    # v_1 has 1 at e and h.
    
    # This is correct.
    
    # So we can construct the rank-49 decomposition for 4x4x4 by taking the Kronecker product of the 7 Strassen vectors.
    # Then double it for 4x4x8 to get rank 98.
    # This is better than 128.
    
    # Let's implement this.
    
    # Strassen 2x2x2 decomposition vectors (flat index order: a,b,c,d for A; e,f,g,h for B; i,j,k,l for C)
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=i, 1=j, 2=k, 3=l
    
    s_u = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 0, 1, 0],  # m5
        [0, 0, -1, 1], # m6
        [0, 1, 0, -1]  # m7
    ]
    s_v = [
        [1, 0, 0, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1],  # m5
        [1, 0, 1, 0],  # m6
        [0, 1, 1, 1]   # m7
    ]
    s_w = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 0],  # m2
        [0, 1, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [0, 0, 1, 0],  # m5
        [1, 0, 0, 0],  # m6
        [1, 0, 0, 1]   # m7
    ]
    
    # Construct 4x4x4 rank-49 decomposition
    U4 = []
    V4 = []
    W4 = []
    
    for r1 in range(7):
        for r2 in range(7):
            u = np.kron(s_u[r1], s_u[r2])
            v = np.kron(s_v[r1], s_v[r2])
            w = np.kron(s_w[r1], s_w[r2])
            U4.append(u)
            V4.append(v)
            W4.append(w)
    
    U4 = np.array(U4).T  # Shape (16, 49)
    V4 = np.array(V4).T  # Shape (16, 49)
    W4 = np.array(W4).T  # Shape (16, 49)
    
    # Verify 4x4x4 decomposition
    # T4 = sum_r U4[:,r] ⊗ V4[:,r] ⊗ W4[:,r]
    # We can check a few entries.
    
    # Now construct 4x4x8 rank-98 decomposition
    # U_total (16, 98)
    # V_total (32, 98)
    # W_total (32, 98)
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    for s in range(49):
        # t=0
        U_total[:, 2*s] = U4[:, s]
        V_total[0:16, 2*s] = V4[:, s]
        W_total[0:16, 2*s] = W4[:, s]
        
        # t=1
        U_total[:, 2*s+1] = U4[:, s]
        V_total[16:32, 2*s+1] = V4[:, s]
        W_total[16:32, 2*s+1] = W4[:, s]
    
    # Convert to lists of Fractions or integers
    U_out = [[int(x) for x in row] for row in U_total]
    V_out = [[int(x) for x in row] for row in V_total]
    W_out = [[int(x) for x in row] for row in W_total]
    
    # Quick verification
    # Check a few random entries
    import random
    random.seed(42)
    valid = True
    for _ in range(10):
        i = random.randint(0, 3)
        j = random.randint(0, 3)
        k = random.randint(0, 7)
        val = 0
        for r in range(98):
            val += U_out[i*4+j][r] * V_out[j*8+k][r] * W_out[k*4+i][r]
        if val != 1:
            valid = False
            break
    
    if valid:
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
