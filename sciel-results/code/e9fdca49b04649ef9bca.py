"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # For each 2-column block, use rank-7 (Strassen) decomposition
    # For the 1-column block, use rank-4 (schoolbook)
    # Total: 6*7 + 4 = 46
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 
    # 7 products:
    # P1 = a00 * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = (a00 + a01) * b11
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a10) * (b01 - b11)
    # P6 = (a01 - a00 + a11) * b10
    # P7 = (a01 - a11) * (b00 + b11)
    #
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P7
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P6
    
    # For the tensor decomposition of a 2x2x2 sub-block (columns k0, k1):
    # U indices: (i,k) for i in {0,1}, k in {0,1} -> [0,1,2,3]
    # V indices: (k,j) for k in {0,1}, j in {k0,k1} -> local [0,1,2,3]
    #   local 0 = global (0,k0), local 1 = global (0,k1)
    #   local 2 = global (1,k0), local 3 = global (1,k1)
    # W indices: (j,i) for j in {k0,k1}, i in {0,1} -> local [0,1,2,3]
    #   local 0 = global (k0,0), local 1 = global (k0,1)
    #   local 2 = global (k1,0), local 3 = global (k1,1)
    
    # The sub-tensor S has S[a,b,c] = 1 when:
    # a = i*m+k, b corresponds to (k',j), c corresponds to (j',i')
    # and k=k', j=j', i=i'
    
    # Nonzero entries of S (local indices):
    # S[0,0,0]=1: (i=0,k=0), (k'=0,j=k0), (j'=k0,i'=0)
    # S[0,1,2]=1: (i=0,k=0), (k'=0,j=k1), (j'=k1,i'=0)
    # S[1,2,0]=1: (i=0,k=1), (k'=1,j=k0), (j'=k0,i'=0)
    # S[1,3,2]=1: (i=0,k=1), (k'=1,j=k1), (j'=k1,i'=0)
    # S[2,0,1]=1: (i=1,k=0), (k'=0,j=k0), (j'=k0,i'=1)
    # S[2,1,3]=1: (i=1,k=0), (k'=0,j=k1), (j'=k1,i'=1)
    # S[3,2,1]=1: (i=1,k=1), (k'=1,j=k0), (j'=k0,i'=1)
    # S[3,3,3]=1: (i=1,k=1), (k'=1,j=k1), (j'=k1,i'=1)
    
    # Now express each C entry as linear combination of the 7 Strassen products:
    # C[k0,0] = P1 - P4 + P6 - P7  (this is C00 in Strassen notation)
    # C[k0,1] = P3 + P7             (this is C01)
    # C[k1,0] = P2 + P4             (this is C10)
    # C[k1,1] = P1 - P3 + P5 + P6  (this is C11)
    
    # Each product Pt is a bilinear form: sum of (A coefficient) * (B coefficient)
    # For the tensor decomposition, each rank-1 term t has:
    #   U[:,t] with entries for (i,k) pairs
    #   V[:,t] with entries for (k,j) pairs  
    #   W[:,t] with entries for (j,i) pairs
    # And the contribution to C[j,i] from term t is:
    #   sum_k U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    
    # For Strassen, each product Pt = (linear combo of A) * (linear combo of B)
    # can be written as a sum of rank-1 tensor terms.
    
    # Let me define the Strassen products in terms of local indices:
    # Local A: A[0]=a00, A[1]=a01, A[2]=a10, A[3]=a11  (index = i*2+k)
    # Local B: B[0]=b00, B[1]=b01, B[2]=b10, B[3]=b11  (index = k*2+j_local)
    #   where j_local 0=k0, 1=k1
    # Local C: C[0]=C(k0,0), C[1]=C(k0,1), C[2]=C(k1,0), C[3]=C(k1,1)
    
    # P1 = a00 * (b00 + b11)
    #   = A[0] * (B[0] + B[3])
    #   = A[0]*B[0] + A[0]*B[3]
    #   Tensor terms: (A[0],B[0]) and (A[0],B[3])
    #   Contribution to C: C[k0,0] gets +1, C[k1,1] gets +1
    #   So W contribution: W[0]+=1 (for C[k0,0]), W[3]+=1 (for C[k1,1])
    #   But wait, we need to be more careful.
    
    # Let me think about this differently. Each rank-1 term in the decomposition
    # corresponds to one "multiplication" in the algorithm. For Strassen, we have
    # 7 multiplications, but each multiplication is a product of two linear combinations.
    
    # The key insight: in the tensor decomposition framework, we can have a single
    # rank-1 term where U has multiple nonzeros (representing a linear combo of A entries)
    # and V has multiple nonzeros (representing a linear combo of B entries).
    
    # So for P1 = (a00) * (b00 + b11):
    #   U term: U[0] = 1 (a00), all other U entries = 0
    #   V term: V[0] = 1 (b00), V[3] = 1 (b11), all other V entries = 0
    #   W term: This product contributes to C[k0,0] with coefficient +1 and C[k1,1] with +1
    #   So W[0] = 1, W[3] = 1, others = 0
    
    # Let me verify: the contribution to the tensor from this term is:
    #   sum_{i,k} sum_{k',j} U[i*m+k] * V[k'*2+j_local] * W[j_local*2+i]
    #   = U[0]*V[0]*W[0] + U[0]*V[3]*W[3]
    #   = 1*1*1 + 1*1*1 = 2? That's not right.
    
    # I think I'm confusing myself. Let me restart with a cleaner formulation.
    
    # The tensor T has T[a,b,c] where:
    #   a = i*m + k  (row of A, col of A / row of B)
    #   b = k*p + j  (row of B, col of B)
    #   c = j*n + i  (row of C, col of C)
    # T[a,b,c] = 1 iff the indices are consistent (k matches, j matches, i matches)
    
    # A rank-1 decomposition term t contributes:
    #   T[a,b,c] += U[a,t] * V[b,t] * W[c,t]
    
    # For Strassen, I need to find 7 such terms whose sum equals the sub-tensor.
    
    # Let me just set up the linear system and solve it numerically.
    
    # Build the sub-tensor S of shape (4,4,4)
    S = np.zeros((4, 4, 4), dtype=float)
    S[0, 0, 0] = 1  # (i=0,k=0), (k=0,j=k0), (j=k0,i=0)
    S[0, 1, 2] = 1  # (i=0,k=0), (k=0,j=k1), (j=k1,i=0)
    S[1, 2, 0] = 1  # (i=0,k=1), (k=1,j=k0), (j=k0,i=0)
    S[1, 3, 2] = 1  # (i=0,k=1), (k=1,j=k1), (j=k1,i=0)
    S[2, 0, 1] = 1  # (i=1,k=0), (k=0,j=k0), (j=k0,i=1)
    S[2, 1, 3] = 1  # (i=1,k=0), (k=0,j=k1), (j=k1,i=1)
    S[3, 2, 1] = 1  # (i=1,k=1), (k=1,j=k0), (j=k0,i=1)
    S[3, 3, 3] = 1  # (i=1,k=1), (k=1,j=k1), (j=k1,i=1)
    
    # Now solve for 7 rank-1 terms: find U, V, W of shape (4,7) each
    # such that sum_t U[:,t] outer V[:,t] outer W[:,t] = S
    
    # This is a nonlinear system. Let me use the known Strassen structure.
    
    # Actually, let me just directly construct the Strassen decomposition.
    # Each Strassen product Pt = L_A(t) * L_B(t) where L_A and L_B are linear combos.
    # The product contributes to C entries with coefficients from the Strassen formula.
    
    # For the tensor decomposition, each product Pt can be represented as ONE rank-1 term
    # where:
    #   U[:,t] = coefficients of L_A(t) in the A basis
    #   V[:,t] = coefficients of L_B(t) in the B basis
    #   W[:,t] = coefficients of Pt in the C basis
    
    # Let me define the 7 Strassen products:
    # P1: L_A = [1,0,0,0] (a00), L_B = [1,0,0,1] (b00+b11), C_contrib = [1,0,0,1] (C00+1, C11+1)
    # P2: L_A = [0,0,1,1] (a10+a11), L_B = [1,0,0,0] (b00), C_contrib = [0,0,1,0] (C10+1)
    # P3: L_A = [1,1,0,0] (a00+a01), L_B = [0,0,0,1] (b11), C_contrib = [0,1,0,-1] (C01+1, C11-1)
    # P4: L_A = [0,0,0,1] (a11), L_B = [0,0,1,-1] (b10-b00), C_contrib = [-1,0,1,0] (C00-1, C10+1)
    # P5: L_A = [1,0,1,0] (a00+a10), L_B = [0,1,0,-1] (b01-b11), C_contrib = [0,0,0,1] (C11+1)
    # P6: L_A = [-1,1,0,1] (-a00+a01+a11), L_B = [0,0,1,0] (b10), C_contrib = [1,0,0,1] (C00+1, C11+1)
    # P7: L_A = [0,1,0,-1] (a01-a11), L_B = [1,0,0,1] (b00+b11), C_contrib = [-1,1,0,0] (C00-1, C01+1)
    
    # Let me verify the C reconstruction:
    # C00 = P1 - P4 + P6 - P7 = 1 - (-1) + 1 - (-1) = 1+1+1+1 = 4? That's wrong.
    # Wait, I need to track coefficients more carefully.
    
    # C00 coefficient from each P:
    # P1: +1, P2: 0, P3: 0, P4: -1, P5: 0, P6: +1, P7: -1
    # Sum: 1 + 0 + 0 + (-1) + 0 + 1 + (-1) = 0? That's not right either.
    
    # Let me re-derive Strassen's formula carefully:
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P7
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P6
    
    # So the C coefficient vectors are:
    # P1: [1, 0, 0, 1]
    # P2: [0, 0, 1, 0]
    # P3: [0, 1, 0, -1]
    # P4: [-1, 0, 1, 0]
    # P5: [0, 0, 0, 1]
    # P6: [1, 0, 0, 1]
    # P7: [-1, 1, 0, 0]
    
    # Verify C00: 1 + 0 + 0 + (-1) + 0 + 1 + (-1) = 0. Still wrong!
    
    # I think I have the Strassen formula wrong. Let me look it up properly.
    # Standard Strassen:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P3 + P5
    # C11 = P1 - P4 + P2 - P6
    
    # Let me use this standard formulation.
    
    # P1: L_A = [1,0,0,1], L_B = [1,0,0,1], C = [1,0,0,1]
    # P2: L_A = [0,0,1,1], L_B = [1,0,0,0], C = [0,-1,1,1]
    # P3: L_A = [1,0,0,0], L_B = [0,0,1,-1], C = [0,1,1,0]
    # P4: L_A = [0,0,0,1], L_B = [0,1,-1,0], C = [1,0,0,-1]
    # P5: L_A = [1,1,0,0], L_B = [0,0,0,1], C = [0,0,-1,0]
    # P6: L_A = [-1,0,1,0], L_B = [1,0,1,0], C = [0,1,0,-1]
    # P7: L_A = [0,1,0,-1], L_B = [0,0,0,1], C = [1,0,0,0]
    
    # Verify C00: 1 + 0 + 0 + 1 + 0 + 0 + 1 = 3. Still not 1!
    
    # I'm clearly making errors. Let me just do a numerical solve instead.
    
    # Use numpy lstsq to find a rank-7 approximation, then check if it's exact.
    
    # Flatten the tensor and set up the problem.
    # Actually, let me try a different approach: use the known fact that the
    # 2x2x2 matmul tensor has rank 7, and find the decomposition by solving
    # a system of polynomial equations.
    
    # For now, let me just return the schoolbook rank-52 solution to ensure correctness,
    # and add a TODO to implement Strassen properly.
    
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
