import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We want to decompose the 2x3 by 3x4 matrix multiplication tensor.
    # Strategy: Use Strassen (rank 7) for the 2x2 by 2x2 part (columns 0-1 of A, rows 0-1 of B),
    # and rank-1 terms for the remaining 2x1 by 1x4 part (column 2 of A, row 2 of B).
    # Total rank = 7 + 4 = 11? Wait, the 2x1 by 1x4 part is A[:,2] (2x1) times B[2,:] (1x4),
    # which gives a 2x4 matrix. Each of the 2*4=8 entries needs a rank-1 term? No.
    # Actually, the bilinear map for A[:,2] * B[2,:] is:
    # C[i,k] += A[i,2] * B[2,k]
    # This is a rank-8 tensor (each (i,k) pair needs its own term).
    # So total rank = 7 + 8 = 15.
    
    # Let's build it carefully.
    # A is 2x3, B is 3x4.
    # C = A @ B
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # The first two terms form a 2x2 by 2x2 multiplication: A[:,0:2] @ B[0:2,:]
    # We can use Strassen for this part.
    # Strassen for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    # Strassen computes:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # In our case:
    # A[:,0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B[0:2,:] = [[B[0,0], B[0,1], B[0,2], B[0,3]],
    #             [B[1,0], B[1,1], B[1,2], B[1,3]]]
    # Wait, Strassen is for 2x2 by 2x2. But B[0:2,:] is 2x4.
    # We need to handle this. We can split B[0:2,:] into two 2x2 blocks:
    # B[0:2,0:2] and B[0:2,2:4].
    # So C[:,0:2] = A[:,0:2] @ B[0:2,0:2] (2x2 by 2x2)
    # And C[:,2:4] = A[:,0:2] @ B[0:2,2:4] (2x2 by 2x2)
    # Each 2x2 by 2x2 multiplication uses rank 7.
    # So that's 7 + 7 = 14 terms for the first two columns of A.
    # Plus 8 terms for A[:,2] * B[2,:] = 22 terms total.
    # That's worse than the seed's 24? No, 22 < 24.
    # But can we do better?
    
    # Alternative: Use rank 7 for the full 2x2 by 2x4 multiplication?
    # The tensor for 2x2 by 2x4 has rank at most 7*2=14 (by splitting into two 2x2).
    # But maybe there's a rank 7 decomposition for 2x2 by 2x4? Unlikely.
    
    # Let's just use the straightforward approach:
    # - 7 terms for A[:,0:2] @ B[0:2,0:2] (2x2 by 2x2)
    # - 7 terms for A[:,0:2] @ B[0:2,2:4] (2x2 by 2x2)
    # - 8 terms for A[:,2] * B[2,:] (2x1 by 1x4)
    # Total = 22.
    
    # But wait, the seed uses rank 24. 22 < 24, so this is an improvement.
    # Can we do even better?
    
    # Let's think about the 2x2 by 2x4 multiplication.
    # A is 2x2, B is 2x4. C = A @ B is 2x4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a bilinear map from (2x2) x (2x4) to (2x4).
    # The tensor has dimension 4 * 8 * 8 = 256.
    # The rank is at most 14 (by splitting B into two 2x2 blocks).
    # But maybe it can be done in rank 7? Probably not, but let's try to construct it.
    
    # Actually, let's just go with rank 22 for now. That's already better than 24.
    
    # Let me reconsider. The problem says "Find a rank-minimal exact tensor decomposition".
    # The current seed uses rank 24. If I can find rank 22, that's an improvement.
    # But maybe rank 15 is possible?
    
    # Let me try a different approach: use rank 7 for the 2x2 by 2x2 part,
    # and then handle the 2x1 by 1x4 part more cleverly.
    
    # Actually, the 2x1 by 1x4 part is just A[:,2] * B[2,:], which is an outer product.
    # The outer product of a 2x1 and 1x4 matrix is a 2x4 matrix.
    # The tensor for this outer product has rank 8 (one term for each (i,k) pair).
    # So total = 7 + 7 + 8 = 22.
    
    # Let me just implement rank 22 and see if it's valid.
    
    # Actually, let me reconsider the problem. Maybe I can use a smarter decomposition.
    
    # For now, let's implement rank 22.
    
    # U is (n*m, rank) = (6, 22)
    # V is (m*p, rank) = (12, 22)
    # W is (p*n, rank) = (8, 22)
    
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper function to add a Strassen term for 2x2 by 2x2
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {k0, k0+1}
    # C indices: i in {0,1}, k in {k0, k0+1}
    def add_strassen(A_j0, A_j1, B_j0, B_j1, k0, k1, base_idx):
        """
        Add 7 Strassen terms for 2x2 by 2x2 multiplication.
        A is 2x2 with columns A_j0, A_j1 (so A[0,0]=A[:,A_j0], A[0,1]=A[:,A_j1])
        B is 2x2 with rows B_j0, B_j1 (so B[0,0]=B[B_j0,:], B[1,0]=B[B_j1,:])
        Actually, let me be more careful.
        
        A[i, j] where i in {0,1}, j in {A_j0, A_j1}
        B[j, k] where j in {B_j0, B_j1}, k in {k0, k1}
        C[i, k] where i in {0,1}, k in {k0, k1}
        
        For Strassen, we need:
        a = A[0, A_j0], b = A[0, A_j1], c = A[1, A_j0], d = A[1, A_j1]
        e = B[B_j0, k0], f = B[B_j0, k1], g = B[B_j1, k0], h = B[B_j1, k1]
        
        The 7 products are:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-e)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        And:
        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        
        For each term r (0 to 6), we need:
        U[i*m+j, r] = coefficient of A[i,j] in the r-th product
        V[j*p+k, r] = coefficient of B[j,k] in the r-th product
        W[k*n+i, r] = coefficient of C[i,k] in the r-th product
        """
        # Define the 7 Strassen products
        # Each product is a bilinear form in A and B.
        # We'll represent each as a list of (A_idx, B_idx, coeff) pairs.
        
        # m1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
        m1_A = [(0, A_j0, 1), (0, A_j1, 0), (1, A_j0, 0), (1, A_j1, 1)]
        m1_B = [(B_j0, k0, 1), (B_j0, k1, 1), (B_j1, k0, 1), (B_j1, k1, 1)]
        # Wait, this is getting complicated. Let me just hardcode the coefficients.
        
        # Let me use a different approach. I'll define the 7 terms explicitly.
        
        # Term 0: m1 = (a+d)(e+h)
        # a = A[0, A_j0], d = A[1, A_j1]
        # e = B[B_j0, k0], h = B[B_j1, k1]
        # m1 = a*e + a*h + d*e + d*h
        # U[0*m + A_j0, base] += 1  (a)
        # U[1*m + A_j1, base] += 1  (d)
        # V[B_j0*p + k0, base] += 1  (e)
        # V[B_j1*p + k1, base] += 1  (h)
        # But this gives a*e + d*h, not a*e + a*h + d*e + d*h.
        # I need to be more careful.
        
        # Actually, the bilinear form for m1 is:
        # m1 = (A[0,A_j0] + A[1,A_j1]) * (B[B_j0,k0] + B[B_j1,k1])
        # = A[0,A_j0]*B[B_j0,k0] + A[0,A_j0]*B[B_j1,k1] + A[1,A_j1]*B[B_j0,k0] + A[1,A_j1]*B[B_j1,k1]
        
        # So for term base:
        # U[0*m + A_j0, base] = 1
        # U[1*m + A_j1, base] = 1
        # V[B_j0*p + k0, base] = 1
        # V[B_j1*p + k1, base] = 1
        # And the product is (A[0,A_j0] + A[1,A_j1]) * (B[B_j0,k0] + B[B_j1,k1])
        # But that's not right either. The bilinear form is:
        # sum_{i,j} U[i*m+j, r] * A[i,j] * sum_{j',k} V[j'*p+k, r] * B[j',k]
        # = (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j',k} V[j'*p+k, r] * B[j',k])
        
        # So for m1, we want:
        # sum_{i,j} U[i*m+j, r] * A[i,j] = A[0,A_j0] + A[1,A_j1]
        # sum_{j',k} V[j'*p+k, r] * B[j',k] = B[B_j0,k0] + B[B_j1,k1]
        
        # So:
        # U[0*m + A_j0, r] = 1
        # U[1*m + A_j1, r] = 1
        # V[B_j0*p + k0, r] = 1
        # V[B_j1*p + k1, r] = 1
        
        # Then the product is (A[0,A_j0] + A[1,A_j1]) * (B[B_j0,k0] + B[B_j1,k1])
        # = A[0,A_j0]*B[B_j0,k0] + A[0,A_j0]*B[B_j1,k1] + A[1,A_j1]*B[B_j0,k0] + A[1,A_j1]*B[B_j1,k1]
        
        # But m1 = (a+d)(e+h) where a=A[0,A_j0], d=A[1,A_j1], e=B[B_j0,k0], h=B[B_j1,k1]
        # = a*e + a*h + d*e + d*h
        # = A[0,A_j0]*B[B_j0,k0] + A[0,A_j0]*B[B_j1,k1] + A[1,A_j1]*B[B_j0,k0] + A[1,A_j1]*B[B_j1,k1]
        
        # Yes, that matches!
        
        # Now for the W coefficients:
        # C[0,k0] = m1 + m4 - m5 + m7
        # C[0,k1] = m3 + m5
        # C[1,k0] = m2 + m4
        # C[1,k1] = m1 - m2 + m3 + m6
        
        # So for term base (m1):
        # W[k0*n + 0, base] = 1  (C[0,k0] += m1)
        # W[k1*n + 1, base] = 1  (C[1,k1] += m1)
        
        # Let me implement this more systematically.
        
        # I'll define the 7 terms as tuples of (U_coeff, V_coeff, W_coeff)
        # where U_coeff is a dict mapping (i,j) to coefficient, etc.
        
        # This is getting too complicated. Let me just hardcode the 22 terms.
        
        pass
    
    # Let me take a different approach. I'll just directly construct the 22 terms.
    
    # Terms 0-6: Strassen for A[:,0:2] @ B[0:2,0:2]
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen:
    # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # m2 = (A[1,0] + A[1,1]) * B[0,0]
    # m3 = A[0,0] * (B[0,1] - B[1,1])
    # m4 = A[1,1] * (B[1,0] - B[0,0])
    # m5 = (A[0,0] + A[0,1]) * B[1,1]
    # m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Term 0: m1
    U[0*3 + 0, 0] = 1  # A[0,0]
    U[1*3 + 1, 0] = 1  # A[1,1]
    V[0*4 + 0, 0] = 1  # B[0,0]
    V[1*4 + 1, 0] = 1  # B[1,1]
    W[0*2 + 0, 0] = 1  # C[0,0]
    W[1*2 + 1, 0] = 1  # C[1,1]
    
    # Term 1: m2
    U[1*3 + 0, 1] = 1  # A[1,0]
    U[1*3 + 1, 1] = 1  # A[1,1]
    V[0*4 + 0, 1] = 1  # B[0,0]
    W[0*2 + 1, 1] = 1  # C[1,0]
    
    # Term 2: m3
    U[0*3 + 0, 2] = 1  # A[0,0]
    V[0*4 + 1, 2] = 1  # B[0,1]
    V[1*4 + 1, 2] = -1  # B[1,1]
    W[1*2 + 0, 2] = 1  # C[0,1]
    
    # Term 3: m4
    U[1*3 + 1, 3] = 1  # A[1,1]
    V[1*4 + 0, 3] = 1  # B[1,0]
    V[0*4 + 0, 3] = -1  # B[0,0]
    W[0*2 + 0, 3] = 1  # C[0,0]
    W[0*2 + 1, 3] = 1  # C[1,0]
    
    # Term 4: m5
    U[0*3 + 0, 4] = 1  # A[0,0]
    U[0*3 + 1, 4] = 1  # A[0,1]
    V[1*4 + 1, 4] = 1  # B[1,1]
    W[0*2 + 0, 4] = -1  # C[0,0]
    W[1*2 + 0, 4] = 1  # C[0,1]
    
    # Term 5: m6
    U[1*3 + 0, 5] = 1  # A[1,0]
    U[0*3 + 0, 5] = -1  # A[0,0]
    V[0*4 + 0, 5] = 1  # B[0,0]
    V[0*4 + 1, 5] = 1  # B[0,1]
    W[1*2 + 1, 5] = 1  # C[1,1]
    
    # Term 6: m7
    U[0*3 + 1, 6] = 1  # A[0,1]
    U[1*3 + 1, 6] = -1  # A[1,1]
    V[1*4 + 0, 6] = 1  # B[1,0]
    V[1*4 + 1, 6] = 1  # B[1,1]
    W[0*2 + 0, 6] = 1  # C[0,0]
    
    # Terms 7-13: Strassen for A[:,0:2] @ B[0:2,2:4]
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x2: B[0,2], B[0,3], B[1,2], B[1,3]
    # C is 2x2: C[0,2], C[0,3], C[1,2], C[1,3]
    
    # Same structure, but with k offset by 2.
    
    # Term 7: m1
    U[0*3 + 0, 7] = 1  # A[0,0]
    U[1*3 + 1, 7] = 1  # A[1,1]
    V[0*4 + 2, 7] = 1  # B[0,2]
    V[1*4 + 3, 7] = 1  # B[1,3]
    W[2*2 + 0, 7] = 1  # C[0,2]
    W[3*2 + 1, 7] = 1  # C[1,3]
    
    # Term 8: m2
    U[1*3 + 0, 8] = 1  # A[1,0]
    U[1*3 + 1, 8] = 1  # A[1,1]
    V[0*4 + 2, 8] = 1  # B[0,2]
    W[2*2 + 1, 8] = 1  # C[1,2]
    
    # Term 9: m3
    U[0*3 + 0, 9] = 1  # A[0,0]
    V[0*4 + 3, 9] = 1  # B[0,3]
    V[1*4 + 3, 9] = -1  # B[1,3]
    W[3*2 + 0, 9] = 1  # C[0,3]
    
    # Term 10: m4
    U[1*3 + 1, 10] = 1  # A[1,1]
    V[1*4 + 2, 10] = 1  # B[1,2]
    V[0*4 + 2, 10] = -1  # B[0,2]
    W[2*2 + 0, 10] = 1  # C[0,2]
    W[2*2 + 1, 10] = 1  # C[1,2]
    
    # Term 11: m5
    U[0*3 + 0, 11] = 1  # A[0,0]
    U[0*3 + 1, 11] = 1  # A[0,1]
    V[1*4 + 3, 11] = 1  # B[1,3]
    W[2*2 + 0, 11] = -1  # C[0,2]
    W[3*2 + 0, 11] = 1  # C[0,3]
    
    # Term 12: m6
    U[1*3 + 0, 12] = 1  # A[1,0]
    U[0*3 + 0, 12] = -1  # A[0,0]
    V[0*4 + 2, 12] = 1  # B[0,2]
    V[0*4 + 3, 12] = 1  # B[0,3]
    W[3*2 + 1, 12] = 1  # C[1,3]
    
    # Term 13: m7
    U[0*3 + 1, 13] = 1  # A[0,1]
    U[1*3 + 1, 13] = -1  # A[1,1]
    V[1*4 + 2, 13] = 1  # B[1,2]
    V[1*4 + 3, 13] = 1  # B[1,3]
    W[2*2 + 0, 13] = 1  # C[0,2]
    
    # Terms 14-21: A[:,2] * B[2,:]
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,1,2,3}
    # 8 terms.
    
    for idx, (i, k) in enumerate([(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]):
        r = 14 + idx
        U[i*3 + 2, r] = 1  # A[i,2]
        V[2*4 + k, r] = 1  # B[2,k]
        W[k*2 + i, r] = 1  # C[i,k]
    
    # Verify the decomposition
    def verify(U, V, W, A, B):
        C_target = A @ B
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_calc[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] * A[i, j] * B[j, k]
        return np.allclose(C_calc, C_target)
    
    # Test with random matrices
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        if not verify(U, V, W, A, B):
            # Fallback to rank 24
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
