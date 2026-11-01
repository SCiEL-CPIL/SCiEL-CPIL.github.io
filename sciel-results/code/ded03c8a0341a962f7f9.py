"""Find rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,k,j] = 1 if indices match appropriately
    # Indexing:
    #   U row: i*m + j  (n*m entries)
    #   V row: j*p + k  (m*p entries)
    #   W row: k*n + i  (p*n entries)
    
    # For the multiplication tensor, T[a,b,c] = 1 when:
    #   a = i*m+j, b = j'*p+k, c = k'*n+i'
    #   and i=i', j=j', k=k'
    # So T[a,b,c] = 1 iff (a//m == c%p) and (a%m == b//p) and (b%p == c//n)
    
    # We want: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # This is equivalent to: for each (a,c), sum_r U[a,r]*W[c,r] * V[b,r] = T[a,b,c] for all b
    # Better: for each r, define M_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r M_r = T
    
    # Alternative approach: treat it as a system where for each column r,
    # the "slice" U[:,r] * V[:,r]^T * W[:,r] (in appropriate reshaping) should sum to T.
    # 
    # Let's use the fact that for fixed r, define:
    #   X_r[i,j] = U[i*m+j, r]  (n x m)
    #   Y_r[j,k] = V[j*p+k, r]  (m x p)  
    #   Z_r[k,i] = W[k*n+i, r]  (p x n)
    # Then the contribution of term r to C[i,k] is sum_j X_r[i,j] * Y_r[j,k] * Z_r[k,i]
    # Wait, that's not quite right. Let me re-derive.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-r decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k']) ... 
    # No, that's not how tensor decomposition works for matrix mult.
    #
    # Standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_r sum_j sum_k' U[i,j,r] * V[j,k',r] * W[k',i,r] * A[i,j] * B[j,k']
    # Wait no. The bilinear complexity formulation:
    # Each term r contributes: sum_j A[i,j] * (something) * B[j,k]
    # Actually: C[i,k] = sum_r f_r(i,k; A) * g_r(i,k; B) where f_r depends on A and g_r on B
    # More precisely: C[i,k] = sum_r sum_j sum_k' U[r, i, j] A[i,j] * V[r, j, k'] B[j,k'] * W[r, k', i]
    # Hmm, let me use the standard formulation.
    #
    # The multiplication tensor T has T[i,j,k] = 1 for the (i,j,k) trilinear form:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where C[i,k] = sum_j A[i,j]B[j,k]
    # 
    # In bilinear complexity, rank r means:
    # C[i,k] = sum_{r=1}^r (sum_j a_{r,i,j} A[i,j]) (sum_{k'} b_{r,j,k'} B[j,k'])
    # No wait, it should be:
    # C[i,k] = sum_r (sum_j u_{r,i,j} A[i,j]) (sum_{j',k'} v_{r,j',k'} B[j',k'])
    # But that doesn't work dimensionally unless we're more careful.
    #
    # Correct formulation: 
    # C[i,k] = sum_r sum_j U_{r,i,j} * A[i,j] * (sum_{k'} V_{r,j,k'} * B[j,k'] * W_{r,k',i})
    # This is getting confusing. Let me use the standard rank-r decomposition.
    #
    # For matrix multiplication C = AB where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r algorithm computes C using r scalar multiplications.
    # C[i,k] = sum_{r=1}^r P_r * Q_r[i,k]
    # where P_r = sum_{i,j} u_{r,i,j} A[i,j] and Q_r = (sum_{j,k} v_{r,j,k} B[j,k]) * w_{r,i,k}
    # Hmm, that's not right either.
    #
    # Standard: C[i,k] = sum_r (sum_{i',j} U[r,i',j] A[i',j]) * (sum_{j',k'} V[r,j',k'] B[j',k'])
    # This only works if we have the right structure. Actually the correct form is:
    # 
    # C[i,k] = sum_r sum_j U[r,i,j] V[r,j,k] A[i,j] B[j,k]  -- NO, this is just schoolbook with extra factors.
    #
    # Let me think again. The tensor for matrix multiplication is:
    # M_{n×m×p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k  (in appropriate spaces)
    # where the three factors are: A's index (i,j), B's index (j,k), and output (i,k)
    #
    # Rank r decomposition: M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
    # where u_r is in the space of A-indices, v_r in B-indices, w_r in output-indices
    #
    # Concretely: define for each r:
    #   u_r: vector in R^{n*m} (indexed by (i,j))
    #   v_r: vector in R^{m*p} (indexed by (j,k)) 
    #   w_r: vector in R^{p*n} (indexed by (k,i))
    # 
    # Then: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = T[i,j,k]
    # where T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0.
    #
    # For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]:
    # The tensor element T[i,j,k] in the trilinear form sum T[i,j,k] A[i,j] B[j',k'] ... 
    # Actually I need to be more careful.
    #
    # The bilinear map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The associated tensor has indices: first factor indexed by (i,j) for A, 
    # second by (j',k') for B, third by (i',k') for C.
    # T[(i,j),(j',k'),(i',k')] = 1 if i=i', j=j', k=k' else 0.
    #
    # So: sum_r U[(i,j), r] * V[(j',k'), r] * W[(i',k'), r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    #
    # With our indexing: U row = i*m+j, V row = j'*p+k', W row = k'*n+i'
    # We need: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = 1 if i=i', j=j', k=k' else 0
    #
    # So the constraint is: for all (i,j,j',k',i',k'), 
    #   sum_r U[i*m+j, r] V[j'*p+k', r] W[k'*n+i', r] = [i==i' and j==j' and k==k']
    
    # This is a system of (n*m)*(m*p)*(p*n) equations with rank*3*(...) unknowns... 
    # Actually it's nonlinear (products of U, V, W entries).
    
    # Strategy: For small cases, we can try to find solutions by:
    # 1. Fixing a structure (e.g., entries in {-1, 0, 1, 1/2})
    # 2. Using algebraic approaches
    
    # For 2x2 * 2x4, schoolbook rank is 2*2*4 = 16.
    # Can we do better? Let's think about what's possible.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    # Each C[i,k] is a sum of 2 products. The question is whether we can share computations.
    # 
    # Lower bound: For C[i,k], we need at least the products involving A[i,*] and B[*,k].
    # The A-factors are A[0,0], A[0,1], A[1,0], A[1,1] (4 total)
    # The B-factors are B[0,0..3], B[1,0..3] (8 total)
    # 
    # A rank-r decomposition uses r scalar multiplications. Each scalar multiplication
    # multiplies one linear combination of A entries by one linear combination of B entries.
    # 
    # For the 2x2 * 2x4 case, let's try to find if rank < 16 is possible.
    # 
    # One approach: group by rows of A. 
    # Row 0 of A: [A[0,0], A[0,1]] produces row 0 of C: [C[0,0..3]]
    # Row 1 of A: [A[1,0], A[1,1]] produces row 1 of C: [C[1,0..3]]
    # 
    # Row 0 of C = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # This is a 2x4 outer-product-like combination. The rank of the bilinear map
    # (A[0,:], B) -> C[0,:] is at most 2 (since it's a sum of 2 rank-1 bilinear terms).
    # But we need exactness.
    #
    # Actually, for the sub-problem of computing C[0,:] from A[0,:] and B:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0,1,2,3
    # This requires, in the worst case, 2*4 = 8 multiplications if done separately,
    # but since A[0,0] and A[0,1] are shared across k, we might do better.
    # However, each product A[0,j]*B[j,k] is a distinct scalar that must be computed.
    # Unless we can combine them.
    #
    # For example: (A[0,0] + A[0,1])*(B[0,k] + B[1,k]) = A[0,0]B[0,k] + A[0,0]B[1,k] + A[0,1]B[0,k] + A[0,1]B[1,k]
    # This gives us cross terms we don't want. So we can't simply combine.
    #
    # The rank of the 2x2*2xn matrix multiplication is known. For 2x2*2x2 it's 7 (Strassen).
    # For 2x2*2x4, by block structure, it might be related.
    #
    # Actually, 2x2 * 2x4 can be seen as two 2x2 * 2x2 multiplications side by side:
    # C[:, 0:2] = A * B[:, 0:2] and C[:, 2:4] = A * B[:, 2:4]
    # Each 2x2*2x2 has rank 7, so total rank at most 14.
    # Can we share between the two blocks? Possibly.
    #
    # Let me try to construct a rank-14 solution using Strassen for each half.
    #
    # Strassen for 2x2 * 2x2:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For 2x2 * 2x4, split B into B_left (2x2) and B_right (2x2):
    # B_left = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # B_right = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
    
    # Using Strassen for each:
    # Left side (7 mults):
    # L1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # L2 = (A[1,0]+A[1,1])*B[0,0]
    # L3 = A[0,0]*(B[0,1]-B[1,1])
    # L4 = A[1,1]*(B[1,0]-B[0,0])
    # L5 = (A[0,0]+A[0,1])*B[1,1]
    # L6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # L7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # Right side (7 mults):
    # R1 = (A[0,0]+A[1,1])*(B[0,2]+B[1,3])
    # R2 = (A[1,0]+A[1,1])*B[0,2]
    # R3 = A[0,0]*(B[0,3]-B[1,3])
    # R4 = A[1,1]*(B[1,2]-B[0,2])
    # R5 = (A[0,0]+A[0,1])*B[1,3]
    # R6 = (A[1,0]-A[0,0])*(B[0,2]+B[0,3])
    # R7 = (A[0,1]-A[1,1])*(B[1,2]+B[1,3])
    
    # Total: 14 multiplications.
    #
    # Now I need to express this in the U, V, W format.
    # Each "multiplication" is one column of U, V, W.
    # For column r: U[:,r] is the linear combination coefficients for A entries,
    # V[:,r] for B entries, W[:,r] tells how this product contributes to each C[i,k].
    
    # Let's build the 14-column solution.
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3  (row = i*m+j)
    # B indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7 (row = j*p+k)
    # C indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7 (row = k*n+i)
    #   W row = k*n + i, so:
    #   k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    #   k=2,i=0 -> 4; k=2,i=1 -> 5; k=3,i=0 -> 6; k=3,i=1 -> 7
    
    rank = 14
    U = [[0.0]*rank for _ in range(4)]  # 4 A-entries
    V = [[0.0]*rank for _ in range(8)]  # 8 B-entries
    W = [[0.0]*rank for _ in range(8)]  # 8 C-entries
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    # Column 0: L1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # Contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] = L1 + L4 - L5 + L7  -> coeff +1
    # C[0,1] = L3 + L5  -> coeff 0
    # C[1,0] = L2 + L4  -> coeff 0  
    # C[1,1] = L1 - L2 + L3 + L6 -> coeff +1
    # Wait, let me recheck Strassen:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    # So C[0,0] gets +1 from L1, C[1,1] gets +1 from L1
    set_col(0,
            {0: 1, 3: 1},           # A[0,0] + A[1,1]
            {0: 1, 5: 1},           # B[0,0] + B[1,1]
            {0: 1, 3: 1})           # C[0,0] + C[1,1]
    
    # Column 1: L2 = (A[1,0]+A[1,1])*B[0,0]
    # C[1,0] = L2 + L4 -> +1; C[1,1] = L1 - L2 + L3 + L6 -> -1
    set_col(1,
            {2: 1, 3: 1},           # A[1,0] + A[1,1]
            {0: 1},                 # B[0,0]
            {4: 1, 7: -1})          # C[1,0] + C[1,1]... 
    # Wait, W index for C[1,0] is k*n+i = 0*2+1 = 1? No.
    # C[i,k]: i is row, k is col. W index = k*n + i.
    # C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
    # C[0,1]: i=0,k=1 -> W[1*2+0] = W[2]
    # C[0,2]: i=0,k=2 -> W[2*2+0] = W[4]
    # C[0,3]: i=0,k=3 -> W[3*2+0] = W[6]
    # C[1,0]: i=1,k=0 -> W[0*2+1] = W[1]
    # C[1,1]: i=1,k=1 -> W[1*2+1] = W[3]
    # C[1,2]: i=1,k=2 -> W[2*2+1] = W[5]
    # C[1,3]: i=1,k=3 -> W[3*2+1] = W[7]
    
    # Let me redo with correct W indices.
    # W[k*n+i]:
    # C[0,0]: k=0,i=0 -> 0
    # C[0,1]: k=1,i=0 -> 2
    # C[0,2]: k=2,i=0 -> 4
    # C[0,3]: k=3,i=0 -> 6
    # C[1,0]: k=0,i=1 -> 1
    # C[1,1]: k=1,i=1 -> 3
    # C[1,2]: k=2,i=1 -> 5
    # C[1,3]: k=3,i=1 -> 7
    
    # Redo:
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    # Col 0: L1 = (A00+A11)(B00+B11)
    # C[0,0]: +1, C[1,1]: +1
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    
    # Col 1: L2 = (A10+A11)B00
    # C[1,0]: +1, C[1,1]: -1
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    
    # Col 2: L3 = A00(B01-B11)
    # C[0,1]: +1, C[1,1]: +1
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    
    # Col 3: L4 = A11(B10-B00)
    # C[0,0]: +1, C[1,0]: +1
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    
    # Col 4: L5 = (A00+A01)B11
    # C[0,0]: -1, C[0,1]: +1
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    # C[1,1]: +1
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    # C[0,0]: +1
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Now right side (columns 7-13) for B[:,2:4]
    # B indices for right side: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    # C[0,2]: +1, C[1,3]: +1
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    
    # Col 8: R2 = (A10+A11)B02
    # C[1,2]: +1, C[1,3]: -1
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    
    # Col 9: R3 = A00(B03-B13)
    # C[0,3]: +1, C[1,3]: +1
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    
    # Col 10: R4 = A11(B12-B02)
    # C[0,2]: +1, C[1,2]: +1
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    
    # Col 11: R5 = (A00+A01)B13
    # C[0,2]: -1, C[0,3]: +1
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    # C[1,3]: +1
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    # C[0,2]: +1
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    # Verify the solution
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # T[i*m+j, j*p+k, k*n+i] should be 1, others 0
                    pass
        # Better: check that sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k'*n+i',r] = delta
        errors = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                actual = sum(U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r] for r in range(rank))
                                if abs(actual - expected) > 1e-10:
                                    errors += (actual - expected)**2
        return errors
    
    err = verify(U, V, W, n, m, p)
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_fb = n * m * p
    U_fb = [[0]*rank_fb for _ in range(n*m)]
    V_fb = [[0]*rank_fb for _ in range(m*p)]
    W_fb = [[0]*rank_fb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
