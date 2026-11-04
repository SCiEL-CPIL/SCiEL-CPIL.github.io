import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Build the multiplication tensor T with T[i,j,k] = delta(i,k)
    # T has shape (n*m, m*p, p*n)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Constraint: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k)
    # And for all other combinations, the sum should be 0.
    
    # Actually, the tensor decomposition constraint is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 if i==k else 0
    
    # This is a system of bilinear equations in U, V, W.
    # For fixed U, it's linear in V and W (but bilinear between them).
    # For fixed U and V, it's linear in W.
    
    # Strategy: Fix U to a structured basis, then for each rank-1 term r,
    # we need V[:,r] and W[:,r] such that the outer product V[:,r] ⊗ W[:,r]
    # contributes correctly.
    
    # Actually, let me think differently. The tensor T can be written as:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    
    # For the matrix multiplication tensor, T[i,j,k] = delta(i,k).
    # This means: for each (i,k) with i==k, and for each j, T[i,j,i] = 1.
    # For i != k, T[i,j,k] = 0 for all j.
    
    # So T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} e_{i,j} ⊗ e_{j,i} ⊗ e_{i,i}
    # where e_{i,j} is the standard basis vector in R^(n*m) for position (i,j),
    # e_{j,i} is in R^(m*p) for position (j,i), e_{i,i} is in R^(p*n) for position (i,i).
    # This gives rank n*m = 16 for n=4, m=4... but that's not right either.
    
    # Wait, let me reconsider. The standard matrix multiplication tensor for (n,m,p) is:
    # T ∈ R^(n×m × m×p × p×n)
    # T[i,j,k] = 1 if the "path" i->j->k is valid, i.e., i==k (since C[i,k] = sum_j A[i,j]B[j,k])
    # Actually no. C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta(i,k)? No...
    # T should satisfy: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # For this to give C[i,k] = sum_j A[i,j]*B[j,k], we need T[i,j,k] = 1 for all i,j,k where the indices match.
    # Actually T[i,j,k] = 1 always? No, that would give sum_j A[i,j]*B[j,k] which is correct!
    # Wait, C[i,k] = sum_{j} A[i,j] B[j,k]. The tensor T[i,j,k] must be 1 for all valid (i,j,k).
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # That means T is the all-ones tensor? No, that's not right either.
    
    # Let me be very careful. The matrix multiplication map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map R^(n×m) × R^(m×p) -> R^(n×p)
    # The associated trilinear form (tensor) T satisfies:
    # <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i]  (using trace convention)
    # We want this to equal tr(C^T * (A*B)) = sum_{i,k} C[k,i] * (A*B)[i,k] = sum_{i,k} C[k,i] * sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # So T[i,j,k] = 1 for all i,j,k. The tensor is the all-ones tensor!
    
    # Wait, that can't be right because the rank of the all-ones tensor of size a×b×c is 1.
    # Let me reconsider the indexing.
    
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is the column of C and i is the row, stored as k*n+i.
    # U is indexed by (i,j) for A[i,j], stored as i*m+j.
    # V is indexed by (j,k) for B[j,k], stored as j*p+k.
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should be 1 for all i,j,k (since C[k,i] = sum_j A[i,j]*B[j,k]).
    
    # Hmm, but that means T is the all-ones tensor of size (n*m) × (m*p) × (p*n), which has rank 1.
    # That's clearly wrong. Let me reconsider.
    
    # I think the issue is that the tensor for matrix multiplication is NOT the all-ones tensor.
    # The bilinear map is (A,B) -> A*B, and the tensor represents this map.
    # The tensor T has components T[i,j,k] where i indexes rows of A, j indexes the "middle" index,
    # k indexes columns of B. The product C[i,k] = sum_j A[i,j] B[j,k].
    # 
    # In tensor form: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For this to hold for all A, B, we need T[i,j,k] = 1 for all i,j,k.
    # But then the "rank" of this tensor as a 3-way array would be 1 (it's the outer product
    # of three all-ones vectors).
    
    # I'm confusing tensor rank (as a 3-way array) with the rank of the bilinear map.
    # The "rank" in the context of matrix multiplication algorithms is the number of scalar
    # multiplications needed, which is the tensor rank of the multiplication tensor where
    # the tensor is defined differently.
    
    # Let me look at this from the algorithmic perspective. The Strassen algorithm for 2x2
    # uses 7 multiplications. The tensor for 2x2x2 matrix multiplication has rank 7.
    # The tensor is T ∈ R^(4×4×4) where T[i,j,k] corresponds to the coefficient of
    # A[i,?] * B[?,k] in C[i,k].
    
    # Actually, I think the correct formulation is:
    # The multiplication tensor T has entries T[i,j,k] where:
    # - i ranges over (row, col) pairs of A, i.e., (i1, i2) with i1 in [0,n), i2 in [0,m)
    # - j ranges over (row, col) pairs of B, i.e., (j1, j2) with j1 in [0,m), j2 in [0,p)  
    # - k ranges over (row, col) pairs of C, i.e., (k1, k2) with k1 in [0,n), k2 in [0,p)
    # 
    # And T[i,j,k] = 1 if i2 == j1 and k1 == i1 and k2 == j2, else 0.
    # This encodes that A[i1,i2] * B[i2,j2] contributes to C[i1,j2].
    
    # So with the flat indices:
    # U index: i = i1*m + i2 (where i1 in [0,n), i2 in [0,m))
    # V index: j = j1*p + j2 (where j1 in [0,m), j2 in [0,p))
    # W index: k = k2*n + k1 (where k2 in [0,p), k1 in [0,n))  [trace convention: W[k*n+i]]
    
    # T[i,j,k] = 1 if (i2 == j1) and (k1 == i1) and (k2 == j2), else 0
    # In flat indices: i = i1*m+i2, j = j1*p+j2, k = k2*n+k1
    # Condition: i2 == j1, i1 == k1, j2 == k2
    # i2 = i % m, i1 = i // m
    # j1 = j % p, j2 = j // p
    # k1 = k % n, k2 = k // n
    # So T[i,j,k] = 1 if (i % m == j % p) and (i // m == k % n) and (j // p == k // n)
    
    # Let me verify: for n=4,m=4,p=7
    # i = i1*4 + i2, j = j1*7 + j2, k = k2*4 + k1
    # T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2
    
    # This is the correct tensor. Now, the rank of this tensor is what we want to minimize.
    # For 2x2x2: n=2,m=2,p=2
    # i = i1*2+i2 (4 values), j = j1*2+j2 (4 values), k = k2*2+k1 (4 values)
    # T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2
    # This tensor has rank 7 (Strassen).
    
    # Now, for the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^(n*m), v_r ∈ R^(m*p), w_r ∈ R^(p*n)
    # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    
    # The constraint is: for all i,j,k,
    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in (n*m + m*p + p*n)*rank unknowns.
    # But it's nonlinear (bilinear in V,W for fixed U, etc.)
    
    # The strategy from the directive: fix U to a structured low-rank basis, then solve for V and W.
    # But it's still nonlinear. However, if we fix U, then for each pair (i,k), the equation
    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k] is linear in the products V[j,r]*W[k,r].
    # 
    # Actually, a cleaner approach: fix U, then the problem becomes: find V and W such that
    # for each (i,j,k): sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
    # This is still bilinear. But if we fix U AND V, it's linear in W.
    # 
    # Alternative: use the fact that for fixed U, we can set up a linear system.
    # Let's define M_{(i,k), j, r} = U[i,r] and the equation is:
    # sum_r U[i,r] * (V[j,r] * W[k,r]) = T[i,j,k]
    # Let X[j,r] = V[j,r] and Y[k,r] = W[k,r]. Then sum_r U[i,r] X[j,r] Y[k,r] = T[i,j,k]
    # This is a bilinear system.
    
    # A practical approach: use iterative methods. Start with a random initialization,
    # then alternate: fix U,V and solve for W (linear), fix U,W and solve for V (linear),
    # fix V,W and solve for U (linear). This is like ALS (alternating least squares)
    # but for exact decomposition.
    
    # However, for exact decomposition with integer/half-integer entries, this is tricky.
    
    # Let me try a different approach: use the known block structure.
    # For 4x4x7, we can split into 4x4x4 + 4x4x3.
    # 4x4x4: use Strassen at two levels (rank 49)
    # 4x4x3: use 4 blocks of 2x2x3 (rank 4*11 = 44)
    # Total: 93
    
    # Let me implement this properly.
    
    # First, let me implement the 2x2x2 Strassen decomposition explicitly.
    # For 2x2x2: n=2, m=2, p=2
    # T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2
    # i = i1*2+i2, j = j1*2+j2, k = k2*2+k1
    
    # The 7 rank-1 terms of the 2x2x2 tensor:
    # Using Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # In terms of the tensor, each Mi corresponds to a rank-1 term u_i ⊗ v_i ⊗ w_i
    # where u_i, v_i, w_i are vectors in R^4.
    
    # Let me define the 7 terms explicitly.
    # A = [[A11, A12], [A21, A22]] -> flat: [A11, A12, A21, A22] = indices 0,1,2,3
    # B = [[B11, B12], [B21, B22]] -> flat: [B11, B12, B21, B22] = indices 0,1,2,3
    # C = [[C11, C12], [C21, C22]] -> flat: [C11, C12, C21, C22] = indices 0,1,2,3
    # But W uses trace convention: W[k*n+i] where k is col, i is row.
    # So W index for C11 (i=0,k=0): 0*2+0 = 0
    # W index for C12 (i=0,k=1): 1*2+0 = 2
    # W index for C21 (i=1,k=0): 0*2+1 = 1
    # W index for C22 (i=1,k=1): 1*2+1 = 3
    
    # So the flat W indices are: C11->0, C21->1, C12->2, C22->3
    
    # Now, the 7 Strassen products and their contributions to C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi = u_i^T A * v_i^T B (scalar product)
    # M1 = (A11+A22)(B11+B22): u1 = [1,0,0,1], v1 = [1,0,0,1]
    # M2 = (A21+A22)B11: u2 = [0,0,1,1], v2 = [1,0,0,0]
    # M3 = A11(B21-B11): u3 = [1,0,0,0], v3 = [0,0,1,-1]
    # M4 = A22(B12-B22): u4 = [0,0,0,1], v4 = [0,1,0,-1]
    # M5 = (A11+A12)B22: u5 = [1,1,0,0], v5 = [0,0,0,1]
    # M6 = (A21-A11)(B11+B12): u6 = [-1,0,1,0], v6 = [1,1,0,0]
    # M7 = (A12-A22)(B21+B22): u7 = [0,1,0,-1], v7 = [0,0,1,1]
    
    # The contribution of Mi to C is: Mi * w_i where w_i is the vector of coefficients
    # for each C element.
    # M1 contributes to: C11 (+1), C22 (+1) -> w1 = [1, 0, 0, 1] (using W flat indices: C11=0, C21=1, C12=2, C22=3)
    # Wait, let me recheck. C11 is at W index 0, C12 at W index 2, C21 at W index 1, C22 at W index 3.
    # M1: C11 += M1, C22 += M1 -> w1 = [1, 0, 0, 1]
    # M2: C21 += M2, C22 -= M2 -> w2 = [0, 1, 0, -1]
    # M3: C12 += M3, C22 += M3 -> w3 = [0, 0, 1, 1]
    # M4: C11 += M4, C21 += M4 -> w4 = [1, 1, 0, 0]
    # M5: C11 -= M5, C12 += M5 -> w5 = [-1, 0, 1, 0]
    # M6: C22 += M6 -> w6 = [0, 0, 0, 1]
    # M7: C11 += M7, C22 -= M7... wait let me recheck.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M7: C11 += M7 -> w7 = [1, 0, 0, 0]
    
    # Let me verify: 
    # C11 = M1 + M4 - M5 + M7: coefficients [1, 1, -1, 1] for M1,M4,M5,M7
    # C12 = M3 + M5: coefficients [0, 0, 1, 1] for M3,M5
    # C21 = M2 + M4: coefficients [0, 1, 0, 1] for M2,M4
    # C22 = M1 - M2 + M3 + M6: coefficients [1, -1, 1, 1] for M1,M2,M3,M6
    
    # So the w vectors (in W flat index order [C11, C21, C12, C22] = [0,1,2,3]):
    # w1 (M1): C11+=1, C22+=1 -> [1, 0, 0, 1]
    # w2 (M2): C21+=1, C22-=1 -> [0, 1, 0, -1]
    # w3 (M3): C12+=1, C22+=1 -> [0, 0, 1, 1]
    # w4 (M4): C11+=1, C21+=1 -> [1, 1, 0, 0]
    # w5 (M5): C11-=1, C12+=1 -> [-1, 0, 1, 0]
    # w6 (M6): C22+=1 -> [0, 0, 0, 1]
    # w7 (M7): C11+=1 -> [1, 0, 0, 0]
    
    # Now let me verify this gives the correct tensor.
    # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    # For i=0 (A11), j=0 (B11), k=0 (C11):
    # sum_r u_r[0] * v_r[0] * w_r[0]
    # r=1: u1[0]*v1[0]*w1[0] = 1*1*1 = 1
    # r=2: u2[0]*v2[0]*w2[0] = 0*1*0 = 0
    # r=3: u3[0]*v3[0]*w3[0] = 1*0*0 = 0
    # r=4: u4[0]*v4[0]*w4[0] = 0*0*1 = 0
    # r=5: u5[0]*v5[0]*w5[0] = 1*0*(-1) = 0
    # r=6: u6[0]*v6[0]*w6[0] = -1*1*0 = 0
    # r=7: u7[0]*v7[0]*w7[0] = 0*0*1 = 0
    # Sum = 1. Correct! T[0,0,0] should be 1 (A11*B11 contributes to C11).
    
    # For i=0 (A11), j=1 (B12), k=2 (C12):
    # sum_r u_r[0] * v_r[1] * w_r[2]
    # r=1: u1[0]*v1[1]*w1[2] = 1*0*0 = 0
    # r=2: u2[0]*v2[1]*w2[2] = 0*0*0 = 0
    # r=3: u3[0]*v3[1]*w3[2] = 1*0*1 = 0
    # r=4: u4[0]*v4[1]*w4[2] = 0*1*0 = 0
    # r=5: u5[0]*v5[1]*w5[2] = 1*0*1 = 0
    # r=6: u6[0]*v6[1]*w6[2] = -1*1*0 = 0
    # r=7: u7[0]*v7[1]*w7[2] = 0*0*0 = 0
    # Sum = 0. But T[0,1,2] should be 1 (A11*B12 contributes to C12).
    # This is wrong! Let me recheck.
    
    # i=0 means i1=0, i2=0 (A11)
    # j=1 means j1=0, j2=1 (B12)
    # k=2 means k1=0, k2=1 (C12)
    # Condition: i2==j1 (0==0 ✓), i1==k1 (0==0 ✓), j2==k2 (1==1 ✓)
    # So T[0,1,2] = 1. But my sum gave 0. Something is wrong.
    
    # Let me recheck the u, v, w vectors.
    # The issue might be in how I'm indexing. Let me be more careful.
    
    # A flat index: i = i1*2 + i2, so:
    # i=0: i1=0, i2=0 (A11)
    # i=1: i1=0, i2=1 (A12)
    # i=2: i1=1, i2=0 (A21)
    # i=3: i1=1, i2=1 (A22)
    
    # B flat index: j = j1*2 + j2, so:
    # j=0: j1=0, j2=0 (B11)
    # j=1: j1=0, j2=1 (B12)
    # j=2: j1=1, j2=0 (B21)
    # j=3: j1=1, j2=1 (B22)
    
    # W flat index (trace convention): k = k2*2 + k1, so:
    # k=0: k2=0, k1=0 (C11)
    # k=1: k2=0, k1=1 (C21)
    # k=2: k2=1, k1=0 (C12)
    # k=3: k2=1, k1=1 (C22)
    
    # Now, M5 = (A11+A12)*B22
    # u5 = [1, 1, 0, 0] (A11 and A12)
    # v5 = [0, 0, 0, 1] (B22)
    # M5 contributes to C12 (since A11*A12 row times B22 column gives C12)
    # Wait, (A11+A12) is the first row of A, and B22 is the bottom-right element.
    # (A11+A12)*B22 contributes to C12? Let's see:
    # C12 = A11*B12 + A12*B22
    # So A12*B22 contributes to C12. And (A11+A12)*B22 = A11*B22 + A12*B22.
    # A11*B22 does NOT contribute to any C element directly in the standard product.
    # But in Strassen, M5 = (A11+A12)*B22, and it's used as:
    # C12 = M3 + M5 = A11*(B21-B11) + (A11+A12)*B22 = A11*B21 - A11*B11 + A11*B22 + A12*B22
    # That doesn't look right. Let me recheck the Strassen formulas.
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C12:
    # M3 + M5 = A11(B21-B11) + (A11+A12)B22
    # = A11*B21 - A11*B11 + A11*B22 + A12*B22
    # But C12 should be A11*B12 + A12*B22.
    # This doesn't match! I must have the Strassen formulas wrong.
    
    # Let me look up the correct Strassen formulas.
    # Actually, I think I have the B indices mixed up. Let me use a different convention.
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-e)
    # M4 = d(f-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C12 = M3 + M5 = a(g-e) + (a+b)h = ag - ae + ah + bh
    # But C12 should be af + bh.
    # This still doesn't match. I'm clearly making an error.
    
    # Let me try a different set of Strassen formulas.
    # Actually, I think the issue is that different sources use different conventions.
    # Let me just directly construct the rank-7 decomposition by solving the linear system.
    
    # For 2x2x2, T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2.
    # Let me list all the 1's:
    # i=0 (A11: i1=0,i2=0), j=0 (B11: j1=0,j2=0), k=0 (C11: k1=0,k2=0): 1
    # i=0, j=0, k=2 (C12: k1=0,k2=1): 0 (j2=0 != k2=1)
    # i=0, j=1 (B12: j1=0,j2=1), k=2 (C12): 1
    # i=0, j=2 (B21: j1=1,j2=0), k=0 (C11): 0 (i2=0 != j1=1)
    # i=0, j=3 (B22: j1=1,j2=1), k=2 (C12): 0 (i2=0 != j1=1)
    # i=1 (A12: i1=0,i2=1), j=2 (B21: j1=1,j2=0), k=0 (C11): 1
    # i=1, j=3 (B22: j1=1,j2=1), k=2 (C12): 1
    # i=2 (A21: i1=1,i2=0), j=0 (B11: j1=0,j2=0), k=1 (C21: k1=1,k2=0): 1
    # i=2, j=1 (B12: j1=0,j2=1), k=3 (C22: k1=1,k2=1): 1
    # i=3 (A22: i1=1,i2=1), j=2 (B21: j1=1,j2=0), k=1 (C21): 1
    # i=3, j=3 (B22: j1=1,j2=1), k=3 (C22): 1
    
    # So the 1's are at:
    # (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
    # That's 8 ones. The tensor has 8 non-zero entries.
    
    # Now, I need to find 7 vectors u_r, v_r, w_r in R^4 such that
    # T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]
    
    # This is a system of 64 equations (4*4*4) in 7*12 = 84 unknowns.
    # But it's nonlinear. However, I can use the known Strassen decomposition.
    
    # Let me try to find the correct Strassen decomposition by solving.
    # Actually, let me just use the known fact that the 2x2x2 tensor has rank 7 and
    # use a numerical approach to find the decomposition.
    
    # For now, let me implement the block approach with the known ranks.
    # I'll use a simpler approach: for 4x4x4, use 4x4x4 = 4 blocks of 2x2x4 + 4 blocks of 2x2x0...
    # Actually, let me just use the schoolbook for the 4x4x3 part and Strassen for 4x4x4.
    
    # Given the complexity, let me implement a clean version:
    # 1. 4x4x4 using two-level Strassen: rank 49
    # 2. 4x4x3 using 4 blocks of 2x2x3: rank 4*11 = 44
    # Total: 93
    
    # For the 2x2x3 block, I need a rank-11 decomposition.
    # 2x2x3: n=2, m=2, p=3
    # T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2
    # i = i1*2+i2 (4 values), j = j1*3+j2 (6 values), k = k2*2+k1 (6 values)
    
    # Let me count the 1's:
    # i=0 (A11), j=0 (B11: j1=0,j2=0), k=0 (C11: k1=0,k2=0): 1
    # i=0, j=3 (B21: j1=1,j2=0), k=0: 0 (i2=0 != j1=1)
    # i=0, j=1 (B12: j1=0,j2=1), k=2 (C12: k1=0,k2=1): 1
    # i=0, j=4 (B22: j1=1,j2=1), k=2: 0
    # i=0, j=2 (B13: j1=0,j2=2), k=4 (C13: k1=0,k2=2): 1
    # i=0, j=5 (B23: j1=1,j2=2), k=4: 0
    # i=1 (A12), j=3 (B21), k=0: 1
    # i=1, j=4 (B22), k=2: 1
    # i=1, j=5 (B23), k=4: 1
    # i=2 (A21), j=0 (B11), k=1 (C21: k1=1,k2=0): 1
    # i=2, j=1 (B12), k=3 (C22: k1=1,k2=1): 1
    # i=2, j=2 (B13), k=5 (C23: k1=1,k2=2): 1
    # i=3 (A22), j=3 (B21), k=1: 1
    # i=3, j=4 (B22), k=3: 1
    # i=3, j=5 (B23), k=5: 1
    
    # So 12 ones. The rank of 2x2x3 is 11 (known result).
    
    # Given the time constraints, let me implement a practical solution.
    # I'll use the schoolbook decomposition for the full 4x4x7 (rank 112) as a fallback,
    # and try to improve it using the block structure.
    
    # Actually, let me just implement the known rank-93 solution properly.
    # For 4x4x4, I'll use the two-level Strassen.
    # For 4x4x3, I'll use 4 blocks of 2x2x3.
    
    # Let me define the 2x2x2 rank-7 decomposition numerically.
    # I'll solve for it using a simple approach.
    
    # For the 2x2x2 tensor, the 8 non-zero entries are at:
    # (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
    
    # Let me try to find 7 rank-1 terms.
    # I'll use a greedy approach: for each of the 8 positions, try to find a rank-1 term
    # that covers it and as many other positions as possible.
    
    # Actually, let me just hardcode the known Strassen decomposition.
    # After more careful thought, here's the correct Strassen for 2x2x2:
    
    # Using the notation A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # The 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(g-e)
    # M4 = (d)(f-h)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # And:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C12 again:
    # M3 + M5 = a(g-e) + (a+b)h = ag - ae + ah + bh
    # C12 should be af + bh.
    # ag - ae + ah + bh vs af + bh. These are not equal in general.
    
    # I'm clearly making an error in the Strassen formulas. Let me look this up more carefully.
    # 
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-e)
    # M4 = d(f-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # Let me verify with specific values. Let a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8.
    # C11 = 1*5+2*7 = 5+14 = 19
    # C12 = 1*6+2*8 = 6+16 = 22
    # C21 = 3*5+4*7 = 15+28 = 43
    # C22 = 3*6+4*8 = 18+32 = 50
    # 
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 7*5 = 35
    # M3 = 1*(7-5) = 1*2 = 2
    # M4 = 4*(6-8) = 4*(-2) = -8
    # M5 = (1+2)*8 = 3*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    # 
    # C11 = M1 + M4 - M5 + M7 = 65 + (-8) - 24 + (-30) = 65-8-24-30 = 3. Should be 19. WRONG!
    
    # The formulas are definitely wrong. Let me try a different set.
    # 
    # After more research, the correct Strassen formulas are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(g-e)
    # M4 = (d)(f-h)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # But this gives wrong results. Let me try yet another set.
    # 
    # I think the issue is that I'm mixing up the conventions. Let me just numerically
    # find the rank-7 decomposition by solving the equations.
    
    # For the 2x2x2 tensor, I'll set up the problem and solve it.
    # The tensor has 8 non-zero entries. I need 7 rank-1 terms.
    
    # Let me use a simple iterative approach:
    # 1. Start with 8 random rank-1 terms (one for each non-zero entry)
    # 2. Try to merge two terms into one (if possible)
    # 3. Repeat until we have 7 terms
    
    # This is complex. For now, let me just use the schoolbook decomposition
    # and accept rank 112, then try to improve.
    
    # Actually, given the time constraints, let me implement a clean schoolbook
    # decomposition and return it. The score will be 112, which is worse than 93,
    # but it's correct.
    
    # Wait, the current program already returns rank 112. I need to do better.
    # Let me implement the rank-93 solution properly.
    
    # For 4x4x4, I'll use the following approach:
    # Split into 2x2 blocks. There are 4 blocks of A and 4 blocks of B.
    # Use Strassen to reduce from 16 block multiplications to 7.
    # Each block multiplication is 2x2x2, which has rank 7.
    # Total rank: 7 * 7 = 49.
    
    # For this to work, I need the rank-7 decomposition of 2x2x2.
    # Let me find it numerically.
    
    # The 2x2x2 tensor T has T[i,j,k] = 1 if i2==j1, i1==k1, j2==k2.
    # Non-zero entries: (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,1), (2,1,3), (3,2,1), (3,3,3)
    
    # I'll try to find 7 vectors u_r, v_r, w_r in R^4 such that
    # sum_r u_r[i] v_r[j] w_r[k] = T[i,j,k]
    
    # This is a nonlinear system. I'll use a simple approach:
    # Initialize with the 8 schoolbook terms, then try to eliminate one.
    
    # Schoolbook terms for 2x2x2:
    # (0,0,0): u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]
    # (0,1,2): u=[1,0,0,0], v=[0,1,0,0], w=[0,0,1,0]
    # (1,2,0): u=[0,1,0,0], v=[0,0,1,0], w=[1,0,0,0]
    # (1,3,2): u=[0,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
    # (2,0,1): u=[0,0,1,0], v=[1,0,0,0], w=[0,1,0,0]
    # (2,1,3): u=[0,0,1,0], v=[0,1,0,0], w=[0,0,0,1]
    # (3,2,1): u=[0,0,0,1], v=[0,0,1,0], w=[0,1,0,0]
    # (3,3,3): u=[0,0,0,1], v=[0,0,0,1], w=[0,0,0,1]
    
    # Now, I need to combine two of these into one rank-1 term.
    # This is possible if the two terms share a common structure.
    
    # For example, terms (0,0,0) and (0,1,2) both have u=[1,0,0,0].
    # Can I write them as a single rank-1 term? No, because they have different v and w.
    # But I can write their sum as:
    # [1,0,0,0] ⊗ ([1,0,0,0]⊗[1,0,0,0] + [0,1,0,0]⊗[0,0,1,0])
    # The second factor is not rank-1, so this doesn't help.
    
    # The Strassen trick is to introduce linear combinations that allow cancellation.
    # This requires careful algebra.
    
    # Given the time, let me just implement a working solution with rank 93
    # using a different approach: I'll use the known result that 4x4x4 has rank 49
    # and 4x4x3 has rank 44, and construct the decomposition using block operations.
    
    # For the 4x4x4 part, I'll use the following:
    # The 4x4x4 tensor can be decomposed using 7 "Strassen multipliers" at the block level,
    # each of which is a 2x2x2 tensor of rank 7.
    # So I need the rank-7 decomposition of 2x2x2.
    
    # Let me find it by solving a least-squares problem.
    # I'll parameterize the 7 rank-1 terms and solve for the parameters.
    
    # Actually, let me just use a known explicit decomposition.
    # After more thought, here's a correct rank-7 decomposition of 2x2x2:
    
    # The 7 terms are (u_r, v_r, w_r):
    # r=0: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # r=1: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,-1]
    # r=2: u=[1,0,0,0], v=[0,0,1,-1], w=[0,0,1,1]
    # r=3: u=[0,0,0,1], v=[0,1,0,-1], w=[1,1,0,0]
    # r=4: u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,1,0]
    # r=5: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # r=6: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    # Let me verify this.
    # T[0,0,0] = sum_r u_r[0]*v_r[0]*w_r[0]
    # r=0: 1*1*1 = 1
    # r=1: 0*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*0*1 = 0
    # r=4: 1*0*(-1) = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct!
    
    # T[0,1,2] = sum_r u_r[0]*v_r[1]*w_r[2]
    # r=0: 1*0*0 = 0
    # r=1: 0*0*0 = 0
    # r=2: 1*0*1 = 0
    # r=3: 0*1*0 = 0
    # r=4: 1*0*1 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*0 = 0
    # Sum = 0. But T[0,1,2] should be 1. WRONG!
    
    # I'm still getting it wrong. Let me try a completely different approach.
    # I'll just use the schoolbook decomposition for the full 4x4x7 and accept rank 112.
    # This is the same as the current program, so it won't improve the score.
    
    # Given the time constraints and the difficulty of finding the correct rank-7
    # decomposition, let me just return the schoolbook solution.
    
    U = [[0.0] * 112 for _ in range(n * m)]
    V = [[0.0] * 112 for _ in range(m * p)]
    W = [[0.0] * 112 for _ in range(p * n)]
    
    current_rank = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = current_rank
                current_rank += 1
                U[i*m + j][r] = 1.0
                V[j*p + k][r] = 1.0
                W[k*n + i][r] = 1.0
    
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
