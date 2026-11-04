"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    # Strategy: Use Strassen for 2x2x2 blocks and optimize the remainder.
    # 
    # A is 2x4, B is 4x7, C is 2x7.
    # Split A into two 2x2 column blocks: A_L (cols 0-1), A_R (cols 2-3)
    # Split B into row blocks: B_T (rows 0-1), B_B (rows 2-3)
    # C = A_L * B_T + A_R * B_B
    #
    # For A_L * B_T (2x2 * 2x7):
    #   Split B_T into column blocks: B_T1 (cols 0-1), B_T2 (cols 2-3), B_T3 (cols 4-5), B_T4 (col 6)
    #   A_L * B_T1: 2x2x2 -> rank 7 (Strassen)
    #   A_L * B_T2: 2x2x2 -> rank 7 (Strassen)
    #   A_L * B_T3: 2x2x2 -> rank 7 (Strassen)
    #   A_L * B_T4: 2x2x1 -> rank 4 (schoolbook)
    #   Subtotal: 25
    #
    # For A_R * B_B (2x2 * 2x7): similarly 25
    #
    # Total: 50
    #
    # To improve: Can we do better on the 2x2x1 parts?
    # Actually, 2x2x1 multiplication has rank exactly 4 (it's just 4 independent products).
    #
    # Alternative: Use a 2x2x3 block for the last 3 columns.
    # 2x2x3 multiplication: rank? 
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=0,1; k=0,1,2
    # That's 6 output entries, each is a bilinear form in 4 variables.
    # The tensor has 6*4*4 = 96 entries? No, it's a 2x2x3 tensor with 12 entries.
    # Rank of 2x2x3 matrix multiplication: We can use 3 Strassen-like terms?
    # Actually, 2x2x3: C[:,0:3] = A * B[:,0:3]
    # We can write this as: for each column k, C[:,k] = A * B[:,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # 
    # For 3 columns, that's 12 schoolbook terms.
    # Can we do better? Use Strassen on 2x2x2 for first 2 cols (7 terms) + 4 for last col = 11.
    # Or: group all 3 columns together.
    #
    # Let me think about 2x2x3 more carefully.
    # T is a 2x2x3 tensor. We want min rank.
    # The tensor entries: T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor.
    # For 2x2x3 multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = delta (it's 1 for all valid i,j,k since each product A[i,j]*B[j,k] appears)
    # Wait, the tensor is T[i,j,k] where the trilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # No, the standard form: sum T[i,j,k] * A[i,j] * B[j,k] = C[i,k] (with summation implied)
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (since each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k])
    # Wait no. The tensor is defined as: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C'[i,k] where C' is dual.
    # For matrix multiplication: T[i,j,k] = 1 for all valid indices (it's the identity tensor in the appropriate sense).
    #
    # For 2x2x3: T has shape (2,2,3) with all entries = 1.
    # Rank of the all-ones tensor of shape (a,b,c): it's min(a,b,c) if we can factor it.
    # Actually, the all-ones tensor T[i,j,k]=1 can be written as sum_{l=1}^{min(a,b,c)} u_l[i]*v_l[j]*w_l[k]
    # where u_l, v_l, w_l are all-ones vectors. But we need exactly min(a,b,c) terms?
    # No: T[i,j,k] = sum_l u_l[i] * v_l[j] * w_l[k]
    # If u_l = v_l = w_l = e_l (standard basis), then sum_l e_l[i]*e_l[j]*e_l[k] = delta_{i,j} delta_{j,k}
    # That's not what we want.
    #
    # We want T[i,j,k] = 1 for all i,j,k.
    # T[i,j,k] = sum_l (1/sqrt(min)) * 1 * 1 * 1 would give min(a,b,c)/min(a,b,c) = 1? No.
    # If we use rank r: T[i,j,k] = sum_{l=1}^r u_l[i] * v_l[j] * w_l[k]
    # For this to equal 1 for all i,j,k, we need the outer products to sum to the all-ones tensor.
    # The all-ones tensor has rank 1! Because T[i,j,k] = 1*1*1 = (all-ones vector in i) ⊗ (all-ones in j) ⊗ (all-ones in k)
    # So rank = 1!
    #
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    #
    # The matrix multiplication tensor for (n,m,p) is:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # And the relation is: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * (dual)
    # Actually the standard way: the tensor T has T[i,j,k] = 1 if the indices are "compatible" for the product.
    # For matrix multiplication, T[i,j,k] = 1 for ALL valid (i,j,k) because every A[i,j]*B[j,k] term appears in exactly one C[i,k].
    #
    # Hmm, but that would mean the tensor is just the all-ones tensor, which has rank 1. That's clearly wrong.
    #
    # I think I'm confusing the tensor with its "flattened" version. Let me be more careful.
    #
    # The matrix multiplication tensor M_{n,m,p} is a trilinear form:
    # M(A,B,C) = sum_{i,j,k} A[i,j] * B[j,k] * C'[i,k]  where C' is the "dual" or we identify C with C'
    # But actually, the standard definition: the tensor has components T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This means: sum_{i,j,k} delta_{i,k} * ... no.
    #
    # Let me use the correct definition. The multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k  ... no that's not right either.
    #
    # The correct tensor: T is in U ⊗ V ⊗ W^* where U = R^{n*m}, V = R^{m*p}, W = R^{n*p}
    # T(u,v) = the bilinear map (A,B) -> C = AB
    # In components: T = sum_{i,j,k} C^{i,k}_{j} * (e_i ⊗ e_j)_U ⊗ (e_j ⊗ e_k)_V ⊗ (e_i ⊗ e_k)^*_W
    # where C^{i,k}_j = 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
    #
    # So T = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)^*
    #
    # In the flattened indices:
    # U index: (i,j) -> i*m + j
    # V index: (j,k) -> j*p + k  
    # W index: (i,k) -> i*p + k  (but the task says W[k*n+i,r], so W index: k*n + i)
    #
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    #
    # So the tensor has exactly n*m*p = 2*4*7 = 56 non-zero entries, each equal to 1.
    #
    # Now, the rank is the minimum number of rank-1 tensors that sum to T.
    #
    # OK so my original analysis was correct. The all-ones argument was wrong because the tensor
    # isn't all-ones in the flattened space; it's a specific pattern.
    #
    # Let me just implement the rank-50 solution correctly and verify it.

    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Standard Strassen for 2x2x2:
        # C[0,0] = M1 - M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 - M6
        
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 2x2 blocks of B in columns: (0,1), (2,3), (4,5)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen(0, j0, k0)
    
    # Single column k=6: 4 terms for each of the 2 A-blocks = 8 terms
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1.0
            V[j*p + 6][col] = 1.0
            W[6*n + i][col] = 1.0
            col += 1
    
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
