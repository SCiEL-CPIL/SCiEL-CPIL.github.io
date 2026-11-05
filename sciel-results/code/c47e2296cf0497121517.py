def construct():
    import numpy as np
    
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    rank = 16
    
    U = np.zeros((n * m, rank))  # rows: i*m + j
    V = np.zeros((m * p, rank))  # rows: j*p + k
    W = np.zeros((p * n, rank))  # rows: k*n + i
    
    # Use Strassen-based decomposition
    # Split middle dimension into two groups of 2
    # For each output (i,k), C[i,k] = sum_{j=0}^3 A[i,j]*B[j,k]
    # 
    # Use 16 rank-1 terms based on the following scheme:
    # For each i in {0,1,2} and each "Strassen product" index, assign terms.
    
    # Actually, let me use a direct verified construction.
    # Rank-16 for 3x4x4:
    # 
    # Term c=0..3: handle j=0,1 for all i,k using schoolbook on 3x2*2x4
    #   But that's 3*4*2 = 24 terms. Too many.
    
    # Let me use a different strategy: exploit linearity.
    # 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # Define for each i:
    #   s0 = A[i,0] + A[i,2]
    #   s1 = A[i,1] + A[i,3]  
    #   d0 = A[i,0] - A[i,2]
    #   d1 = A[i,1] - A[i,3]
    #
    # And for each k:
    #   t0 = B[0,k] + B[2,k]
    #   t1 = B[1,k] + B[3,k]
    #   u0 = B[0,k] - B[2,k]
    #   u1 = B[1,k] - B[3,k]
    #
    # Then:
    # A[i,0]*B[0,k] + A[i,2]*B[2,k] = (s0*t0 + d0*u0)/2
    # A[i,1]*B[1,k] + A[i,3]*B[3,k] = (s1*t1 + d1*u1)/2
    #
    # So C[i,k] = (s0*t0 + d0*u0 + s1*t1 + d1*u1)/2
    #
    # This uses 4 products per (i,k) pair, but the products s0*t0, d0*u0, s1*t1, d1*u1
    # are shared across k! Wait no, t0, u0, t1, u1 depend on k.
    
    # So for each i and k, we have 4 products. Total: 3*4*4 = 48. Not helpful.
    
    # The key is that s0, s1, d0, d1 depend only on i, and t0,t1,u0,u1 depend only on k.
    # So the products are:
    #   s0[i] * t0[k], d0[i] * u0[k], s1[i] * t1[k], d1[i] * u1[k]
    #
    # For fixed i, s0[i] is a scalar, t0[k] is a scalar for each k.
    # The product s0[i]*t0[k] is a rank-1 matrix in (i,k) space!
    #
    # So we have 4 rank-1 matrices in (i,k) space, each contributing to all (i,k) entries.
    # But wait, the full decomposition needs to produce C[i,k] for all i,k.
    #
    # C[i,k] = 0.5*(s0[i]*t0[k] + d0[i]*u0[k] + s1[i]*t1[k] + d1[i]*u1[k])
    #
    # This means the entire 3x4 matrix C is a sum of 4 rank-1 matrices!
    # But that can't be right because the matrix multiplication tensor isn't rank 4.
    
    # I made an error. Let me recheck:
    # A[i,0]*B[0,k] + A[i,2]*B[2,k] 
    #   = ((A[i,0]+A[i,2]) + (A[i,0]-A[i,2]))/2 * ((B[0,k]+B[2,k]) + (B[0,k]-B[2,k]))/2
    #     + ... no this doesn't work that way.
    
    # A[i,0]*B[0,k] + A[i,2]*B[2,k] 
    # Let a=A[i,0], b=A[i,2], c=B[0,k], d=B[2,k]
    # a*c + b*d = ((a+b)(c+d) + (a-b)(c-d))/2
    # Yes! So:
    # a*c + b*d = 0.5*(a+b)*(c+d) + 0.5*(a-b)*(c-d)
    
    # So C[i,k] = 0.5*(s0[i]*t0[k] + d0[i]*u0[k] + s1[i]*t1[k] + d1[i]*u1[k])
    # where s0[i]=A[i,0]+A[i,2], d0[i]=A[i,0]-A[i,2], s1[i]=A[i,1]+A[i,3], d1[i]=A[i,1]-A[i,3]
    #       t0[k]=B[0,k]+B[2,k], u0[k]=B[0,k]-B[2,k], t1[k]=B[1,k]+B[3,k], u1[k]=B[1,k]-B[3,k]
    
    # Now, s0[i]*t0[k] is a product of a function of i and a function of k.
    # In the tensor decomposition framework:
    # U row for (i, j) has entries, V row for (j, k) has entries, W row for (k, i) has entries.
    # The contribution to C[i,k] from term c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j.
    
    # For the term s0[i]*t0[k]: this depends on i and k but not on j in a simple way.
    # s0[i] = A[i,0]+A[i,2] involves j=0 and j=2.
    # t0[k] = B[0,k]+B[2,k] involves j=0 and j=2.
    
    # So the product (A[i,0]+A[i,2])*(B[0,k]+B[2,k]) expands to:
    # A[i,0]*B[0,k] + A[i,0]*B[2,k] + A[i,2]*B[0,k] + A[i,2]*B[2,k]
    
    # In the tensor decomposition, this single "product" corresponds to multiple rank-1 terms
    # because it involves multiple j values.
    
    # Specifically, (A[i,0]+A[i,2])*(B[0,k]+B[2,k]) can be written as a sum of rank-1 tensors:
    # For the j=0 part: A[i,0]*B[0,k] and A[i,0]*B[2,k]
    # For the j=2 part: A[i,2]*B[0,k] and A[i,2]*B[2,k]
    
    # So each "Strassen product" like s0[i]*t0[k] actually requires 4 rank-1 terms in the full decomposition (one for each (j_from_A, j_from_B) combination where they must match).
    
    # Wait, no. In matrix multiplication, the j index must be the same in A and B. So A[i,j]*B[j,k].
    # The product (A[i,0]+A[i,2])*(B[0,k]+B[2,k]) is NOT a valid single term in the matrix multiplication tensor because it mixes j=0 and j=2.
    
    # I think I'm overcomplicating this. Let me just use a known rank-16 construction or fall back to a smaller rank than 48.
    
    # Let me use rank 24: for each i, use Strassen on the 4 middle elements.
    # For fixed i, C[i,k] = sum_{j=0}^3 A[i,j]*B[j,k]
    # This is a 1x4 * 4x4 = 1x4 multiplication, which is just a vector-matrix product.
    # The tensor rank for 1x4x4 is 4 (each k needs its own term).
    # So total rank = 3*4 = 12? No, because B[j,k] is shared across i.
    
    # The full 3x4x4 tensor: T[i,j,k] = 1 (for the multiplication structure).
    # Rank is the minimum number of rank-1 tensors whose sum equals T.
    
    # For 1x4x4: T[j,k] structure... actually for 1xn x nxp, the rank is n*p.
    # For 3x4x4, I believe the rank is between 16 and 48.
    
    # Let me just implement rank 16 using explicit coefficients and verify numerically.
    
    # After careful thought, here's a rank-16 construction:
    # Use the following 16 terms where each term c has:
    # - A-side coefficients for each (i,j)
    # - B-side coefficients for each (j,k) 
    # - Output coefficient for each (k,i)
    
    # I'll use a construction based on splitting j into {0,1} and {2,3} and using
    # Strassen-like formulas for each half.
    
    # For j in {0,1}: C_half1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For j in {2,3}: C_half2[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Each half is a 3x2 * 2x4 multiplication.
    # The 3x2*2x4 tensor has rank... let me compute.
    # For 1x2*2x1: rank 2 (a*b + c*d needs 2 terms... actually rank 2)
    # For 3x2*2x4: I'll use schoolbook which is 3*2*4=24, but we can do better.
    
    # Actually, for 3x2*2x4, note that it's like computing 3 separate 1x2*2x4 products.
    # Each 1x2*2x4 has rank 4 (for each k, we need A[i,0]*B[0,k] + A[i,1]*B[1,k]).
    # But the B[j,k] values are shared, so we can't just multiply ranks.
    
    # Let me just use rank 16 and define explicit coefficients, then verify.
    
    # Construction: 16 terms
    # Terms 0-7: for the "even" j group {0,2}
    # Terms 8-15: for the "odd" j group {1,3}
    
    # For j in {0,2}: define new indices j' in {0,1} where j=2*j'
    # C_even[i,k] = A[i,0]*B[0,k] + A[i,2]*B[2,k]
    # This is a 3x2*2x4 problem with j' in {0,1}.
    
    # Use Strassen-like: 
    # M1 = (A[i,0]+A[i,2]) * (B[0,k]+B[2,k]) -- but this mixes j indices
    # This doesn't directly work for the tensor decomposition.
    
    # I'll just use schoolbook for each half: 3*2*4 = 24 terms total. Rank 24.
    # That's better than 48.
    
    # Let me do rank 24.
    rank = 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: for each (i,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] over j
    # should give delta[i,k] structure... actually it should reconstruct the multiplication.
    
    # Test with random matrices
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for c_idx in range(rank):
                    s += U_arr[i*m+j, c_idx] * V_arr[j*p+k, c_idx] * W_arr[k*n+i, c_idx]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
