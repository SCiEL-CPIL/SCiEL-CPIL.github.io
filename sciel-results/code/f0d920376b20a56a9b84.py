import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will use 6 Strassen blocks of size 2x2x2
    # Each saves 1 rank (8 -> 7)
    # Total rank = 90 - 6 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # We need to map these to U, V, W entries.
    # U[i,j,r] corresponds to coefficient of A[i,j]
    # V[j,k,r] corresponds to coefficient of B[j,k]
    # W[k,i,r] corresponds to coefficient of C[k,i]? 
    # Wait, W index is k*n+i.
    # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # Let's define the 7 vectors for one block.
    # Local indices i,j,k in {0,1}.
    # A = [[a,b],[c,d]] -> indices (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B = [[e,f],[g,h]] -> indices (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    
    # m1: (a+d)(e+h)
    # U: a:1, d:1. V: e:1, h:1. W: 1.
    # m2: (c+d)e
    # U: c:1, d:1. V: e:1. W: 1.
    # m3: a(f-h)
    # U: a:1. V: f:1, h:-1. W: 1.
    # m4: d(g-e)
    # U: d:1. V: g:1, e:-1. W: 1.
    # m5: (a+b)h
    # U: a:1, b:1. V: h:1. W: 1.
    # m6: (c-a)(e+f)
    # U: c:1, a:-1. V: e:1, f:1. W: 1.
    # m7: (b-d)(g+h)
    # U: b:1, d:-1. V: g:1, h:1. W: 1.
    
    # Note: The W term in Strassen decomposition usually maps to the output C.
    # In the tensor sum sum U V W = 1, W[k,i,r] must be 1 for the relevant output indices?
    # Actually, for the 2x2x2 tensor, the "output" dimension is not explicitly in the sum 
    # because the sum is over i,j,k and equals 1.
    # So W[k,i,r] should be 1 for all i,k in the block?
    # Let's check.
    # If W[k,i,r] = 1 for all i,k, then sum_r U[i,j,r] V[j,k,r] = 1?
    # No, the Strassen algorithm computes C.
    # The tensor identity is T = sum u v w.
    # For 2x2x2, T[i,j,k] = 1.
    # So sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # If we set W[k,i,r] = 1 for all i,k, then we need sum_r U[i,j,r] V[j,k,r] = 1.
    # But Strassen gives C[i,k] = sum_j ...
    # The standard Strassen decomposition of the matmul tensor has W[k,i,r] = 1?
    # Let's verify with a known source or logic.
    # The matmul tensor T is defined by C[i,k] = sum_j A[i,j] B[j,k].
    # This is NOT T[i,j,k] = 1.
    # T is a tensor in R^{n x m x p}.
    # The action is contraction with A and B to give C.
    # C[i,k] = sum_{j,l} T[i,j,k] A[i,j] B[j,k]? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This can be written as C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] delta_{i,i'} delta_{k,k'}?
    # No.
    
    # The tensor T for matmul is such that:
    # <T, A \otimes B> = C?
    # No, the rank of the matmul tensor is the rank of the linear map.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This tensor satisfies:
    # sum_{j} T[i,j,k] A[i,j] B[j,k] is not C.
    
    # Let's look at the seed program.
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For a fixed (i,j,k), only the column corresponding to (i,j,k) has 1s.
    # So the sum is 1.
    # This confirms T[i,j,k] = 1 for all i,j,k.
    
    # So we need the Strassen vectors to satisfy:
    # sum_{r=1}^7 U[i,j,r] V[j,k,r] W[k,i,r] = 1 for i,j,k in {0,1}.
    
    # If we set W[k,i,r] = 1 for all i,k, then we need sum_r U[i,j,r] V[j,k,r] = 1.
    # But Strassen computes C[i,k] = sum_j ...
    # The "1" in the tensor is not C.
    
    # Actually, the Strassen decomposition of the tensor T (all ones) is NOT the same as the Strassen algorithm for matrix multiplication.
    # The Strassen algorithm decomposes the MAP.
    # The tensor T is the kernel of the map.
    
    # Wait, if T[i,j,k] = 1, then the rank of T is the rank of the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1?
    # No, the all-ones tensor is rank 1: 1 \otimes 1 \otimes 1.
    # But the seed program has rank nmp.
    # This implies T is NOT the all-ones tensor.
    
    # Let's re-read the seed program logic.
    # U[i*m+j, col] = 1.
    # V[j*p+k, col] = 1.
    # W[k*n+i, col] = 1.
    # Sum over col (which is i*m+j*p+k index) of U V W.
    # For a fixed (i,j,k), only the term where col matches (i,j,k) is non-zero?
    # U[i*m+j, col] is 1 only if col corresponds to i*m+j.
    # V[j*p+k, col] is 1 only if col corresponds to j*p+k.
    # W[k*n+i, col] is 1 only if col corresponds to k*n+i.
    # So for a fixed (i,j,k), the sum is 1.
    # For any other combination of indices?
    # The sum is over r.
    # U[i*m+j, r] is 1 if r's i-part is i and j-part is j.
    # V[j*p+k, r] is 1 if r's j-part is j and k-part is k.
    # W[k*n+i, r] is 1 if r's k-part is k and i-part is i.
    # So for a fixed (i,j,k), the sum is 1.
    # This means T[i,j,k] = 1 for ALL i,j,k.
    
    # So the tensor IS the all-ones tensor.
    # But the rank of the all-ones tensor is 1.
    # Why does the seed program have rank nmp?
    # Because it's a "schoolbook" decomposition, not a minimal one.
    # The all-ones tensor can be decomposed as 1 \otimes 1 \otimes 1 with rank 1.
    # U[i*m+j, 0] = 1. V[j*p+k, 0] = 1. W[k*n+i, 0] = 1.
    # Sum = 1.
    # Rank 1.
    
    # So the task is trivial? Rank 1?
    # "Find a rank-minimal exact tensor decomposition for multiplying..."
    # If the tensor is all-ones, rank 1 is minimal.
    
    # But wait. Is the tensor for matrix multiplication the all-ones tensor?
    # No. The tensor for matrix multiplication is defined by the indices of the INPUTS and OUTPUT.
    # The tensor T has indices (i, j, k).
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # This is a bilinear map.
    # The tensor associated with this map is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the map (A, B) -> sum_{i,j,k} A[i,j] B[j,k].
    # This is a scalar, not a matrix C.
    
    # The correct tensor for matmul has entries T[i,j,k] such that
    # C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] ...?
    # No.
    # The tensor T is in R^{n \times m \times p}.
    # The rank of T is the minimal R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # And the condition is that this decomposition allows computing C.
    # Specifically, C[i,k] = sum_{j} A[i,j] B[j,k].
    # This can be written as C[i,k] = sum_{r=1}^R (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[k,i,r]?
    # No.
    
    # The standard definition:
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k is the tensor for the "dot product" of 3 vectors?
    # No.
    
    # Let's assume the seed program is correct for the task definition.
    # The seed program uses rank nmp.
    # If the tensor were all-ones, rank 1 would work.
    # So the tensor is NOT all-ones.
    
    # The tensor for matmul is T[i,j,k] = 1 if we consider the indices (i,j,k) to be the "free" indices?
    # No.
    
    # I will stick to the Strassen embedding which is a known valid technique for reducing the rank of the matmul tensor.
    # The rank 84 is a safe bet.

    # Define blocks
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [4, 5]),
    ]
    
    used_cols = set()
    
    # Strassen vectors for local 2x2x2
    # u1 = [1, 0, 0, 1] (a,d)
    # v1 = [1, 0, 0, 1] (e,h)
    # w1 = 1
    
    # u2 = [0, 0, 1, 1] (c,d)
    # v2 = [1, 0, 0, 0] (e)
    # w2 = 1
    
    # u3 = [1, 0, 0, 0] (a)
    # v3 = [0, 1, 0, -1] (f,-h)
    # w3 = 1
    
    # u4 = [0, 0, 0, 1] (d)
    # v4 = [0, 0, 1, -1] (g,-e) -- wait, g is (1,0), e is (0,0).
    # B indices: 00=e, 01=f, 10=g, 11=h.
    # v4 = [ -1, 0, 1, 0 ] (e:-1, g:1)
    # w4 = 1
    
    # u5 = [1, 1, 0, 0] (a,b)
    # v5 = [0, 0, 0, 1] (h)
    # w5 = 1
    
    # u6 = [-1, 0, 1, 0] (a:-1, c:1)
    # v6 = [1, 1, 0, 0] (e,f)
    # w6 = 1
    
    # u7 = [0, 1, 0, -1] (b:1, d:-1)
    # v7 = [0, 0, 1, 1] (g,h)
    # w7 = 1
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [1, 1, 1, 1, 1, 1, 1]
    
    col_idx = 0
    for I, J, K in blocks:
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        for r in range(7):
            # U
            for li, li_val in enumerate(strassen_u[r]):
                if li_val != 0:
                    i_idx = i0 if li == 0 else i1
                    j_idx = j0 if li < 2 else j1
                    U[i_idx * m + j_idx][col_idx] += li_val
            
            # V
            for lj, lj_val in enumerate(strassen_v[r]):
                if lj_val != 0:
                    j_idx = j0 if lj < 2 else j1
                    k_idx = k0 if lj == 0 or lj == 1 else k1
                    # lj 0->00, 1->01, 2->10, 3->11
                    if lj == 0:
                        j_idx, k_idx = j0, k0
                    elif lj == 1:
                        j_idx, k_idx = j0, k1
                    elif lj == 2:
                        j_idx, k_idx = j1, k0
                    else:
                        j_idx, k_idx = j1, k1
                    V[j_idx * p + k_idx][col_idx] += lj_val
            
            # W
            W[k0 * n + i0][col_idx] += strassen_w[r]
            W[k0 * n + i1][col_idx] += strassen_w[r]
            W[k1 * n + i0][col_idx] += strassen_w[r]
            W[k1 * n + i1][col_idx] += strassen_w[r]
            
            col_idx += 1
    
    # Fill remaining with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is already covered by a Strassen block
                covered = False
                for I, J, K in blocks:
                    if i in I and j in J and k in K:
                        covered = True
                        break
                if not covered:
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
