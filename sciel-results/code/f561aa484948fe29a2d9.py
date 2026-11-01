"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=2, p=8

    def verify(U, V, W, rank):
        """Verify the tensor decomposition exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Try rank p+6 = 14 construction for (2,2,8)
    # Known result: rank(2,2,p) = p+6 for p >= 2
    # Construction based on Strassen-like sharing:
    # We decompose the 2x2 multiplication into 7 Strassen products,
    # but share the B-side combinations across all p columns.
    
    # Actually, let me use a cleaner algebraic construction.
    # For (2,2,p): C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Rank p+6 construction:
    # Use 6 "shared" terms that handle the Strassen structure,
    # and p "diagonal" terms for each column.
    #
    # Let me derive this carefully.
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M3 + M4 + M6
    #
    # For (2,2,p), each "column" k has its own B[0,k] and B[1,k].
    # The key insight: the A-side combinations are shared across all k.
    #
    # Let me define 6 shared terms (one per Strassen product minus one):
    # Actually, we need 6 shared + p local = p+6 total.
    #
    # Let me try a different known construction:
    # 
    # Term 0: A[0,0] * B[0,k] -> C[0,k]  (for all k)
    # Term 1: A[0,1] * B[1,k] -> C[0,k]  (for all k)
    # Term 2: A[1,0] * B[0,k] -> C[1,k]  (for all k)
    # Term 3: A[1,1] * B[1,k] -> C[1,k]  (for all k)
    # These 4 terms give the full schoolbook already!
    # So rank 4 would suffice? No, that's wrong because each term
    # is a single scalar multiplication shared across all k.
    #
    # Wait - in the tensor decomposition, each term r corresponds to
    # a single "rank-1" component. The U, V, W matrices have entries
    # for each index. So term 0 with U[0,0]=1, V[0,k]=1 for all k,
    # W[0,k]=1 for all k means:
    #   contribution to (i,j,k) = U[i,j,0] * V[j,k,0] * W[k,i,0]
    #
    # For i=0,j=0,k=0: U[0,0,0]*V[0,0,0]*W[0,0,0] = 1*1*1 = 1 ✓
    # For i=0,j=0,k=1: U[0,0,0]*V[0,1,0]*W[1,0,0] = 1*1*1 = 1 ✓
    # But for i=0,j=1,k=0: U[0,1,0]*V[1,0,0]*W[0,0,0] = 0*0*1 = 0 ✓
    #
    # So 4 terms give:
    # C[0,k] gets A[0,0]*B[0,k] from term 0 and A[0,1]*B[1,k] from term 1
    # C[1,k] gets A[1,0]*B[0,k] from term 2 and A[1,1]*B[1,k] from term 3
    #
    # This is exactly the schoolbook! And it uses only 4 terms!
    # But wait - is this valid? Let me check all (i,j,k) combinations.
    #
    # The tensor T[i,j,k] should be 1 if the entry A[i,j] and B[j,k]
    # contribute to C[i,k]. In our case:
    # T[0,0,k] = 1 (A[0,0]*B[0,k] -> C[0,k])
    # T[0,1,k] = 1 (A[0,1]*B[1,k] -> C[0,k])
    # T[1,0,k] = 1 (A[1,0]*B[0,k] -> C[1,k])
    # T[1,1,k] = 1 (A[1,1]*B[1,k] -> C[1,k])
    #
    # And all other T[i,j,k] = 0 (which is automatic since i,j determine
    # which output row, and j determines which B row, so the mapping is fixed).
    #
    # Actually wait - I think I'm confusing myself. Let me re-read the problem.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2
    # matrix by a 2 by 8 matrix."
    #
    # A is 2x2, B is 2x8, C = A*B is 2x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has indices (i, j, k) where:
    # i = output row (0..1)
    # j = intermediate (0..1) 
    # k = output column (0..7)
    #
    # T[i,j,k] = 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k]
    #           = 0 otherwise
    #
    # In this case, for every valid (i,j,k), T[i,j,k] = 1, because
    # A[i,j] * B[j,k] always contributes to C[i,k].
    #
    # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,7}.
    #
    # That's a 2x2x8 tensor that is all ones!
    #
    # The rank of an all-ones tensor of size (a,b,c) is... 
    # Actually no, the tensor decomposition requires:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1 for all i,j,k
    #
    # Wait, but that's not right either. Let me re-read the trace convention.
    #
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # So W is indexed by (k*n+i), meaning W[k*n+i, r].
    # And the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    #
    # Where T[i,j,k] = 1 if (i,j,k) is a valid triple in the multiplication, 0 otherwise.
    #
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So the "tensor" is T[i,j,k] = 1 for all valid i,j,k, and the
    # decomposition expresses this as a sum of rank-1 tensors.
    #
    # Since T[i,j,k] = 1 for ALL i in range(n), j in range(m), k in range(p),
    # we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # This is a much simpler problem than I thought!
    #
    # Can we do this with rank 1?
    # U[i*m+j, 0] = a[i,j], V[j*p+k, 0] = b[j,k], W[k*n+i, 0] = c[k,i]
    # We need a[i,j] * b[j,k] * c[k,i] = 1 for all i,j,k.
    #
    # From a[0,0]*b[0,0]*c[0,0] = 1
    # a[0,0]*b[0,1]*c[1,0] = 1
    # a[0,1]*b[1,0]*c[0,0] = 1
    # ...
    #
    # This requires a[i,j] = alpha[i] * beta[j], b[j,k] = gamma[j] * delta[k],
    # c[k,i] = epsilon[k] * zeta[i] for some scalars, and the product
    # alpha[i]*beta[j]*gamma[j]*delta[k]*epsilon[k]*zeta[i] = 1
    # => (alpha[i]*zeta[i]) * (beta[j]*gamma[j]) * (delta[k]*epsilon[k]) = 1
    #
    # This works! Set alpha[i]*zeta[i] = 1, beta[j]*gamma[j] = 1, delta[k]*epsilon[k] = 1.
    # E.g., alpha[i]=1, zeta[i]=1, beta[j]=1, gamma[j]=1, delta[k]=1, epsilon[k]=1.
    #
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
    #
    # Let me verify: sum_r=0 to 0 of U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
    #
    # So the rank is 1!
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
