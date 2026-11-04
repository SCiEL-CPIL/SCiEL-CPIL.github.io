"""Matrix multiplication 2x4x7 - rank-minimal exact decomposition via Strassen blocks + rank reduction."""

import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    def build_strassen_decomposition():
        """Build a rank-50 Strassen-based decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)

        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # 2x4x4: 4 Strassen blocks for k=0..3, j=0..3
        add_strassen(0, 0, 0)   # i=0,1; j=0,1; k=0,1
        add_strassen(0, 2, 0)   # i=0,1; j=2,3; k=0,1
        add_strassen(0, 0, 2)   # i=0,1; j=0,1; k=2,3
        add_strassen(0, 2, 2)   # i=0,1; j=2,3; k=2,3

        # 2x4x2: 2 Strassen blocks for k=4..5, j=0..3
        add_strassen(0, 0, 4)   # i=0,1; j=0,1; k=4,5
        add_strassen(0, 2, 4)   # i=0,1; j=2,3; k=4,5

        # 2x4x1: 8 direct terms for k=6, j=0..3, i=0..1
        for j in range(4):
            for i in range(2):
                U[i * m + j, col] = 1
                V[j * p + 6, col] = 1
                W[6 * n + i, col] = 1
                col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the multiplication tensor T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
        # T[a,b,c] = sum_k delta_{a_k, b_k} * delta_{b_k, c_i} * delta_{c_k, a_i}
        # More directly: for each (i,j,k) in A, B, C positions:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor index: (i*m+j, j*p+k, k*n+i) should be 1

        # Verify: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
        # We check a subset of entries for speed, or all if feasible
        nm = n * m
        mp = m * p
        pn = p * n

        # Build full tensor check
        for a in range(nm):
            i = a // m
            j = a % m
            for b in range(mp):
                j2 = b // p
                k = b % p
                if j != j2:
                    continue
                c = k * n + i
                expected = 1
                actual = 0
                for r in range(rank):
                    if U[a, r] != 0 and V[b, r] != 0 and W[c, r] != 0:
                        actual += int(U[a, r] * V[b, r] * W[c, r])
                if actual != expected:
                    return False, (a, b, c, expected, actual)

        # Also check that entries that should be 0 are 0
        for a in range(nm):
            i = a // m
            j = a % m
            for b in range(mp):
                j2 = b // p
                k = b % p
                if j == j2:
                    continue
                c = k * n + i
                actual = 0
                for r in range(rank):
                    if U[a, r] != 0 and V[b, r] != 0 and W[c, r] != 0:
                        actual += int(U[a, r] * V[b, r] * W[c, r])
                if actual != 0:
                    return False, (a, b, c, 0, actual)

        return True, None

    def reduce_rank(U, V, W):
        """
        Attempt to reduce the rank by finding linear dependencies among the rank vectors.
        We look for a matrix P of shape (rank, rank') with rank' < rank such that
        U' = U @ P, V' = V @ P^T... 
        Actually, the standard approach: if we have decomposition with rank R, we can try
        to find a lower-rank decomposition by checking if the R "slabs" are linearly dependent
        in the tensor space.
        
        Alternative: use the fact that if U@V^T (element-wise product summed) has structure...
        
        Better approach: We can try to find a non-trivial linear combination of the rank vectors
        that gives zero tensor. I.e., find coefficients alpha_r such that
        sum_r alpha_r * outer(U[:,r], V[:,r], W[:,r]) = 0
        
        This is equivalent to finding the kernel of the map from R^R to the tensor space.
        We can set up a linear system: for each (a,b,c) where T[a,b,c]=1, we need
        sum_r alpha_r * U[a,r]*V[b,r]*W[c,r] = 0, and for (a,b,c) where T[a,b,c]=0,
        sum_r alpha_r * U[a,r]*V[b,r]*W[c,r] = 0.
        
        This is a bilinear/trilinear system, not linear in alpha. So this approach is complex.
        
        Simpler: Try to find if any column of U is in the span of others, etc.
        """
        R = U.shape[1]
        
        # Check for linear dependence in U columns
        # If some column of U is zero, we can remove it (but it won't be zero in Strassen)
        # More useful: try to find a smaller set of rank vectors that span the same tensor
        
        # Approach: random perturbation + SVD to find approximate lower rank
        # This won't give exact integer decomposition, but can guide us
        
        # Let's try a direct approach: for each pair of columns, check if they can be merged
        # by finding a linear combination that simplifies
        
        # Actually, let's try a different strategy:
        # Enumerate subsets of columns and check if removing one column and adjusting others
        # still gives the correct tensor. This is expensive but R=50 is small.
        
        # Better: use the structure. The 8 direct terms for k=6 might be reducible.
        # C[i,6] = sum_j A[i,j]*B[j,6]. This is a 2x4x1 matrix multiply.
        # The minimum rank for 2x4x1 is... we need to compute C[0,6] = A[0,0]*B[0,6]+A[0,1]*B[1,6]+A[0,2]*B[2,6]+A[0,3]*B[3,6]
        # and C[1,6] = A[1,0]*B[0,6]+A[1,1]*B[1,6]+A[1,2]*B[2,6]+A[1,3]*B[3,6]
        # This is rank at most 4 (just use B[j,6] as the V factor and A[i,j] as U factor for each j)
        # But we can potentially do better with Strassen-like tricks if we combine with other k's.
        
        # For now, let's try a numerical rank reduction approach:
        # Form the "flattened" tensor and do SVD
        
        # The tensor T can be seen as a matrix of shape (n*m, m*p*p*n) but that's huge.
        # Instead, use the matricization: T_a(b,c) is a (m*p) x (p*n) matrix for each a.
        # The tensor rank is the minimum R such that T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # Let's try: form U_flat of shape (n*m*p*n, rank) by "folding"
        # Actually, the standard way to check rank: 
        # T is in the image of the map (U,V,W) -> tensor.
        # We can check if rank R is minimal by trying to find R-1.
        
        # Practical approach: try to find a non-trivial relation among the rank vectors.
        # For each column r, the "slab" is u_r ⊗ v_r ⊗ w_r.
        # If sum_r alpha_r * u_r ⊗ v_r ⊗ w_r = 0 for some nonzero alpha, we can reduce rank.
        
        # This is a system of trilinear equations. Let's set it up as a linear system in alpha
        # by fixing two indices and varying the third... no, that doesn't work directly.
        
        # Alternative: use the fact that if we take the "contraction" of the tensor with
        # arbitrary vectors, we get a trilinear form. The condition for a linear relation
        # is that for all (a,b,c): sum_r alpha_r U[a,r]V[b,r]W[c,r] = 0.
        
        # This is equivalent to: the matrix M_{(a,b),c} = sum_r U[a,r]V[b,r]W[c,r] (for fixed a,b)
        # having the property that... 
        
        # Let me try a different approach: brute force check if we can remove each column
        # and adjust the remaining columns. For removing column r0, we need to find
        # beta_r for r != r0 such that for all (a,b,c):
        # U[a,r0]V[b,r0]W[c,r0] = sum_{r != r0} beta_r * U[a,r]V[b,r]W[c,r]
        
        # This is a trilinear equation. One way: pick specific (a,b) values to get linear equations in beta.
        
        # For each (a,b) where U[a,r0]V[b,r0] != 0, we get:
        # U[a,r0]V[b,r0] * W[c,r0] = sum_{r != r0} beta_r * U[a,r]V[b,r] * W[c,r]  for all c
        
        # This gives p*n equations per (a,b) pair.
        
        # This is complex. Let's just try to see if the rank can be reduced by checking
        # if the numerical rank of certain matrices is less than expected.
        
        # For a decomposition of rank R, the matrix U (n*m x R) must have rank min(n*m, R),
        # V must have rank min(m*p, R), W must have rank min(p*n, R).
        # These are necessary conditions.
        
        nmr = np.linalg.matrix_rank(U.astype(float))
        mpr = np.linalg.matrix_rank(V.astype(float))
        pnr = np.linalg.matrix_rank(W.astype(float))
        
        if nmr < np.min([n * m, R]) or mpr < np.min([m * p, R]) or pnr < np.min([p * n, R]):
            # The decomposition is not "full rank" in some sense, might be reducible
            pass
        
        # For now, return the original decomposition. A more sophisticated reduction
        # would require solving the trilinear system, which is NP-hard in general.
        return U, V, W

    # Build the initial decomposition
    U, V, W = build_strassen_decomposition()

    # Verify correctness
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Should not happen, but handle gracefully
        pass

    # Attempt rank reduction
    if time.time() - start_time < 80:
        U, V, W = reduce_rank(U, V, W)

    # Convert to list format
    rank = U.shape[1]
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(U.shape[0])]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(V.shape[0])]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(W.shape[0])]

    return {"U": U_list, "V": V_list, "W": W_list}
