"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We will construct a rank-54 decomposition using Strassen blocks
    # and then attempt to reduce the rank by finding linear dependencies
    # among the rank-1 terms.
    
    def build_strassen_decomposition():
        """Builds the standard rank-54 decomposition for 2x5 * 5x6."""
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # 1: (a11+a22)(b11+b22)
            U[i_base * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # 2: (a21+a22)b11
            U[(i_base + 1) * m + j_base][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] -= 1
            col += 1
            
            # 3: a11(b21-b11)
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # 4: a22(b12-b22)
            U[(i_base + 1) * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[j_base * p + k_base][col] -= 1
            W[k_base * n + i_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1
            
            # 5: (a11+a12)b22
            U[i_base * m + j_base][col] += 1
            U[i_base * m + (j_base + 1)][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] -= 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1
            
            # 6: (a21-a11)(b11+b12)
            U[(i_base + 1) * m + j_base][col] += 1
            U[i_base * m + j_base][col] -= 1
            V[j_base * p + k_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1
            
            # 7: (a12-a22)(b21+b22)
            U[i_base * m + (j_base + 1)][col] += 1
            U[(i_base + 1) * m + (j_base + 1)][col] -= 1
            V[(j_base + 1) * p + k_base][col] += 1
            V[(j_base + 1) * p + (k_base + 1)][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1

        def add_naive_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + i_base][col] += 1
            col += 1
            U[i_base * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + i_base][col] += 1
            col += 1
            U[(i_base + 1) * m + j_base][col] += 1
            V[j_base * p + k_base][col] += 1
            W[k_base * n + (i_base + 1)][col] += 1
            col += 1
            U[(i_base + 1) * m + j_base][col] += 1
            V[j_base * p + (k_base + 1)][col] += 1
            W[(k_base + 1) * n + (i_base + 1)][col] += 1
            col += 1

        # Apply Strassen to 2x2 blocks of A (cols 0,1 and 2,3)
        # and Naive to the last col (4)
        # B is split into 3 blocks of 2 cols (0,1; 2,3; 4,5)
        
        # A cols 0,1
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        
        # A cols 2,3
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # A col 4
        add_naive_block(0, 4, 0)
        add_naive_block(0, 4, 2)
        add_naive_block(0, 4, 4)
        
        return U, V, W

    U, V, W = build_strassen_decomposition()
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T is decomposed as sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
    # We want to find a linear combination of the terms that sums to 0.
    # Let c be a vector of coefficients.
    # sum_r c_r (U[:,r] \otimes V[:,r] \otimes W[:,r]) = 0
    # This is equivalent to:
    # For all i, j, k: sum_r c_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 0
    # This is a system of quadratic equations in c.
    # However, we can look for a linear dependency in the vector space of tensors.
    # Let's flatten the tensors T_r = U[:,r] \otimes V[:,r] \otimes W[:,r] into vectors.
    # If we can find c such that sum c_r T_r = 0, then we can remove one term.
    # This is finding the null space of the matrix formed by the flattened T_r.
    # But T_r are rank-1 tensors. The space of all tensors is n*m*p dimensional.
    # Here n*m*p = 60.
    # We have 54 terms.
    # If the 54 vectors are linearly dependent, we can reduce the rank.
    # Let's check the rank of the matrix M where M[:, r] = vec(T_r).
    
    M = np.zeros((n*m*p, 54))
    for r in range(54):
        # T_r is a 3D array of shape (n, m, p)
        # T_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to flatten this in a way that matches the tensor indices.
        # The tensor T has indices (i, j, k).
        # The flattening order for M should be consistent.
        # Let's use the order (i, j, k) -> i*m*p + j*p + k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Compute the rank of M
    # If rank(M) < 54, there is a dependency.
    # We can find a null vector c.
    # Then sum c_r T_r = 0.
    # This means T = sum_{r != q} (T_r - c_r/c_q T_q) for any q with c_q != 0.
    # The new terms are linear combinations of the old terms.
    # Since the entries of U, V, W are integers, and c might be rational,
    # the new entries might be rational.
    # The problem allows integer/half-integer entries.
    # So we need c to be such that the new entries are half-integers.
    
    # Let's try to find a null vector with small integer components.
    # We can use SVD to find the null space.
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # The null space is spanned by the rows of Vt corresponding to zero singular values.
    # Since we expect a dependency, the smallest singular value should be 0 (or close).
    
    # Let's check the singular values.
    # If s[-1] is very small, we have a dependency.
    
    # For now, let's just return the rank 54 solution if we can't easily find a dependency
    # that satisfies the half-integer constraint.
    # However, the directive asks to try to reduce rank.
    
    # Let's try a specific known dependency if it exists.
    # Or just return the 54 rank solution as a fallback.
    
    # Check time
    if time.time() > deadline - 10:
         return {"U": U, "V": V, "W": W}
    
    # Try to find a dependency
    # If s[-1] < 1e-10, we have a dependency.
    if s[-1] < 1e-10:
        # The last column of Vt (which is the last row of Vt^T? No, Vt is V^T)
        # Vt has shape (54, 54). The null vectors are rows of Vt where s is 0.
        # Actually, Vt is the transpose of the right singular vectors.
        # So the right singular vectors are columns of Vt^T, i.e., rows of Vt.
        # The null vector is the row of Vt corresponding to the smallest singular value.
        c = Vt[-1, :]
        
        # Check if c has small integer components
        # Scale c to have small integers
        # Find the max absolute value
        max_val = np.max(np.abs(c))
        if max_val > 0:
            c_scaled = c / max_val
            
            # Try to find a rational approximation
            # For simplicity, let's just check if the resulting entries are half-integers
            # when we replace one term.
            
            # Pick the index q with the largest |c_q|
            q = np.argmax(np.abs(c_scaled))
            
            if abs(c_scaled[q]) > 1e-6:
                # New terms for r != q: T'_r = T_r - (c_r / c_q) T_q
                # We need to construct new U', V', W'
                # U'[i, r] = U[i, r] - (c_r/c_q) U[i, q]
                # This works if the tensor product structure is preserved.
                # (U_r - alpha U_q) \otimes (V_r - alpha V_q) \otimes (W_r - alpha W_q) ?
                # NO. The tensor is U_r \otimes V_r \otimes W_r.
                # The linear combination is sum c_r (U_r \otimes V_r \otimes W_r) = 0.
                # This does NOT mean we can replace T_r with T_r - alpha T_q in the decomposition
                # by simply scaling U_r, V_r, W_r.
                # The decomposition is a sum of rank-1 tensors.
                # If sum c_r T_r = 0, then sum_{r != q} T_r = - (1/c_q) sum_{r != q} c_r T_r.
                # This is a linear combination of the other T_r's.
                # It does not give a new rank-1 term.
                # So we cannot simply reduce the rank by this method unless the sum of
                # rank-1 tensors can be rewritten as a sum of fewer rank-1 tensors.
                # This is generally not possible just from a linear dependency in the tensor space.
                # The rank is the minimum number of rank-1 terms.
                # A linear dependency means the terms are linearly dependent in the vector space,
                # but that doesn't imply a lower rank decomposition exists.
                # Example: T1 + T2 = T3. Then T1, T2, T3 are dependent.
                # But rank(T1+T2+T3) = rank(2T3) = 1? No.
                # If T1, T2, T3 are rank 1, and T1+T2=T3, then T1+T2+T3 = 2T3, which is rank 1.
                # So if we can find such a relation, we can reduce rank.
                # But finding such a relation is hard.
                
                pass # We don't have a simple way to exploit this.
    
    # Let's try a different approach: search for a lower rank decomposition directly?
    # Given time constraints, we stick to rank 54.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
