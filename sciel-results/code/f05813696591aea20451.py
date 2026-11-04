import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])

    # Verify rank-28
    res = np.zeros((n, m, p))
    for c in range(rank_28):
        Uc = U28[:, c].reshape(n, m)
        Vc = V28[:, c].reshape(m, p)
        Wc = W28[:, c].reshape(p, n)
        res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
    assert np.allclose(res, 1.0, atol=1e-9)
    
    # Try to find rank 27 by checking if the 28 factor vectors are linearly dependent
    # in the tensor space. The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # If there's a relation, we can reduce rank.
    
    # Check: is the set of 28 rank-1 tensors linearly independent in the tensor space?
    # The tensor space has dimension 2*3*5 = 30, so 28 vectors can be independent.
    # But we need to check if they span a 27-dim subspace that still contains the target tensor.
    
    # Alternative approach: try a different block decomposition
    # What if we use Strassen on columns 1:3 of B instead of 0:2 and 2:4?
    
    # Try: Strassen on B cols 1:3 (j=1,2) for both k-blocks
    # A[:,0:2] * B[0:2, 1:3] -> C[0:2, 1:3]  (Strassen, rank 7)
    # A[:,0:2] * B[0:2, 0:1] -> C[0:2, 0:1]  (schoolbook, rank 4)
    # A[:,0:2] * B[0:2, 3:5] -> C[0:2, 3:5]  (schoolbook, rank 8)
    # A[:,2:3] * B[2:3, 0:1] -> C[0:2, 0:1]  (schoolbook, rank 2)
    # A[:,2:3] * B[2:3, 1:3] -> C[0:2, 1:3]  (schoolbook, rank 4)
    # A[:,2:3] * B[2:3, 3:5] -> C[0:2, 3:5]  (schoolbook, rank 4)
    # Total: 7+4+8+2+4+4 = 29 (worse)
    
    # Try: Strassen on B cols 0:2 for k=0:2, and use a combined approach for the rest
    # What if we use a rank-6 algorithm for the 2x2x2 problem?
    # No, minimal rank for 2x2x2 is 7.
    
    # Let's try: can we save rank by combining the two Strassen blocks?
    # Strassen on (i=0,1; j=0,1; k=0,1) and Strassen on (i=0,1; j=0,1; k=2,3)
    # These are independent, so no savings there.
    
    # New idea: use a different partitioning of the 3 middle dimension
    # Split j into {0}, {1}, {2} and try various combinations
    
    # Or: use the fact that for the 2x1x2 subproblem (A is 2x1, B is 1x2),
    # the rank is 4, but maybe we can share factors with adjacent blocks.
    
    # Let me try a direct optimization approach:
    # Start from rank-28, try to perturb to find a rank-27 solution
    
    # The condition for rank 27: there exist U (6x27), V (15x27), W (10x27)
    # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = target tensor
    
    # This is a system of 30 equations (one per tensor entry) with 27*31 = 837 unknowns
    # So it's very underdetermined. Let's try to find a solution.
    
    # Use the rank-28 solution as a starting point and try to zero out one factor
    # by finding a linear combination.
    
    # Actually, let's just try random rank-27 with structured initialization
    # based on the rank-28 solution.
    
    # Take rank-28 and try to merge two columns
    for c1 in range(rank_28 - 1):
        for c2 in range(c1 + 1, rank_28):
            if time.time() > deadline - 20:
                break
            # Check if columns c1 and c2 can be merged
            # i.e., if u_{c1}⊗v_{c1}⊗w_{c1} + u_{c2}⊗v_{c2}⊗w_{c2}
            # can be written as a single rank-1 tensor
            # This happens iff the three vectors are "aligned" in a specific way
            
            u1 = U28[:, c1]
            u2 = U28[:, c2]
            v1 = V28[:, c1]
            v2 = V28[:, c2]
            w1 = W28[:, c1]
            w2 = W28[:, c2]
            
            # Check if u1, u2 are linearly dependent, v1, v2 are linearly dependent, w1, w2 are linearly dependent
            # If so, we can merge them into one factor
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                continue
            if np.allclose(w1, 0) or np.allclose(w2, 0):
                continue
            
            # Check if u2 = alpha * u1
            try:
                alpha_u = np.dot(u2, u1) / np.dot(u1, u1)
                if not np.allclose(u2, alpha_u * u1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                alpha_v = np.dot(v2, v1) / np.dot(v1, v1)
                if not np.allclose(v2, alpha_v * v1, atol=1e-9):
                    continue
            except:
                continue
            
            try:
                alpha_w = np.dot(w2, w1) / np.dot(w1, w1)
                if not np.allclose(w2, alpha_w * w1, atol=1e-9):
                    continue
            except:
                continue
            
            # All three pairs are linearly dependent! We can merge.
            # New factor: (u1+u2) ⊗ (v1+v2) ⊗ (w1+w2) doesn't work directly
            # We need: u1⊗v1⊗w1 + u2⊗v2⊗w2 = (a*u1)⊗(b*v1)⊗(c*w1)
            # where a*b*c = 1 + alpha_u*alpha_v*alpha_w... this is getting complex
            
            # Simpler: if u2=au*u1, v2=av*v1, w2=aw*w1, then
            # u1⊗v1⊗w1 + u2⊗v2⊗w2 = (1 + au*av*aw) * u1⊗v1⊗w1
            # So we can replace both with a single factor (1+au*av*aw)*u1, v1, w1
            # But we need integer/half-integer entries
            
            factor = 1 + alpha_u * alpha_v * alpha_w
            if abs(factor) > 1e-9:
                # Check if the merged factor has nice entries
                new_u = (1 + alpha_u) * u1  # This isn't right either
                # Actually: u1⊗v1⊗w1 + au*u1⊗av*v1⊗aw*w1 = (1+au*av*aw) * u1⊗v1⊗w1
                # So the merged factor is just (1+au*av*aw)*u1, v1, w1
                # But this changes the scale. We need:
                # U_new = (1+au*av*aw)*u1, V_new = v1, W_new = w1
                # OR any distribution of the scalar factor
                # Let's just check if this gives integer/half-integer entries
                
                merged_U = (1 + alpha_u * alpha_v * alpha_w) * u1
                if np.allclose(merged_U, np.round(merged_U * 2) / 2, atol=1e-9):
                    # Found a merge!
                    new_rank = rank_28 - 1
                    U_new = np.zeros((n * m, new_rank))
                    V_new = np.zeros((m * p, new_rank))
                    W_new = np.zeros((p * n, new_rank))
                    
                    new_col = 0
                    for c in range(rank_28):
                        if c == c2:
                            continue
                        if c == c1:
                            U_new[:, new_col] = merged_U
                            V_new[:, new_col] = v1
                            W_new[:, new_col] = w1
                        else:
                            U_new[:, new_col] = U28[:, c]
                            V_new[:, new_col] = V28[:, c]
                            W_new[:, new_col] = W28[:, c]
                        new_col += 1
                    
                    # Verify
                    res_check = np.zeros((n, m, p))
                    for c in range(new_rank):
                        Uc = U_new[:, c].reshape(n, m)
                        Vc = V_new[:, c].reshape(m, p)
                        Wc = W_new[:, c].reshape(p, n)
                        res_check += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
                    
                    if np.allclose(res_check, 1.0, atol=1e-9):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If no merge found, return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
