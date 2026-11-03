import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # Precompute the target tensor indices
    # T[i, j, k] = 1 for all i, j, k in {0, 1, 2}
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))
    
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        Checks that sum_r U[a,r]*V[b,r]*W[c,r] is 1 for ones_indices and 0 for zero_indices.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # --- Step 1: Construct Rank-26 Solution ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11) -> C00, C11
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    # r=1: (A10+A11)B00 -> C10, -C11
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    # r=2: A00(B01-B11) -> C01, C11
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    # r=3: A11(B10-B00) -> C00, C10
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    # r=4: (A00+A01)B11 -> -C00, C01
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    # r=5: (A10-A00)(B00+B01) -> C11
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    # r=6: (A01-A11)(B10+B11) -> C00
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms (indices involving 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Verify Rank 26
    if not verify(U26, V26, W26):
        # Fallback to Schoolbook Rank 27 if construction is flawed (should not happen)
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26

    # --- Step 2: Try to reduce rank to 25 via linear dependency ---
    # Form the 26 term-tensors T_r of size 27x1
    T = np.zeros((27, 26))
    for r in range(26):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = idx_u(i, j)
                    b = idx_v(j, k)
                    c = idx_w(k, i)
                    T[a, r] = U26[a, r] * V26[b, r] * W26[c, r]
    
    # Check for linear dependencies
    # If the rank of T is < 26, there is a dependency.
    rank_T = np.linalg.matrix_rank(T, tol=1e-9)
    
    if rank_T < 26:
        # Find null space vector
        # Use SVD to find the null space
        U_svd, s, Vt = np.linalg.svd(T)
        # The last row of Vt corresponds to the null space if s[-1] is close to 0
        # Actually, Vt is (26, 26). The null space vector is the last column of Vt.T (or last row of Vt)
        # Let's find the vector v such that T @ v = 0
        # We can use the smallest singular value
        if s[-1] < 1e-7:
            v = Vt[-1, :] # This is a row vector of length 26
            # v is the coefficient vector for the dependency: sum_r v[r] T_r = 0
            # We can eliminate any term r where v[r] is not zero.
            # Let's pick the term with the largest |v[r]| to eliminate
            r_elim = np.argmax(np.abs(v))
            
            # Construct new matrices U25, V25, W25
            # We remove column r_elim and adjust the other columns
            # T_new = T[:, :r_elim] + T[:, r_elim+1:] + v[r_elim] * ... wait.
            # The dependency is sum_{r != r_elim} v[r] T_r + v[r_elim] T_{r_elim} = 0
            # So T_{r_elim} = -1/v[r_elim] * sum_{r != r_elim} v[r] T_r
            # This means we can replace the term r_elim with a linear combination of the others.
            # However, the decomposition is sum_r T_r.
            # If we remove T_{r_elim}, we must add the equivalent contribution from the other terms.
            # But the other terms are already in the sum.
            # The sum is S = sum_{r=0}^{25} T_r.
            # We know sum_{r=0}^{25} v[r] T_r = 0.
            # This implies a linear relationship between the terms.
            # It does NOT mean we can just drop one term.
            # It means the set of 26 vectors is linearly dependent.
            # The rank of the tensor is the minimum number of terms.
            # If the 26 terms are dependent, the rank is at most 25.
            # We need to find a basis of 25 terms that spans the same space?
            # No, the tensor is the sum of the terms.
            # If T_0 = c_1 T_1 + ... + c_25 T_25, then sum T_r = (1+c_1)T_1 + ... + (1+c_25)T_25.
            # So we can combine the coefficients.
            
            # Let's try to express T_{r_elim} as a combination of others.
            # coeff[r] = v[r] / v[r_elim] for r != r_elim
            # T_{r_elim} = - sum_{r != r_elim} (v[r] / v[r_elim]) T_r
            
            # New coefficient for term r (r != r_elim) is 1 - (v[r] / v[r_elim])
            # Wait, the original sum is sum_{all} 1 * T_r.
            # Substitute T_{r_elim}:
            # Sum = sum_{r != r_elim} 1 * T_r + ( - sum_{k != r_elim} (v[k] / v[r_elim]) T_k )
            # Sum = sum_{r != r_elim} ( 1 - v[r] / v[r_elim] ) T_r
            
            # So we can construct a new decomposition with 25 terms.
            # The new matrices U25, V25, W25 will have 25 columns.
            # For each r != r_elim:
            #   U25[:, new_idx] = U26[:, r] * (1 - v[r] / v[r_elim])
            #   V25[:, new_idx] = V26[:, r] * (1 - v[r] / v[r_elim])  <-- WRONG.
            #   The term is U_r * V_r * W_r.
            #   We cannot just scale U, V, W independently.
            #   The term is a rank-1 tensor.
            #   We are replacing a rank-1 tensor with a linear combination of other rank-1 tensors.
            #   The sum of rank-1 tensors is the tensor.
            #   The new decomposition is a sum of 25 rank-1 tensors.
            #   But the coefficients are applied to the *terms*, not the matrices.
            #   The term T_r is defined by U_r, V_r, W_r.
            #   We cannot simply scale U_r. We need to form the linear combination of the tensors.
            #   T_new = sum_{r != r_elim} c_r T_r.
            #   This is NOT a rank-1 decomposition of the new term.
            #   The rank of the tensor is the minimum number of rank-1 tensors needed.
            #   If T_{r_elim} is in the span of the others, the set of 26 tensors spans a space of dimension 25.
            #   But the tensor is the SUM of the 26 tensors.
            #   The sum is a specific vector in that space.
            #   Can the sum be written as a sum of 25 rank-1 tensors?
            #   Yes, if the 25 basis tensors can be weighted to form the sum.
            #   But the basis tensors are T_r.
            #   The sum S = sum_{r != r_elim} (1 - v[r]/v[r_elim]) T_r.
            #   This is a linear combination of 25 rank-1 tensors.
            #   So the rank is at most 25.
            #   We can construct U25, V25, W25 by taking the 25 tensors and scaling them?
            #   No, scaling a rank-1 tensor T_r = U_r V_r W_r by a scalar c gives c U_r V_r W_r.
            #   This is still rank 1.
            #   So we can just scale the matrices U_r, V_r, W_r by the scalar?
            #   If we scale U_r by c, the term becomes c U_r V_r W_r.
            #   So yes, we can scale one of the matrices (e.g., U) by the coefficient.
            
            U25 = np.zeros((nm, 25))
            V25 = np.zeros((mp, 25))
            W25 = np.zeros((pn, 25))
            
            new_idx = 0
            for r in range(26):
                if r == r_elim:
                    continue
                coeff = 1 - v[r] / v[r_elim]
                U25[:, new_idx] = U26[:, r] * coeff
                V25[:, new_idx] = V26[:, r]
                W25[:, new_idx] = W26[:, r]
                new_idx += 1
            
            if verify(U25, V25, W25):
                best_U, best_V, best_W = U25, V25, W25
                best_rank = 25

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
