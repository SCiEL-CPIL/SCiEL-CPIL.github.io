"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
Strategy: Start with schoolbook (rank 40) and iteratively reduce rank by 
exploiting linear dependencies in the tensor vector space.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40
    
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin before 120s limit

    def time_left():
        return DEADLINE - time.time()

    def build_matrices(terms):
        """
        terms: list of (u_dict, v_dict, w_dict)
        u_dict: {(i, j): coeff}
        v_dict: {(j, k): coeff}
        w_dict: {(i, k): coeff}
        Returns U, V, W matrices.
        """
        r = len(terms)
        U = np.zeros((N_A, r), dtype=float)
        V = np.zeros((N_B, r), dtype=float)
        W = np.zeros((N_C, r), dtype=float)
        for c, (ud, vd, wd) in enumerate(terms):
            for (i, j), coeff in ud.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in vd.items():
                V[j * p + k, c] = coeff
            for (i, k), coeff in wd.items():
                W[k * n + i, c] = coeff
        return U, V, W

    def tensor_vecs(U, V, W):
        """
        Compute the tensor vectors for each rank-1 term.
        Returns a matrix T of shape (r, T_DIM) where T[c, idx] is the 
        coefficient of the tensor for term c.
        """
        r = U.shape[1]
        T = np.zeros((r, T_DIM), dtype=float)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        b = vc[j * p + k]
                        if b == 0:
                            continue
                        d = wc[k * n + i]
                        if d == 0:
                            continue
                        T[c, i * (m * p) + j * p + k] = a * b * d
        return T

    def verify(U, V, W, tol=1e-7):
        """
        Check if the decomposition exactly reproduces the multiplication tensor.
        The target tensor has 1s at positions (i, j, k) where C_ik = sum A_ij B_jk.
        Since A and B are identity-like in the tensor definition, the target is 
        all ones for the valid indices? No, the target tensor T_target is defined 
        by T_target[i, j, k] = 1 if we are computing C_ik from A_ij and B_jk.
        Wait, the tensor for matrix multiplication M_{n,m,p} has entries 
        M[i, j, k, i', j', k']? No, the rank-1 decomposition is for the 
        trilinear form or the tensor itself.
        
        The standard definition: 
        C_{ik} = sum_j A_{ij} B_{jk}
        This is a bilinear map. The tensor is in V_A^* \otimes V_B^* \otimes V_C.
        The entries of the tensor are 1 at (i, j, k) for the term A_ij B_jk C_ik?
        No, the tensor T is defined such that T(A, B, C) = sum T_{ijk} A_{ij} B_{jk} C_{ik}.
        For matrix multiplication, T_{ijk} = 1 for all i, j, k?
        Let's check: sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
        If C = AB, then C_{ik} = sum_j A_{ij} B_{jk}.
        So sum_{i,j,k} A_{ij} B_{jk} (AB)_{ik} = sum_{i,k} (AB)_{ik}^2? No.
        
        The correct tensor for matrix multiplication (as a bilinear complexity problem) 
        is often represented by the structure tensor.
        The rank of the tensor M_{n,m,p} is the minimum r such that 
        M = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
        Here M is a 3-tensor of size n*m x m*p x p*n? No.
        The tensor has dimensions (n*m, m*p, p*n) ? No.
        
        Let's look at the interface:
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        
        The condition is: 
        sum_{r=1}^R U[:, r] \otimes V[:, r] \otimes W[:, r] = T_target
        
        What is T_target?
        The matrix multiplication tensor T_{M} is defined by:
        T_{M}[(i, j), (j, k), (i, k)] = 1
        Note the middle index j must match.
        So, T_target is a tensor of size (n*m) x (m*p) x (p*n).
        The entry at index (a, b, c) where a=(i,j), b=(j,k), c=(i,k) is 1 if the indices are consistent, else 0.
        Specifically, for a fixed j, the term A_{ij} B_{jk} contributes to C_{ik}.
        So the tensor has 1s at:
        Index 1: i * m + j
        Index 2: j * p + k
        Index 3: k * n + i  (Note the order in W is k*n+i)
        
        Let's verify the target tensor construction.
        """
        T_target = np.zeros(T_DIM, dtype=float)
        # The tensor is flattened.
        # Index in U: i * m + j
        # Index in V: j * p + k
        # Index in W: k * n + i
        # The tensor T is of size n*m * m*p * p*n ? No, the decomposition is sum of rank-1 tensors.
        # The tensor T_target is an element of (R^{nm}) \otimes (R^{mp}) \otimes (R^{pn}).
        # We represent it as a matrix of shape (r, T_DIM) where T_DIM = nm * mp * pn? 
        # No, T_DIM in the code is n*m*p = 40.
        # This implies the tensor is being represented in a different way or I am misinterpreting T_DIM.
        
        # Let's re-read the code's tensor_vecs.
        # T[c, i * (m * p) + j * p + k]
        # This index is i * (mp) + j * p + k.
        # This looks like a flattening of a 3D array of size n x m x p.
        # So the tensor is of size n*m*p.
        # The entry T[i, j, k] corresponds to the coefficient of A_ij B_jk C_ik?
        # If T is of size n*m*p, then the decomposition is:
        # T_target[i, j, k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # What is T_target[i, j, k]?
        # It must be 1 for all i, j, k?
        # Let's check: 
        # If T_target[i, j, k] = 1, then 
        # 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This would imply that the product of the matrices reconstructs the all-ones tensor?
        # No, that's not matrix multiplication.
        
        # Let's look at the standard definition of the matrix multiplication tensor M_{n,m,p}.
        # It is a tensor in V_A \otimes V_B \otimes V_C where V_A = R^{nm}, V_B = R^{mp}, V_C = R^{pn}.
        # The entries are M_{(i,j), (j,k), (i,k)} = 1.
        # If we flatten this tensor into a vector of length nm * mp * pn, it would be huge.
        # But the code uses T_DIM = n*m*p = 40.
        # This suggests the tensor is NOT the full outer product structure, but rather 
        # the tensor is defined on the indices (i, j, k) directly?
        
        # If the tensor is defined on (i, j, k) with size n*m*p, then the factors U, V, W 
        # must map to these indices.
        # U has size n*m. V has size m*p. W has size p*n.
        # This doesn't match a tensor of size n*m*p directly unless there's a specific mapping.
        
        # Let's look at the index in tensor_vecs again:
        # idx = i * (m * p) + j * p + k
        # This is a linear index for a 3D array of shape (n, m, p).
        # So the target tensor is a 3D array T[i, j, k].
        # And the decomposition is:
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # What is T[i, j, k] for matrix multiplication?
        # Usually, the complexity is about the bilinear map (A, B) -> C.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # This can be viewed as a tensor T such that C = T(A, B).
        # But here we have a symmetric-looking decomposition with U, V, W.
        # This looks like the "structure tensor" or "multiplication tensor" where we are 
        # decomposing the identity map on the space of matrices?
        
        # Actually, if T[i, j, k] = 1 for all i, j, k, then:
        # 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is a very strong constraint.
        
        # Let's test with n=1, m=1, p=1.
        # U[0, r] V[0, r] W[0, r] sums to 1.
        # Rank 1 is sufficient. U=1, V=1, W=1.
        
        # Let's test n=2, m=1, p=1.
        # A is 2x1, B is 1x1, C is 1x2.
        # C_0 = A_0 B_0, C_1 = A_1 B_0.
        # T[i, 0, 0] = 1.
        # T[0, 0, 0] = sum U[0, r] V[0, r] W[0, r] = 1
        # T[1, 0, 0] = sum U[1, r] V[0, r] W[1, r] = 1
        # Note W index is k*n+i. k=0. W[0, r] and W[1, r].
        # This seems consistent.
        
        # So the target tensor T_target[i, j, k] is 1 for all valid i, j, k.
        # Let's verify this interpretation.
        # If T_target is all ones, then we are looking for a decomposition of the all-ones tensor?
        # No, the indices are coupled.
        # U index: i*m+j.
        # V index: j*p+k.
        # W index: k*n+i.
        
        # The sum is over r.
        # For a fixed i, j, k:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        
        # This is exactly the condition for the "tensor rank" of the matrix multiplication 
        # if we consider the tensor defined by these index couplings.
        # This is the standard definition of the rank of the matrix multiplication tensor M_{n,m,p}.
        # The tensor M_{n,m,p} has entries 1 at (i, j, k) in the "coupled" space.
        
        return np.allclose(T.sum(axis=0), np.ones(T_DIM), atol=tol)

    # Initialize with Schoolbook (Rank 40)
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((
                    {(i, j): 1}, 
                    {(j, k): 1}, 
                    {(i, k): 1}
                ))
    
    U, V, W = build_matrices(terms)
    
    # Verify initial
    if not verify(U, V, W):
        # Fallback: Return schoolbook (should be valid)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Reduction Loop
    max_iters = 100
    for _ in range(max_iters):
        if time_left() < 10.0:
            break
            
        r = U.shape[1]
        if r <= 1:
            break
            
        T = tensor_vecs(U, V, W)
        
        # Find null space of T
        # T is (r, T_DIM). We want z such that T @ z = 0.
        # SVD: T = U_svd S Vt
        # Null space is Vt[rank:, :]
        try:
            _, svals, Vt = np.linalg.svd(T, full_matrices=False)
        except np.linalg.LinAlgError:
            break
            
        tol = 1e-9 * (svals[0] if svals.size > 0 else 1.0)
        rank_T = int(np.sum(svals > tol))
        
        if rank_T >= r:
            # No linear dependencies
            break
            
        # Null space vectors
        null_space = Vt[rank_T:, :]
        
        found_reduction = False
        for nv in null_space:
            if time_left() < 10.0:
                break
                
            # We want to scale columns by (1 + epsilon * nv)
            # and set one column to 0.
            # Let's try to zero out the column with the largest |nv| component.
            idx = int(np.argmax(np.abs(nv)))
            if abs(nv[idx]) < 1e-12:
                continue
                
            # New scales: s_c = 1 + nv[c] / nv[idx]
            # For c = idx, s_idx = 1 + 1 = 2? No.
            # We want sum z_c T_c = 0.
            # If we replace U_c with U_c * (1 + epsilon z_c), the new tensor sum is
            # sum (1 + epsilon z_c) T_c = sum T_c + epsilon sum z_c T_c = T_target.
            # So any epsilon works for validity.
            # To drop a term, we need 1 + epsilon z_c = 0 for some c.
            # epsilon = -1 / z_c.
            # Let's pick c = idx. epsilon = -1 / nv[idx].
            # Then s_c = 1 - nv[c] / nv[idx].
            
            new_U = U.copy()
            new_V = V.copy()
            new_W = W.copy()
            
            for c in range(r):
                s = 1.0 - nv[c] / nv[idx]
                if abs(s) < 1e-12:
                    # This term is dropped
                    new_U[:, c] = 0
                    new_V[:, c] = 0
                    new_W[:, c] = 0
                else:
                    new_U[:, c] *= s
                    new_V[:, c] *= s
                    new_W[:, c] *= s
                    
            # Remove the dropped column
            keep = [c for c in range(r) if c != idx]
            new_U = new_U[:, keep]
            new_V = new_V[:, keep]
            new_W = new_W[:, keep]
            
            if verify(new_U, new_V, new_W):
                U, V, W = new_U, new_V, new_W
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
                found_reduction = True
                break
                
        if not found_reduction:
            # If no reduction found in this null space, maybe try a different pivot?
            # Or break if we can't find any.
            # For now, break to avoid infinite loop if no valid reduction.
            break

    # Final verification of best
    if not verify(best_U, best_V, best_W):
        # Revert to schoolbook if best is invalid (should not happen)
        best_U, best_V, best_W = U, V, W
        if not verify(best_U, best_V, best_W):
             terms = []
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         terms.append((
                             {(i, j): 1}, 
                             {(j, k): 1}, 
                             {(i, k): 1}
                         ))
             best_U, best_V, best_W = build_matrices(terms)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
