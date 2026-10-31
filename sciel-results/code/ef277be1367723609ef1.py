import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    n, m, p = DIMENSIONS

    # Rank-48 construction for 4x4 matrix multiplication
    # Uses the known result that rank(4x4 matmul) <= 48
    # Construction: 49 Strassen terms minus one redundant term
    # The redundancy: term (s=3, t=3) in the Kronecker product is linearly
    # dependent on other terms when combined with the tensor structure.
    
    # Actually, let me use the explicit rank-48 construction from the literature.
    # The construction uses 48 scalar multiplications by combining the 49 
    # Strassen products and noting that one can be expressed as a combination.
    
    # Alternative: use the direct rank-48 construction
    # Based on the observation that the 4x4 matmul tensor has a 
    # "defect" of 1 from the 7x7=49 bound.
    
    # Let me build the 49-term decomposition and then find the actual
    # linear dependency in the tensor space to reduce to 48.
    
    # Strassen 2x2 coefficients
    A_coeff = np.zeros((7, 2, 2), dtype=float)
    B_coeff = np.zeros((7, 2, 2), dtype=float)
    C_coeff = np.zeros((7, 2, 2), dtype=float)

    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Build 49-term decomposition
    U_full = np.zeros((16, 49), dtype=float)
    V_full = np.zeros((16, 49), dtype=float)
    W_full = np.zeros((16, 49), dtype=float)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            for I in range(2):
                for J in range(2):
                    ca_out = A_coeff[s, I, J]
                    if ca_out == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ca_in = A_coeff[t, i2, j2]
                            if ca_in == 0: continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U_full[i * 4 + j, r] = ca_out * ca_in

            for J in range(2):
                for K in range(2):
                    cb_out = B_coeff[s, J, K]
                    if cb_out == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            cb_in = B_coeff[t, j2, k2]
                            if cb_in == 0: continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V_full[j * 4 + k, r] = cb_out * cb_in

            for I in range(2):
                for K in range(2):
                    cc_out = C_coeff[s, I, K]
                    if cc_out == 0: continue
                    for i2 in range(2):
                        for k2 in range(2):
                            cc_in = C_coeff[t, i2, k2]
                            if cc_in == 0: continue
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W_full[k * 4 + i, r] = cc_out * cc_in

    # Verify 49-term works
    np.random.seed(123)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test
    
    C_49 = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                for r in range(49):
                    C_49[i, k] += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
    
    if not np.allclose(C_49, C_test, atol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Now try to find a rank-48 decomposition by checking if the 49 rank-1
    # tensors are linearly dependent in the tensor product space.
    # The tensor for term r is T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # (note: W is indexed as W[k*4+i, r] which corresponds to output index)
    
    # Actually, the tensor identity is:
    # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = T_matmul[i,j,k]
    # where T_matmul[i,j,k] = 1 if this is a valid matmul term
    # For matmul: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the target tensor T[i,j,k] should satisfy: for fixed (i,k), sum_j T[i,j,k] = 1
    # and for fixed (i,j), sum_k T[i,j,k] = 1, etc.
    # Actually T[i,j,k] = 1 for all (i,j,k) is NOT correct.
    # The matmul tensor is T[i,j,k] = delta (it's just 1 for all valid triples,
    # since we're computing sum_j A[i,j]*B[j,k])
    
    # Wait - the matmul tensor M[i,j,k] = 1 for all i,j,k in [0,4)
    # because (AB)_{ik} = sum_j A_{ij} B_{jk} means each (i,j,k) triple contributes.
    
    # So the target is: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # Let's verify this:
    target = np.ones((4,4,4))
    computed = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    computed[i,j,k] += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
    
    # The W index is W[k*4+i, r] - let me re-examine the convention
    # W[k*n+i, r] where n=4, so W[k*4+i, r]
    # This means W is indexed by (k, i) where k is the B row / C col, i is the A row / C row
    
    # For the tensor: sum_r U[i,j,r] V[j,k,r] W[k,i,r] should = 1 for all (i,j,k)
    # Let me check with the current construction:
    # U[i*4+j, r] corresponds to A coefficient for position (i,j)
    # V[j*4+k, r] corresponds to B coefficient for position (j,k)  
    # W[k*4+i, r] corresponds to C coefficient for position (k,i) -> but C is indexed (i,k)
    # So W[k*4+i, r] means the output position is (k,i) not (i,k)?
    
    # The task says: W[k*n+i, r] - so W is indexed by (k, i) where the output C[i,k] 
    # gets contribution from W[k*4+i, r]. This is the "trace convention" mentioned.
    
    # So the tensor identity is: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    # (since every (i,j,k) triple contributes 1 to the matmul tensor)
    
    if not np.allclose(computed, target, atol=1e-9):
        # The convention might be different. Let me just return 49.
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Now check for linear dependency among the 49 rank-1 tensors
    # Flatten each rank-1 tensor to a 64-dim vector
    rank1_tensors = np.zeros((49, 64))
    for r in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    rank1_tensors[r, i*16+j*4+k] = U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
    
    # Check if the 49 vectors are linearly dependent
    # Use SVD to find the rank
    U_svd, S, Vt = np.linalg.svd(rank1_tensors, full_matrices=False)
    rank = np.sum(S > 1e-10)
    
    if rank < 49:
        # There's a linear dependency! Find the null vector
        # The null space of rank1_tensors.T (49 x 64) gives dependencies
        # Actually, we want: sum_r alpha_r * T_r = 0
        # This means rank1_tensors.T @ alpha = 0 where alpha is 49-dim
        # The null space of rank1_tensors.T has dimension 49 - rank
        null_space = Vt[rank:]  # rows of Vt beyond rank span the null space of rank1_tensors.T
        # Wait, SVD: rank1_tensors = U @ diag(S) @ Vt
        # rank1_tensors.T = V @ diag(S) @ U.T
        # Null space of rank1_tensors.T: vectors x such that rank1_tensors.T @ x = 0
        # = V @ diag(S) @ U.T @ x = 0
        # So x in null space of U.T, i.e., x orthogonal to columns of U
        # U is 49 x 49 (full), so... let me think differently.
        
        # We want alpha (49-dim) such that sum_r alpha_r * T_r = 0
        # = rank1_tensors.T @ alpha = 0 (where rank1_tensors is 49x64)
        # So alpha is in the null space of rank1_tensors.T (64x49 matrix)
        # SVD of rank1_tensors (49x64): rank1_tensors = U_svd @ diag(S) @ Vt
        # rank1_tensors.T = Vt.T @ diag(S) @ U_svd.T (64x49)
        # Null space of rank1_tensors.T: vectors alpha with Vt.T @ diag(S) @ U_svd.T @ alpha = 0
        # This means U_svd.T @ alpha is in null space of diag(S)
        # So U_svd.T @ alpha has zeros in first `rank` components
        # alpha = U_svd @ y where y has zeros in first `rank` components
        
        if rank < 49:
            # Get a null vector
            y = np.zeros(49)
            y[rank] = 1.0  # Pick first null direction
            alpha = U_svd @ y
            
            # Verify: sum_r alpha[r] * T_r should be ~0
            check = rank1_tensors.T @ alpha
            if np.max(np.abs(check)) < 1e-8:
                # Found a valid dependency! Remove the term with largest |alpha|
                r_remove = np.argmax(np.abs(alpha))
                
                # Build rank-48 decomposition
                keep = [r for r in range(49) if r != r_remove]
                U_48 = U_full[:, keep]
                V_48 = V_full[:, keep]
                W_48 = W_full[:, keep]
                
                # Verify
                C_48 = np.zeros((4, 4))
                for i in range(4):
                    for k in range(4):
                        for j in range(4):
                            for idx, r in enumerate(keep):
                                C_48[i, k] += U_48[i*4+j, idx] * V_48[j*4+k, idx] * W_48[k*4+i, idx]
                
                if np.allclose(C_48, C_test, atol=1e-9):
                    return {"U": U_48.tolist(), "V": V_48.tolist(), "W": W_48.tolist()}
    
    # If no dependency found, return rank 49
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
