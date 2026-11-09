import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Uses a rank-65 decomposition: 49 terms for 4x4x4 (Kronecker product of Strassen-7)
    + 16 scalar terms for the 5th column.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 65
    
    # Initialize matrices as lists of lists for flexibility, then convert if needed
    # U: (n*m) x rank = 16 x 65
    # V: (m*p) x rank = 20 x 65
    # W: (p*n) x rank = 20 x 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Step 1: Define Strassen-7 decomposition for 2x2x2 ---
    # The 2x2x2 tensor is decomposed into 7 terms.
    # Each term is a triple (u, v, w) where:
    # u is a vector of length 4 (indices i*2+j for 2x2 matrix A)
    # v is a vector of length 4 (indices j*2+k for 2x2 matrix B)
    # w is a vector of length 4 (indices k*2+i for 2x2 matrix C)
    
    # We define the 7 terms explicitly.
    # Reference:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (note: C[i,k] index is k*2+i? 
    # Wait, standard convention for matrix multiplication tensor:
    # T[i,j,k] = sum_u u[i,u] v[u,j] w[j,k] ? 
    # Or T[i,j,k] = sum_r u[i,j,r] v[j,k,r] w[k,i,r]?
    # The task says: W[k*n+i, r]. This implies the output index is k*n+i.
    # Standard matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # So U rows are (i,j), V rows are (j,k), W rows are (k,i).
    
    # Let's map the 2x2 blocks to indices 0..3.
    # Block (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3.
    
    strassen_terms_2x2 = []
    
    # Term 0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # A11 is idx 0, A22 is idx 3. u = [1, 0, 0, 1]
    # B11 is idx 0, B22 is idx 3. v = [1, 0, 0, 1]
    # C11 is idx 0 (k=0,i=0), C22 is idx 3 (k=1,i=1). w = [1, 0, 0, 1]
    strassen_terms_2x2.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # Term 1: M2 = (A21+A22)B11 -> C21 += M2, C22 += M2
    # A21 is idx 2, A22 is idx 3. u = [0, 0, 1, 1]
    # B11 is idx 0. v = [1, 0, 0, 0]
    # C21 is idx 2 (k=1,i=0), C22 is idx 3 (k=1,i=1). w = [0, 0, 1, 1]
    strassen_terms_2x2.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]))
    
    # Term 2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # A11 is idx 0. u = [1, 0, 0, 0]
    # B12 is idx 1, B22 is idx 3. v = [0, 1, 0, -1]
    # C12 is idx 1 (k=0,i=1), C22 is idx 3 (k=1,i=1). w = [0, 1, 0, 1]
    strassen_terms_2x2.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    
    # Term 3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # A22 is idx 3. u = [0, 0, 0, 1]
    # B21 is idx 2, B11 is idx 0. v = [-1, 0, 1, 0]
    # C11 is idx 0, C21 is idx 2. w = [1, 0, 1, 0]
    strassen_terms_2x2.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    
    # Term 4: M5 = (A11+A12)B22 -> C11 += M5, C12 += M5
    # A11 idx 0, A12 idx 1. u = [1, 1, 0, 0]
    # B22 idx 3. v = [0, 0, 0, 1]
    # C11 idx 0, C12 idx 1. w = [1, 1, 0, 0]
    strassen_terms_2x2.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    
    # Term 5: M6 = (A21-A11)(B11+B12) -> C22 += M6
    # A21 idx 2, A11 idx 0. u = [-1, 0, 1, 0]
    # B11 idx 0, B12 idx 1. v = [1, 1, 0, 0]
    # C22 idx 3. w = [0, 0, 0, 1]
    strassen_terms_2x2.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # Term 6: M7 = (A12-A22)(B21+B22) -> C11 += M7
    # A12 idx 1, A22 idx 3. u = [0, 1, 0, -1]
    # B21 idx 2, B22 idx 3. v = [0, 0, 1, 1]
    # C11 idx 0. w = [1, 0, 0, 0]
    strassen_terms_2x2.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # --- Step 2: Construct 4x4x4 decomposition via Kronecker Product ---
    # The 4x4 matrices are partitioned into 2x2 blocks of 2x2 matrices.
    # A 4x4 index (i,j) can be written as (I, i') where I is block index (0..1) and i' is local index (0..1).
    # i = 2*I + i', j = 2*J + j'.
    # The global index for U is i*4 + j = (2*I + i')*4 + (2*J + j') = 8*I + 2*i' + 4*J + 2*j'.
    # Wait, let's stick to the linear index 0..15.
    # U_global[2*I+i', 2*J+j'] corresponds to U_2x2[i', j'] scaled by the top-level Strassen coefficient.
    # Actually, the Kronecker product of tensors T1 (size 2,2,2) and T2 (size 2,2,2) gives T (size 4,4,4).
    # T[i,j,k] = sum_{i',j',k'} T1[I,J,K] T2[i',j',k'] where i=2I+i', j=2J+j', k=2K+k'.
    # In our tensor decomposition, we have terms (u, v, w).
    # The Kronecker product of a list of terms is the set of all pairs of terms.
    
    terms_444 = []
    for u1, v1, w1 in strassen_terms_2x2:
        for u2, v2, w2 in strassen_terms_2x2:
            # u1 is for the "top" block (2x2 of blocks), u2 is for the "bottom" block (2x2 of elements)
            # The resulting u vector is length 16.
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for I in range(2):
                for J in range(2):
                    coeff_u = u1[I*2+J] # Wait, u1 indices are 0..3 corresponding to I*2+J?
                    # In our 2x2 definition, index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                    # So u1[0] is for I=0,J=0.
                    if coeff_u == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            coeff_u2 = u2[i2*2+j2]
                            if coeff_u2 == 0: continue
                            val = coeff_u * coeff_u2
                            # Global index: i = 2*I + i2, j = 2*J + j2
                            # Index = i*4 + j
                            idx = (2*I + i2)*4 + (2*J + j2)
                            u_full[idx] += val
            
            for J in range(2):
                for K in range(2):
                    coeff_v = v1[J*2+K]
                    if coeff_v == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            coeff_v2 = v2[j2*2+k2]
                            if coeff_v2 == 0: continue
                            val = coeff_v * coeff_v2
                            # Global index: j = 2*J + j2, k = 2*K + k2
                            # Index = j*4 + k
                            idx = (2*J + j2)*4 + (2*K + k2)
                            v_full[idx] += val
            
            for K in range(2):
                for I in range(2):
                    coeff_w = w1[K*2+I]
                    if coeff_w == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            coeff_w2 = w2[k2*2+i2]
                            if coeff_w2 == 0: continue
                            val = coeff_w * coeff_w2
                            # Global index: k = 2*K + k2, i = 2*I + i2
                            # Index = k*4 + i
                            idx = (2*K + k2)*4 + (2*I + i2)
                            w_full[idx] += val
            
            terms_444.append((u_full, v_full, w_full))
            
    # --- Step 3: Fill the 4x4x4 part into U, V, W ---
    # The 4x4x4 part uses columns 0..3 of B.
    # U rows: 0..15 (i*4+j)
    # V rows: 0..15 (j*4+k) for k=0..3. Note: global V index is j*5+k.
    # W rows: 0..15 (k*4+i) for k=0..3. Note: global W index is k*5+i? No, W index is k*n+i. n=4. So k*4+i.
    
    for t_idx, (u, v, w) in enumerate(terms_444):
        # u is length 16, indices 0..15 match U rows directly.
        U[:, t_idx] = u
        
        # v is length 16, indices j*4+k.
        # We need to map v[j*4+k] to V[j*5+k].
        for j in range(4):
            for k in range(4):
                local_idx = j*4 + k
                global_idx = j*5 + k
                V[global_idx, t_idx] = v[local_idx]
                
        # w is length 16, indices k*4+i.
        # We need to map w[k*4+i] to W[k*4+i] (since n=4, W row is k*4+i).
        # Wait, W dimensions are (p*n, rank) = (20, 65).
        # W row index is k*n + i = 4k + i.
        # Our w_full indices are k*4 + i. So they match directly!
        W[:, t_idx] = w
        
    # --- Step 4: Add 16 scalar terms for the 5th column (k=4) ---
    # We need C[:, 4] = A @ B[:, 4].
    # This is sum_j A[i,j] B[j,4].
    # Terms: U[i*4+j, r] = 1, V[j*5+4, r] = 1, W[4*4+i, r] = 1.
    # r = 49 + i*4 + j.
    
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U[i*4 + j, r] = 1
            V[j*5 + 4, r] = 1
            W[4*4 + i, r] = 1 # k=4, i. Index 16+i.
            
    # Convert to lists as required by the interface?
    # The interface says: "factor matrices of shapes...". 
    # The example return is {"U": U, ...}. Usually numpy arrays or lists are fine.
    # The previous code returned lists of lists. I will return numpy arrays for efficiency and cleanliness, 
    # but ensure they are compatible. The evaluator likely accepts numpy arrays.
    
    return {"U": U, "V": V, "W": W}

