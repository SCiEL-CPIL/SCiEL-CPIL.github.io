"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build rank-21 Strassen decomposition
    rank21 = 21
    U21 = np.zeros((n * m, rank21))
    V21 = np.zeros((m * p, rank21))
    W21 = np.zeros((p * n, rank21))
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
        [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
        [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
        [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
        [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
        [(1, 1, 1)],              # k=5 (M6) -> C11
        [(0, 0, 1)]               # k=6 (M7) -> C00
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U21[idx, col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V21[v_idx, col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W21[w_idx, col] = val
    
    # Build the full matrix multiplication tensor T
    # T[(i,j),(j,k),(i,k)] = 1 for all valid (i,j,k)
    # Flatten: T_flat[idx] where idx = ((i*m + j)*p + k) * (n) + ... 
    # Actually, let's use the convention: T[(i,j),(j,k),(i,k)]
    # Number of entries: n*m*p = 2*2*6 = 24
    # Index: ((i*m + j)*p + k) gives 0..23
    T = np.ones(n * m * p)
    
    # Compute the tensor contribution of each rank-1 term
    # Term r contributes: U21[:, r] ⊗ V21[:, r] ⊗ W21[:, r]
    # The (i,j,k) entry of this tensor is U21[i*m+j, r] * V21[j*p+k, r] * W21[k*n+i, r]
    
    def tensor_entry(U_col, V_col, W_col, i, j, k):
        return U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
    
    # For each column r, check if T - sum_{r' != r} (U_{r'} ⊗ V_{r'} ⊗ W_{r'}) has rank 1
    # A 3-tensor X has rank 1 iff there exist vectors u, v, w such that X[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    # Equivalently, all 2x2 minors of certain matricizations vanish, and the tensor is not zero.
    
    # Check if a tensor X (flattened to 24 entries) has rank 1
    def has_rank_1(X_flat):
        """Check if a tensor with entries X_flat[((i*m+j)*p+k)] has tensor rank 1."""
        # Reshape to (n*m, p, n) -> but we need (i,j),(j,k),(i,k) structure
        # X_flat[idx] where idx = (i*m+j)*p + k
        # Entry (i,j,k) = X_flat[(i*m+j)*p + k]
        
        # A tensor has rank 1 iff it can be written as u ⊗ v ⊗ w
        # Check: pick a reference entry that is non-zero, then all other entries must be
        # consistent with a rank-1 factorization.
        
        # Find first non-zero entry
        nonzero_idx = None
        for idx in range(len(X_flat)):
            if abs(X_flat[idx]) > 1e-10:
                nonzero_idx = idx
                break
        
        if nonzero_idx is None:
            return None  # Zero tensor, not rank 1
        
        # Decompose idx into (i, j, k)
        i = (nonzero_idx // p) // m
        j = (nonzero_idx // p) % m
        k = nonzero_idx % p
        
        # For rank 1: X[i',j',k'] = X[i,j,k] * (u[i',j']/u[i,j]) * (v[j',k']/v[j,k]) * (w[k',i']/w[k,i])
        # This is complex. Instead, use the fact that for rank-1 tensor,
        # all "slice" matrices must be rank 1.
        
        # Alternative: use the condition that for any (i,j,k), (i',j,k), (i,j',k), (i,j,k')
        # we have X[i,j,k]*X[i',j',k'] = X[i',j,k]*X[i,j',k']
        # This is a necessary condition for rank 1.
        
        # Check all quadruples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                lhs = X_flat[(i*m+j)*p+k] * X_flat[(i2*m+j2)*p+k2]
                                rhs = X_flat[(i2*m+j)*p+k] * X_flat[(i*m+j2)*p+k2]
                                if abs(lhs - rhs) > 1e-8:
                                    return False
        
        # Also check that it's not zero (already done)
        # And check consistency with another set of indices
        # For rank 1, we also need: X[i,j,k]*X[i,j2,k2] = X[i,j2,k]*X[i,j,k2]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        for k2 in range(p):
                            lhs = X_flat[(i*m+j)*p+k] * X_flat[(i*m+j2)*p+k2]
                            rhs = X_flat[(i*m+j2)*p+k] * X_flat[(i*m+j)*p+k2]
                            if abs(lhs - rhs) > 1e-8:
                                return False
        
        # And: X[i,j,k]*X[i2,j,k2] = X[i2,j,k]*X[i,j,k2]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            lhs = X_flat[(i*m+j)*p+k] * X_flat[(i2*m+j)*p+k2]
                            rhs = X_flat[(i2*m+j)*p+k] * X_flat[(i*m+j)*p+k2]
                            if abs(lhs - rhs) > 1e-8:
                                return False
        
        # If all these conditions hold, the tensor is rank 1 (or zero)
        # Extract the factors
        # U[i*m+j] = X[i,j,k0] / (V[j,k0] * W[k0,i]) for some fixed k0
        # This is getting complex. Let's just return True and extract later.
        return True
    
    # For each column to remove, check if the residual has rank 1
    for r_remove in range(rank21):
        if time.time() > deadline - 20:
            break
        
        # Compute residual tensor
        X_flat = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    total = 0
                    for r in range(rank21):
                        if r == r_remove:
                            continue
                        total += U21[i*m+j, r] * V21[j*p+k, r] * W21[k*n+i, r]
                    X_flat[idx] = T[idx] - total
        
        # Check if X_flat has rank 1
        if has_rank_1(X_flat):
            # Extract factors
            # Find a non-zero entry to anchor
            anchor_idx = None
            for idx in range(len(X_flat)):
                if abs(X_flat[idx]) > 1e-10:
                    anchor_idx = idx
                    break
            
            if anchor_idx is None:
                continue
            
            i_a = (anchor_idx // p) // m
            j_a = (anchor_idx // p) % m
            k_a = anchor_idx % p
            
            # For rank-1 tensor: X[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # Fix: set u[i_a, j_a] = 1, then v[j_a, k] = X[i_a, j_a, k] / w[k_a, i_a]
            # This is underdetermined. We can choose a gauge.
            # Let's set: u[i,j] = X[i,j,k_ref] for some fixed k_ref where possible
            # Better: use the fact that for any (i,j,k), X[i,j,k] = u[i,j]*v[j,k]*w[k,i]
            # Choose: for each (i,j), u[i,j] = X[i,j,k0] where k0 is fixed
            # Then v[j,k] = X[i0,j,k] / (u[i0,j] * w[k,i0])
            # This requires careful gauge fixing.
            
            # Simpler: just construct U, V, W from X_flat directly
            # We need U_new, V_new, W_new such that for all (i,j,k):
            # U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i] = X_flat[(i*m+j)*p+k]
            
            # Gauge: set W[k*n+i] = 1 for all (k,i) where possible, then
            # U[i*m+j] * V[j*p+k] = X_flat[(i*m+j)*p+k]
            # This is a matrix factorization problem for each... no, it's coupled.
            
            # Alternative gauge: set U[i*m+j] = 1 for all (i,j), then
            # V[j*p+k] * W[k*n+i] = X_flat[(i*m+j)*p+k]
            # For fixed j: V[j*p+k] * W[k*n+i] must equal X_flat[(i*m+j)*p+k] for all (i,k)
            # This means the matrix [X_flat[(i*m+j)*p+k]]_{i,k} must be rank 1
            # Let's check this.
            
            U_new = np.zeros(n * m)
            V_new = np.zeros(m * p)
            W_new = np.zeros(p * n)
            
            # Try gauge: W[k*n+i] = 1 for all (k,i)
            valid = True
            for j in range(m):
                # Matrix M[i,k] = X_flat[(i*m+j)*p+k]
                # Must be rank 1: M[i,k] = a[i] * b[k]
                # Check rank
                M = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        M[i, k] = X_flat[(i*m+j)*p+k]
                
                # Check if M is rank 1
                if n > 0 and p > 0:
                    if abs(M[0,0]) > 1e-10:
                        # Rank 1 iff M[i,k] = M[0,0] * (M[i,0]/M[0,0]) * (M[0,k]/M[0,0]) = M[i,0]*M[0,k]/M[0,0]
                        for i in range(n):
                            for k in range(p):
                                if abs(M[i,k] - M[i,0]*M[0,k]/M[0,0]) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                    else:
                        # M[0,0] = 0, check if all entries are 0 or consistent
                        # For rank 1 with M[0,0]=0, either row 0 is all 0 or col 0 is all 0
                        row0_all_zero = all(abs(M[0,k]) < 1e-10 for k in range(p))
                        col0_all_zero = all(abs(M[i,0]) < 1e-10 for i in range(n))
                        if not (row0_all_zero or col0_all_zero):
                            valid = False
                        else:
                            # Check consistency
                            for i in range(n):
                                for k in range(p):
                                    if not row0_all_zero and not col0_all_zero:
                                        pass
                                    elif row0_all_zero:
                                        if abs(M[i,k]) > 1e-10 and abs(M[i,0]) < 1e-10:
                                            valid = False
                                    elif col0_all_zero:
                                        if abs(M[i,k]) > 1e-10 and abs(M[0,k]) < 1e-10:
                                            valid = False
                            if not valid:
                                break
                if not valid:
                    break
            
            if valid:
                # Extract factors with gauge W[k*n+i] = 1
                # Then U[i*m+j] * V[j*p+k] = X_flat[(i*m+j)*p+k]
                # For each (i,j), U[i*m+j] = X_flat[(i*m+j)*p+0] / V[j*p+0] if V[j*p+0] != 0
                # This is still coupled. Let's use a different approach.
                
                # Set U[i*m+j] = X_flat[(i*m+j)*p+0] for k=0, then V[j*p+0] = 1
                # Then for k>0: V[j*p+k] = X_flat[(i*m+j)*p+k] / U[i*m+j] for any i where U[i*m+j] != 0
                # This is messy. Let's just return rank 21 if extraction is too complex.
                pass
        
        # If we found a valid rank-1 residual, we'd construct the rank-20 solution here
        # For now, continue searching
    
    # If no rank-20 solution found, return rank 21
    U = U21.tolist()
    V = V21.tolist()
    W = W21.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
