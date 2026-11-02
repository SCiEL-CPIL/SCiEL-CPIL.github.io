import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    rank_school = n * m * p  # 56

    # Build the multiplication tensor T as a flat vector
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0

    # Schoolbook solution
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)

    # Strategy: Exploit structure - for each (i,k), the equation is:
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all j
    # This means for fixed (i,k), the vectors [U[i*m+j,t]*W[k*n+i,t]]_j dotted with V[:,j*p+k] = 1
    
    # Let's try a structured decomposition based on the (i,k) pairs
    # There are n*p = 14 (i,k) pairs, each requiring consistency across m=4 j-values
    
    # Approach: Use rank = n*p + (n-1)*m*p = 14 + 28 = 42? Or other formulas.
    # Actually, let me try to find a decomposition with rank around 35-45.
    
    # Greedy approach with structure-aware initialization
    residual = T_flat.copy()
    U_list, V_list, W_list = [], [], []
    
    max_steps = 50
    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break
        
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        n_trials = 50
        for trial in range(n_trials):
            if time.time() - start_time > deadline:
                break
            
            # Random sparse vectors with ±1 entries
            nnz_u = random.randint(3, 8)
            nnz_v = random.randint(10, 28)
            nnz_w = random.randint(5, 14)
            
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            idx_u = random.sample(range(dim_u), nnz_u)
            idx_v = random.sample(range(dim_v), nnz_v)
            idx_w = random.sample(range(dim_w), nnz_w)
            
            u[idx_u] = np.random.choice([-1, 1], nnz_u)
            v[idx_v] = np.random.choice([-1, 1], nnz_v)
            w[idx_w] = np.random.choice([-1, 1], nnz_w)
            
            tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
            score = np.dot(tensor, residual)
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_score <= 1e-10 or best_u is None:
            break
        
        tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
        residual -= tensor
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(residual, 0, atol=1e-10):
            break
    
    # Try to refine with least squares
    if len(U_list) > 0 and len(U_list) < best_rank:
        U_g = np.column_stack(U_list)
        V_g = np.column_stack(V_list)
        W_g = np.column_stack(W_list)
        rank_g = len(U_list)
        
        M = np.einsum('it,jt,kt->ijt', U_g, V_g, W_g).reshape(-1, rank_g)
        try:
            c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            T_recon = (M @ c).reshape(dim_u, dim_v, dim_w)
            if np.allclose(T_recon, T_flat, atol=1e-10):
                # Check if we can get integer/half-integer entries
                # Scale to make entries nice
                U_final = U_g * c[None, :]
                V_final = V_g
                W_final = W_g
                
                # Try to clean to half-integers
                def clean_half_int(x):
                    x2 = x * 2
                    return np.round(x2) / 2
                
                U_clean = clean_half_int(U_final)
                V_clean = clean_half_int(V_final)
                W_clean = clean_half_int(W_final)
                
                if verify_decomposition(U_clean, V_clean, W_clean, T_flat):
                    if rank_g < best_rank:
                        best_rank = rank_g
                        best_U = U_clean
                        best_V = V_clean
                        best_W = W_clean
        except:
            pass

    # Also try a more structured approach: exploit the fact that n=2
    # For each k, we have 2 rows in C, and 4 columns in A/B per row
    # Let's try rank = 2*p + m*p = 14 + 28 = 42 or similar
    
    # Alternative: use the identity that for n=2, we can sometimes save rank
    # by sharing terms between the two rows
    
    if time.time() - start_time < deadline - 5 and best_rank > 40:
        # Try a structured decomposition
        # For each k in range(p), and for the two i values:
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # C[1,k] = sum_j A[1,j]*B[j,k]
        
        # We can use rank = p*(n + m - 1) = 7*(2+4-1) = 35 if such a decomposition exists
        # This is based on the idea of "sharing" the B[j,k] terms
        
        # Let's try: for each j, use m terms, and for the n=2 structure, use additional terms
        # Actually, a known bound is rank <= n*m*p - min(n,m,p)*something
        
        # Let me try a direct constructive approach for rank = 42
        # Use the fact that we can factor out some structure
        
        # For now, try greedy with better initialization
        residual2 = T_flat.copy()
        U_list2, V_list2, W_list2 = [], [], []
        
        for step in range(45):
            if time.time() - start_time > deadline:
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for trial in range(40):
                if time.time() - start_time > deadline:
                    break
                
                # More structured: ensure some coverage of all (i,k) pairs
                nnz_u = random.randint(4, 8)
                nnz_v = random.randint(12, 28)
                nnz_w = random.randint(6, 14)
                
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                idx_u = random.sample(range(dim_u), nnz_u)
                idx_v = random.sample(range(dim_v), nnz_v)
                idx_w = random.sample(range(dim_w), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual2)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10 or best_u is None:
                break
            
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual2 -= tensor
            
            U_list2.append(best_u)
            V_list2.append(best_v)
            W_list2.append(best_w)
            
            if np.allclose(residual2, 0, atol=1e-10):
                break
        
        if len(U_list2) > 0 and len(U_list2) < best_rank:
            U_g2 = np.column_stack(U_list2)
            V_g2 = np.column_stack(V_list2)
            W_g2 = np.column_stack(W_list2)
            rank_g2 = len(U_list2)
            
            M2 = np.einsum('it,jt,kt->ijt', U_g2, V_g2, W_g2).reshape(-1, rank_g2)
            try:
                c2, _, _, _ = np.linalg.lstsq(M2, T_flat, rcond=None)
                T_recon2 = (M2 @ c2).reshape(dim_u, dim_v, dim_w)
                if np.allclose(T_recon2, T_flat, atol=1e-10):
                    def clean_half_int(x):
                        x2 = x * 2
                        return np.round(x2) / 2
                    
                    U_final2 = U_g2 * c2[None, :]
                    V_final2 = V_g2
                    W_final2 = W_g2
                    
                    U_clean2 = clean_half_int(U_final2)
                    V_clean2 = clean_half_int(V_final2)
                    W_clean2 = clean_half_int(W_final2)
                    
                    if verify_decomposition(U_clean2, V_clean2, W_clean2, T_flat):
                        if rank_g2 < best_rank:
                            best_rank = rank_g2
                            best_U = U_clean2
                            best_V = V_clean2
                            best_W = W_clean2
            except:
                pass

    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
