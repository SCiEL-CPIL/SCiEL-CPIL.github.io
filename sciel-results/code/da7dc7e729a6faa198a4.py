import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Fallback: Schoolbook decomposition (Rank 120)
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}

    # Search for lower rank
    # We try ranks from 119 down to a lower bound.
    # Lower bound for rank is max(n*m, m*p, p*n) is not strictly true, but rank >= n*m is true?
    # Actually, rank >= n*m is a known lower bound for M_{n,m,p} if n <= m <= p?
    # Let's just search down to 80 to be safe and fast.
    
    found = False
    best_result = None
    
    # To speed up, we can fix U and V to be sparse random and solve for W.
    # If we find a W that is close to half-integers, we verify it.
    
    for R in range(119, 79, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random instances
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Random U, V with small integers
            # Using -1, 0, 1
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build the linear system for W
            # Equations: for each i, j, jp, k
            # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = 1 if j==jp else 0
            # Variables: W[k*n+i, r]
            
            num_eqs = n * m * m * p
            num_vars = p * n * R
            
            # Construct A matrix (num_eqs x num_vars)
            # A[eq_idx, var_idx]
            # eq_idx = (i * m + j) * m * p + jp * p + k
            # var_idx = (k * n + i) * R + r
            
            # Optimization: We can build A more efficiently by iterating r.
            # But for small sizes, direct construction is fine.
            
            A = np.zeros((num_eqs, num_vars), dtype=float)
            
            # Pre-compute indices for speed
            # This loop is the bottleneck.
            # We can vectorize the filling of A.
            # For each r, u_col is n*m, v_col is m*p.
            # The contribution to A is sparse in a structured way.
            
            for r in range(R):
                u_col = U[:, r]
                v_col = V[:, r]
                
                # For each i, k, the variable is W[k, i, r]
                # It appears in equations for all j, jp with coeff U[i, j, r] * V[jp, k, r]
                
                for i in range(n):
                    u_vals = u_col[i*m : (i+1)*m] # j=0..m-1
                    
                    for k in range(p):
                        v_vals = v_col[k::p] # jp=0..m-1
                        
                        var_idx = (k * n + i) * R + r
                        
                        # Block of size m x m
                        # Row j, Col jp -> coeff u_vals[j] * v_vals[jp]
                        # Eq indices for this block:
                        # base_eq = i * (m * m * p) + k
                        # Indices are not contiguous in 1D array if we flatten (j, jp) row-major?
                        # eq_idx(j, jp) = i*m*m*p + j*m*p + jp*p + k
                        # If we create a 2D array of indices:
                        # js = np.arange(m)
                        # jps = np.arange(m)
                        # eq_indices = i*m*m*p + js[:, None]*m*p + jps[None, :]*p + k
                        # Then A[eq_indices.flatten(), var_idx] = (u_vals[:, None] * v_vals[None, :]).flatten()
                        
                        js = np.arange(m)
                        jps = np.arange(m)
                        eq_indices = i * (m * m * p) + js[:, None] * (m * p) + jps[None, :] * p + k
                        coeffs = np.outer(u_vals, v_vals)
                        
                        A[eq_indices.flatten(), var_idx] = coeffs.flatten()
            
            # Build b
            b = np.zeros(num_eqs)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * m * p + j * p + k
                        b[eq_idx] = 1.0
            
            # Solve A W = b
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W_sol = W_flat.reshape((p, n, R))
            
            # Check if W_sol is close to half-integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.allclose(W_sol, W_rounded / 2.0, atol=1e-6):
                # Candidate found. Verify exactly.
                W_int = W_rounded.astype(int)
                
                # Verify: C_ik = sum_j A_ij B_jk
                # Using the decomposition:
                # C_ik = sum_r (sum_j U_ijr A_ij) (sum_k' V_jk'r B_jk') W_kir ?
                # No, the decomposition is:
                # C_ik = sum_r U_{ijr} V_{jkr} W_{kir} ?
                # The standard tensor decomposition for matrix multiplication is:
                # C_{ik} = sum_{j} A_{ij} B_{jk}
                # = sum_{j, r} U_{ijr} V_{jkr} W_{kir} ?
                # Wait, the indices in the problem statement:
                # U: (n*m, R) -> U[i, j, r]
                # V: (m*p, R) -> V[j, k, r]
                # W: (p*n, R) -> W[k, i, r]
                # The product is sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
                # This sums over r. But we also need to sum over j?
                # The tensor is T[i, j, k] = delta_{j, j'}? No.
                # The identity is:
                # sum_j A_{ij} B_{jk} = sum_{j, r} U_{ijr} V_{jkr} W_{kir}
                # This is not quite right. The standard form is:
                # C_{ik} = sum_{r} (sum_j U_{ijr} A_{ij}) (sum_k V_{jkr} B_{jk}) W_{kir} ??
                # No, the bilinear form is:
                # C_{ik} = sum_{j} A_{ij} B_{jk}
                # Decomposition:
                # C_{ik} = sum_{r} \alpha_r(A) \beta_r(B) \gamma_r(C)? No.
                # It is:
                # C_{ik} = sum_{r} U_{i, j, r} V_{j, k, r} W_{k, i, r} is NOT summing over j.
                # The correct expansion is:
                # sum_{j} A_{ij} B_{jk} = sum_{j, r} U_{ijr} V_{jkr} W_{kir}
                # This requires that for each r, the term is linear in A and B.
                # So, for a fixed r, let U_r be a matrix, V_r be a matrix, W_r be a matrix?
                # No, U, V, W are the factors.
                # The equation is:
                # sum_j A_{ij} B_{jk} = sum_{r} (sum_j U_{ijr} A_{ij}) (sum_k V_{jkr} B_{jk}) W_{kir}
                # This is not standard.
                
                # Let's check the inspiration's equation:
                # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = 1 if j==jp else 0
                # This equation is for the *basis* of the tensor.
                # The tensor T has entries T[i, j, k] = 1 if j is the middle index?
                # No, the matrix multiplication tensor is defined by:
                # C_{ik} = sum_j A_{ij} B_{jk}
                # The "input" variables are A_{ij} and B_{jk}.
                # The "output" variable is C_{ik}.
                # The bilinear map is:
                # <C, A, B> = sum_{i,j,k} C_{ik} A_{ij} B_{jk}
                # We want to decompose this trilinear form.
                # <C, A, B> = sum_r (sum_{i,j} U_{ijr} A_{ij}) (sum_{j,k} V_{jkr} B_{jk}) (sum_{i,k} W_{kir} C_{ik})
                # This is the standard rank decomposition.
                # So the condition is:
                # sum_r U_{ijr} V_{jkr} W_{kir} = delta_{j, j'}? No.
                # The coefficient of A_{ij} B_{jk} C_{ik} in the expansion must be 1.
                # In the sum sum_r U_{ijr} V_{jkr} W_{kir}, the term A_{ij} B_{jk} C_{ik} appears with coefficient:
                # sum_r U_{ijr} V_{jkr} W_{kir}.
                # This must equal 1 for all i, j, k?
                # No, the indices j in A and B are summed over in the matrix product.
                # In the trilinear form <C, A, B> = sum_{i,j,k} C_{ik} A_{ij} B_{jk}, the indices i, j, k are all summed.
                # So the coefficient of the monomial A_{i_1 j_1} B_{j_2 k_2} C_{i_3 k_3} in the expansion of sum_r ...
                # is sum_r U_{i_1 j_1 r} V_{j_2 k_2 r} W_{i_3 k_3 r}.
                # This must be 1 if (i_1, j_1, k_2, i_3, k_3) corresponds to the identity, i.e., i_1=i_3, j_1=j_2, k_2=k_3.
                # And 0 otherwise.
                # So the condition is:
                # sum_r U_{i j r} V_{j' k r} W_{i' k' r} = 1 if i=i', j=j', k=k' else 0.
                
                # The inspiration used:
                # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = 1 if j==jp else 0
                # This implies i=i' and k=k' are fixed in the loop?
                # The inspiration loops i, k and checks j, jp.
                # So it enforces:
                # For fixed i, k: sum_r U_{ijr} V_{j'kr} W_{kir} = 1 if j=j' else 0.
                # This is exactly the condition.
                
                # So we verify this condition exactly with integers.
                valid = True
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for jp in range(m):
                                lhs = 0
                                for r in range(R):
                                    lhs += U[i*m+j, r] * V[jp*p+k, r] * W_int[k*n+i, r]
                                expected = 1 if j == jp else 0
                                if lhs != expected:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_int.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback
    return get_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
