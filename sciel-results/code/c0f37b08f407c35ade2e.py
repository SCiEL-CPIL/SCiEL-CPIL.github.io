def _local_search(a, b, p, deadline):
    """
    Fast local search with 1-0 interchanges and 2-0 swaps.
    deadline: absolute time (time.time()) to stop by.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = _compute_cost(a, b, p)
    
    # Precompute for fast delta calculations
    # For swapping positions i and j (2-0 interchange):
    # Delta = sum_{k != i,j} (a[i,k] + a[k,i]) * (D[j,k] - D[i,k]) 
    #         + (a[i,j] + a[j,i]) * (D[j,i] - D[i,j])
    #         + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
    # Using A_sym = A + A.T:
    # Delta = sum_{k} A_sym[i,k] * D[j,k] - sum_{k} A_sym[i,k] * D[i,k]
    #         + sum_{k} A_sym[j,k] * D[i,k] - sum_{k} A_sym[j,k] * D[j,k]
    #         + correction for k=i and k=j terms
    A_sym = a + a.T
    
    while time.time() < deadline:
        improved = False
        
        # 2-0 swap local search
        for i in range(n - 1):
            if time.time() > deadline:
                return p, current_cost
            
            # For all j > i, compute delta for swapping p[i] and p[j]
            js = np.arange(i + 1, n)
            
            # Vectorized: for each j in js, delta = 
            #   sum_k A_sym[i,k]*(D[j,k]-D[i,k]) + sum_k A_sym[j,k]*(D[i,k]-D[j,k])
            # = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
            
            # But this includes k=i and k=j which need correction.
            # Let's use the standard formula:
            # delta(i,j) = (A_sym[i] - A_sym[j]) @ (D[j] - D[i]) 
            #              + (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
            #              + (A[i,j] - A[j,i]) * (D[j,i] - D[i,j])
            
            # Vectorized version:
            # For each j: (A_sym[i] - A_sym[j]) @ (D[j] - D[i])
            # = A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            
            # Precompute A_sym[i] @ D[i] (scalar)
            AiD_i = A_sym[i] @ D[i]
            
            # For each j: A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            # = (A_sym[i] @ D[js].T) - AiD_i - (D[i] @ A_sym[js].T) + np.sum(D[js] * A_sym[js], axis=1)
            
            term1 = D[js] @ A_sym[i]  # shape: (len(js),)
            term2 = A_sym[js] @ D[i]  # shape: (len(js),)
            term3 = np.sum(D[js] * A_sym[js], axis=1)  # shape: (len(js),)
            
            base_delta = term1 - AiD_i - term2 + term3
            
            # Correction for diagonal and cross terms
            # The base formula double-counts or miscounts some terms.
            # Correct delta for swap(i,j):
            # = sum_{k!=i,j} (A_sym[i,k]-A_sym[j,k])*(D[j,k]-D[i,k])
            #   + (A_sym[i,i]-A_sym[j,i])*(D[j,i]-D[i,i])  -- but this is part of the sum
            #   
            # Actually, let's use the verified formula:
            # delta = (A_sym[i] - A_sym[j]) @ (D[j] - D[i])
            #         + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            #         + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
            # (A_sym[i] - A_sym[j]) @ (D[j] - D[i]) for each j:
            # = A_sym[i] @ D[j] - A_sym[i] @ D[i] - A_sym[j] @ D[i] + A_sym[j] @ D[j]
            # This is what we have as base_delta.
            
            # But we need to add:
            # (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            diag_corr = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
            # (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_corr = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = base_delta + diag_corr + cross_corr
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost
