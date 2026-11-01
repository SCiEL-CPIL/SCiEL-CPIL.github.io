def local_search_vec(x, time_limit):
    x = x.copy()
    R = np.correlate(x, x, mode='full')[N-1:]
    best_psl = np.max(np.abs(R[1:]))
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Build contrib matrix: shape (N, N)
        # contrib[j, k] = (x[j+k] if j+k<N else 0) + (x[j-k] if j-k>=0 else 0)
        
        # Part 1: x[j+k]
        # For fixed j, k ranges 0..N-1.
        # x[j+k] is valid if j+k < N => k < N-j.
        # We can use np.lib.stride_tricks or just loops?
        # N=400, loop over j to build matrix is 400 * O(N) = O(N^2).
        
        contrib = np.zeros((N, N))
        for j in range(N):
            # x[j:] has length N-j
            contrib[j, :N-j] += x[j:]
            # x[j::-1] has length j+1
            contrib[j, :j+1] += x[j::-1]
            
        # R_all[j, k] = R[k] - 2*x[j]*contrib[j, k]
        R_all = R[None, :] - 2 * x[:, None] * contrib
        
        # PSL for each j
        # Ignore k=0
        PSL_all = np.max(np.abs(R_all[:, 1:]), axis=1)
        
        best_j = np.argmin(PSL_all)
        best_new_psl = PSL_all[best_j]
        
        if best_new_psl < best_psl:
            x[best_j] *= -1
            R = R_all[best_j, :]
            best_psl = best_new_psl
            improved = True
        else:
            break
            
    return x, best_psl
