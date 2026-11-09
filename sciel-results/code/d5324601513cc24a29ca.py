    def _makespan(perm, P):
        n_m = P.shape[1]
        # P is (n_jobs, n_machines)
        # perm is list of job indices
        # Extract processing times for the permutation order
        # P_perm = P[perm, :] -> (n_jobs, n_machines)
        # Then compute completion times
        # C[i, m] = max(C[i, m-1], C[i-1, m]) + P_perm[i, m]
        # This is still a loop over i.
        # But we can vectorize the inner part?
        # C[i] = max(C[i-1], P_perm[i]) + P_perm[i] ... no, max is element-wise.
        # C[i] = np.maximum(C[i-1], P_perm[i]) + P_perm[i]
        # This works!
        
        P_perm = P[perm]
        C_prev = np.zeros(n_m)
        for i in range(len(perm)):
            C_curr = np.maximum(C_prev, P_perm[i]) + P_perm[i]
            C_prev = C_curr
        return C_prev[-1]
    