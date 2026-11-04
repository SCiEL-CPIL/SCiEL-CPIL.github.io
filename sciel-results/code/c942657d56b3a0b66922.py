<<<<<<< SEARCH
    def try_structured_random(rank, num_trials=50, time_limit=None):
        """
        Try to find a rank-r exact decomposition using random U, V with integer entries,
        then solve for W.
        """
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            for _ in range(n*m):
                U[_] = np.random.choice([-1, 0, 1], size=rank)
            for _ in range(m*p):
                V[_] = np.random.choice([-1, 0, 1], size=rank)
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            # For each (k,i), we need sum_r M[k*n+i, r] * W[k*n+i, r] = 1
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                found = False
                # Try single non-zero W entry
                for r in range(rank):
                    if M[idx, r] != 0:
                        w_val = 1.0 / M[idx, r]
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                
                if not found:
                    # Try two non-zero W entries
                    for r1 in range(rank):
                        if M[idx, r1] == 0:
                            continue
                        for r2 in range(r1+1, rank):
                            if M[idx, r2] == 0:
                                continue
                            for a in range(-3, 4):
                                for b in range(-3, 4):
                                    if a == 0 and b == 0:
                                        continue
                                    if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                        W[idx, r1] = a
                                        W[idx, r2] = b
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
=======
    def try_sparse_w_random(rank, num_trials=200, time_limit=None):
        """
        Try to find a rank-r exact decomposition using random U, V with entries in {-1, 0, 1}.
        We look for a W that is sparse (at most 1 non-zero entry per row) and integer-valued.
        This requires that for each row idx of M, there exists an r such that M[idx, r] is +/- 1.
        """
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            # Use random integers -1, 0, 1
            U[:] = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V[:] = np.random.choice([-1, 0, 1], size=(m*p, rank))
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            # Check if we can form W with sparse integer entries
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                # Look for a column r where M[idx, r] is 1 or -1
                found = False
                for r in range(rank):
                    if M[idx, r] == 1:
                        W[idx, r] = 1.0
                        found = True
                        break
                    elif M[idx, r] == -1:
                        W[idx, r] = -1.0
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                # Verify full decomposition
                if check_decomposition(U, V, W, rank):
                    return (U, V, W)
        
        return None
>>>>>>> REPLACE

