<<<<<<< SEARCH
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    
    # Try to reduce rank by removing linearly dependent terms
    # A term r is redundant if U[:,r], V[:,r], W[:,r] are linearly dependent on others
    # Check if any term is a linear combination of others in a compatible way
    # Simple approach: check for duplicate or proportional columns
    
    def try_reduce(U, V, W, max_iter=50):
        """Try to reduce rank by finding and removing dependent terms."""
        n_ranks = U.shape[1]
        if n_ranks < 2:
            return U, V, W
        
        # Check for exact duplicates or proportional terms
        removed = set()
        for r in range(n_ranks):
            if r in removed:
                continue
            for s in range(r+1, n_ranks):
                if s in removed:
                    continue
                # Check if term s is proportional to term r
                # U[:,s] = alpha * U[:,r], V[:,s] = alpha * V[:,r], W[:,s] = alpha * W[:,r]
                nonzero_u_r = U[:, r] != 0
                nonzero_v_r = V[:, r] != 0
                nonzero_w_r = W[:, r] != 0
                
                # Check U proportionality
                if np.any(nonzero_u_r):
                    idx = np.where(nonzero_u_r)[0]
                    ratios_u = U[idx, s] / U[idx, r]
                    if len(ratios_u) > 0 and not np.allclose(ratios_u, ratios_u[0], atol=1e-10):
                        continue
                    alpha_u = ratios_u[0] if len(ratios_u) > 0 else 1.0
                else:
                    if np.any(U[:, s] != 0):
                        continue
                    alpha_u = 1.0
                
                # Check V proportionality
                if np.any(nonzero_v_r):
                    idx = np.where(nonzero_v_r)[0]
                    ratios_v = V[idx, s] / V[idx, r]
                    if len(ratios_v) > 0 and not np.allclose(ratios_v, ratios_v[0], atol=1e-10):
                        continue
                    alpha_v = ratios_v[0] if len(ratios_v) > 0 else 1.0
                else:
                    if np.any(V[:, s] != 0):
                        continue
                    alpha_v = 1.0
                
                # Check W proportionality
                if np.any(nonzero_w_r):
                    idx = np.where(nonzero_w_r)[0]
                    ratios_w = W[idx, s] / W[idx, r]
                    if len(ratios_w) > 0 and not np.allclose(ratios_w, ratios_w[0], atol=1e-10):
                        continue
                    alpha_w = ratios_w[0] if len(ratios_w) > 0 else 1.0
                else:
                    if np.any(W[:, s] != 0):
                        continue
                    alpha_w = 1.0
                
                # All must have same ratio
                if np.isclose(alpha_u, alpha_v, atol=1e-10) and np.isclose(alpha_v, alpha_w, atol=1e-10):
                    # Term s is proportional to term r - merge them
                    # U[:,r] += alpha * U[:,s], etc.
                    U[:, r] += alpha_u * U[:, s]
                    V[:, r] += alpha_v * V[:, s]
                    W[:, r] += alpha_w * W[:, s]
                    removed.add(s)
                    break
        
        if removed:
            keep = [i for i in range(n_ranks) if i not in removed]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
        
        return U, V, W
    
    U, V, W = try_reduce(U, V, W)
    
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
