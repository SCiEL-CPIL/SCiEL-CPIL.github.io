<<<<<<< SEARCH
def _greedy_rank_reduce(U, V, W, n, m, p, deadline):
    """Try to remove columns that are linearly redundant."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    reduced = True
    while reduced and time.time() < deadline:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, _ = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W
=======
def _greedy_rank_reduce(U, V, W, n, m, p, deadline):
    """Try to remove columns that are linearly redundant."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    reduced = True
    while reduced and time.time() < deadline:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, _ = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W

def _try_linear_combination_reduce(U, V, W, n, m, p, deadline):
    """
    Try to find linear combinations of columns that allow rank reduction.
    This explores the border rank concept: if columns c1 and c2 can be
    replaced by a single column via linear combination while preserving
    the tensor decomposition, we can reduce rank.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    # Try pairwise linear combinations
    for i in range(rank):
        if time.time() > deadline:
            break
        for j in range(i + 1, rank):
            if time.time() > deadline:
                break
            
            # Try replacing column j with a linear combination of columns i and j
            # that might allow removal of one column
            # We test if column j is "redundant" given column i with specific scaling
            
            # Test: can we express the contribution of column j using only column i?
            # This requires V[:,j] to be proportional to V[:,i] and W[:,j] proportional to W[:,i]
            # with the same proportionality constant, and U[:,j] adjustable
            
            v_i = current_V[:, i]
            v_j = current_V[:, j]
            w_i = current_W[:, i]
            w_j = current_W[:, j]
            
            # Check if v_j = alpha * v_i and w_j = beta * w_i
            # Then we need u_j = (alpha * beta) * u_i for the product to work
            # But we can adjust u_j freely, so we just need the products to match
            
            # Actually, let's try a different approach: test if removing column j
            # and adjusting column i can preserve the decomposition
            # This is complex, so let's try a simpler heuristic:
            # Check if the "direction" of column j in V and W spaces is similar to i
            
            # Compute correlation-like measure
            if np.linalg.norm(v_i) > 1e-10 and np.linalg.norm(v_j) > 1e-10:
                cos_v = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j))
            else:
                cos_v = 0
            
            if np.linalg.norm(w_i) > 1e-10 and np.linalg.norm(w_j) > 1e-10:
                cos_w = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j))
            else:
                cos_w = 0
            
            # If both are highly correlated, try removing j
            if abs(cos_v) > 0.95 and abs(cos_w) > 0.95:
                mask = [k for k in range(rank) if k != j]
                test_U = current_U[:, mask]
                test_V = current_V[:, mask]
                test_W = current_W[:, mask]
                
                valid, _ = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
                
                if valid:
                    current_U, current_V, current_W = test_U, test_V, test_W
                    rank -= 1
                    # Reset i,j for the reduced rank
                    i = 0
                    j = 0
                    break
    
    return current_U, current_V, current_W

def _local_search_reduce(U, V, W, n, m, p, deadline):
    """
    Perform local search by trying to replace pairs of columns with
    optimized linear combinations to enable rank reduction.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    # Try each pair of columns for potential merging
    for i in range(rank):
        if time.time() > deadline - 5:
            break
        for j in range(i + 1, rank):
            if time.time() > deadline - 5:
                break
            
            # Try to find alpha, beta such that:
            # (U[:,i] + alpha*U[:,j], V[:,i] + beta*V[:,j], W[:,i] + gamma*W[:,j])
            # can replace both columns, allowing removal of one
            
            # Simplified: try a few specific alpha values
            for alpha in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
                if time.time() > deadline - 5:
                    break
                
                # Create a new set where column j is modified
                test_U = current_U.copy()
                test_V = current_V.copy()
                test_W = current_W.copy()
                
                # Modify column j to be a combination
                test_U[:, j] = current_U[:, j] + alpha * current_U[:, i]
                test_V[:, j] = current_V[:, j] + alpha * current_V[:, i]
                test_W[:, j] = current_W[:, j] + alpha * current_W[:, i]
                
                # Try removing column i
                mask = [k for k in range(rank) if k != i]
                test_U_rm = test_U[:, mask]
                test_V_rm = test_V[:, mask]
                test_W_rm = test_W[:, mask]
                
                valid, _ = _verify_decomposition(test_U_rm, test_V_rm, test_W_rm, n, m, p, tol=1e-8)
                
                if valid:
                    current_U, current_V, current_W = test_U_rm, test_V_rm, test_W_rm
                    rank -= 1
                    best_rank = rank
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    break
            
            if time.time() > deadline - 5:
                break
    
    return best_U, best_V, best_W
>>>>>>> REPLACE

