<<<<<<< SEARCH
    # Search for lower rank decompositions
    # Start from just below schoolbook and work down
    min_rank_to_try = max(20, best_rank - 50)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = structured_search(r, seed=42 + r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # Continue searching for even lower rank
            continue
        
        # If time allows, also try pure random with gradient descent
        if time.time() < deadline - 30:
            for trial in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + r + trial)
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
                
                # Run gradient descent
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    T = tensor_dot(U, V, W, r)
                    residual = T - 1.0
                    max_res = np.max(np.abs(residual))
                    
                    if max_res < 1e-10:
                        break
                    
                    U3 = U.reshape(n, m, r)
                    V3 = V.reshape(m, p, r)
                    W3 = W.reshape(p, n, r)
                    
                    grad_U = np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                    grad_V = np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
                    grad_W = np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
                    
                    U -= 0.05 * grad_U.reshape(n*m, r)
                    V -= 0.05 * grad_V.reshape(m*p, r)
                    W -= 0.05 * grad_W.reshape(p*n, r)
                
                if check_decomposition(U, V, W, r, tol=1e-9):
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
    
    if time.time() > deadline:
        return best_result

    return best_result
=======
    return best_result
>>>>>>> REPLACE
