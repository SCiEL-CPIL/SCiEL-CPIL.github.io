<<<<<<< SEARCH
    # Try ranks from 50 down to see if we can beat schoolbook
    # Use structured initialization: start from schoolbook and perturb
    for R in [50, 51, 52, 53]:
        if time.time() > deadline - 10:
            break
        
        # Use multiple restarts with different initializations
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(num_a, R) * 0.3
            V = np.random.randn(num_b, R) * 0.3
            W = np.random.randn(num_c, R) * 0.3
            
            for it in range(60):
                if time.time() > deadline - 3:
                    break
                
                U = solve_for_U(V, W, target)
                V = solve_for_V(U, W, target)
                W = solve_for_W(U, V, target)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-14:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_U is not None and best_err < 1e-10:
            final_err = compute_error(best_U, best_V, best_W)
            if final_err < 1e-10:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
=======
    # Try ranks from 48 to 53 to beat schoolbook rank 54
    for R in [48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 10:
            break
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        n_restarts = 20
        n_iters = 80
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(num_a, R) * 0.5
            V = np.random.randn(num_b, R) * 0.5
            W = np.random.randn(num_c, R) * 0.5
            
            for it in range(n_iters):
                if time.time() > deadline - 3:
                    break
                
                U = solve_for_U(V, W, target)
                V = solve_for_V(U, W, target)
                W = solve_for_W(U, V, target)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-14:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_U is not None and best_err < 1e-10:
            final_err = compute_error(best_U, best_V, best_W)
            if final_err < 1e-10:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
