   if _check_exact(U, V, W, n, m, p, r, tol=1e-8):
       # Convert to the required format
       U_full = np.hstack([U0[:, :1], U.T])
       V_full = np.hstack([V0[:, :1], V.T])
       W_full = np.hstack([W0[:, :1], W.T])
       
       if _check(U_full, V_full, W_full, n, m, p, r + 1, tol=1e-6):
           best_rank = r + 1
           ...
   