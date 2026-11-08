<<<<<<< SEARCH
            # Check residual
            resid = np.linalg.norm(T_keep @ x - b)
=======
            # Check residual
            # x is shape (n_keep,), T_keep is shape (n_keep, n*m*p)
            # We want linear combination of rows of T_keep weighted by x
            # Result shape should be (n*m*p,)
            resid = np.linalg.norm(x @ T_keep - b)
>>>>>>> REPLACE
