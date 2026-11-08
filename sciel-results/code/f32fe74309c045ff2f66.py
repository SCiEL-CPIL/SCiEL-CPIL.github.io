<<<<<<< SEARCH
                    U_vec = U[0 : n*m : m, col] # i=0 -> j, i=1 -> m+j ...
                    # U index is i*m + j.
                    # i=0 -> j, i=1 -> m+j, ...
                    U_vec = U[j : j + n*m : m]
                    W_val = W[k * n + i, col] # Wait, i is variable.
                    # W index is k*n + i.
                    # For fixed k, i varies.
                    # W_vec = W[k*n : k*n + n, col]
                    W_vec = W[k * n : k * n + n, col]
                    
                    Delta_P = U_vec * W_vec # Shape (n,)
=======
                    # U index is i*m + j.
                    # i=0 -> j, i=1 -> m+j, ...
                    # We need U[i*m+j, col] for i in 0..n-1
                    U_vec = U[j : j + n*m : m, col]
                    # W index is k*n + i.
                    # For fixed k, i varies.
                    # W_vec = W[k*n : k*n + n, col]
                    W_vec = W[k * n : k * n + n, col]
                    
                    Delta_P = U_vec * W_vec # Shape (n,)
>>>>>>> REPLACE
