<<<<<<< SEARCH
            # Compute current tensor
            # T_curr[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is expensive to compute fully every step.
            # We can compute gradients analytically.
            
            # Error
            T_curr = np.zeros((n, m, p))
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                T_curr += u_r[:, :, None] * v_r[None, :, :] * w_r[None, None, :] # Broadcasting might be wrong shape
                # u_r is (n, m), v_r is (m, p), w_r is (p, n)
                # We need T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is a tensor contraction.
                # np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
                
            T_curr = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
=======
            # Compute current tensor using einsum (efficient and correct)
            T_curr = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
>>>>>>> REPLACE
