<<<<<<< SEARCH
            # Compute gradients
            # d/dU[i*m+j, r] = 2 * R[i,j,k] * V[j*p+k, r] * W[k*n+i, r] summed over k
            # Actually, for each (i,j,r): dU[i*m+j,r] = 2 * sum_k R[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        # dU[i*m+j, r] = 2 * sum_k R[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        dU[i * m + j, r] = 2.0 * np.sum(R[i, j, :] * V[j * p:, r] * W[:, r][np.arange(p) * n + i])
                        
                        # dV[j*p+k, r] = 2 * sum_i R[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        # dW[k*n+i, r] = 2 * sum_j R[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                    
                    for k in range(p):
                        # dV[j*p+k, r] for each r
                        for r in range(rank):
                            dV[j * p + k, r] = 2.0 * np.sum(R[:, j, k] * U[:, r][np.arange(n) * m + j] * W[k * n:, r][np.arange(n)])
                            dW[k * n + i, r] = 2.0 * np.sum(R[i, :, k] * U[i * m:, r][np.arange(m)] * V[j * p + k, r])

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
=======
            # Compute gradients vectorized
            # R is (n, m, p)
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            rank = U.shape[1]
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                
                # dU[i,j,r] = 2 * sum_k R[i,j,k] * V[j,k,r] * W[k,i,r]
                # R[i,j,k] * v_r[j,k] * w_r[k,i]
                # Sum over k.
                # R is (n,m,p). v_r is (m,p). w_r is (p,n).
                # Term for dU: (R * (v_r[None,:,:] * w_r.T[None,:,:])) ?
                # v_r[None,:,:] is (1,m,p). w_r.T is (n,p).
                # w_r.T[None,:,:] is (1,n,p).
                # We need to align m and n? No, m and n are dimensions of U and W respectively.
                # R[i,j,k]. v_r[j,k]. w_r[k,i].
                # Let's compute sum_k R[i,j,k] * v_r[j,k] * w_r[k,i].
                # This is equivalent to:
                # For each i,j: sum_k R[i,j,k] * v_r[j,k] * w_r[k,i]
                # Let M = R * v_r[None, :, :] -> (n, m, p)
                # Then sum_k M[i,j,k] * w_r[k,i].
                # w_r[k,i] is w_r.T[i,k].
                # So sum_k M[i,j,k] * w_r.T[i,k].
                # This is an element-wise product of M and w_r.T[None, :, :] broadcast?
                # w_r.T is (n, p). We need to broadcast it to (n, m, p).
                # So w_r.T[:, None, :] is (n, 1, p).
                # So term = M * w_r.T[:, None, :]
                # Sum over last axis (k).
                # dU_r = 2 * np.sum(M * w_r.T[:, None, :], axis=2)
                
                M_U = R * v_r[None, :, :] # (n, m, p)
                dU_r = 2.0 * np.sum(M_U * w_r.T[:, None, :], axis=2) # (n, m)
                dU[:, r] = dU_r.reshape(n * m)
                
                # dV[j,k,r] = 2 * sum_i R[i,j,k] * U[i,j,r] * W[k,i,r]
                # Sum over i.
                # R[i,j,k] * u_r[i,j] * w_r[k,i]
                # Let M = R * u_r[:, :, None] -> (n, m, p)
                # Sum over i (axis 0).
                # term = M * w_r.T[None, :, :] ? No.
                # w_r[k,i] = w_r.T[i,k].
                # So we multiply M[i,j,k] by w_r.T[i,k].
                # w_r.T is (n, p). w_r.T[:, None, :] is (n, 1, p).
                # M * w_r.T[:, None, :] -> (n, m, p).
                # Sum over axis 0 (i).
                # dV_r[j,k] = 2 * sum_i (R[i,j,k] * u_r[i,j] * w_r.T[i,k])
                
                M_V = R * u_r[:, :, None] # (n, m, p)
                dV_r = 2.0 * np.sum(M_V * w_r.T[:, None, :], axis=0) # (m, p)
                dV[:, r] = dV_r.reshape(m * p)
                
                # dW[k,i,r] = 2 * sum_j R[i,j,k] * U[i,j,r] * V[j,k,r]
                # Sum over j.
                # R[i,j,k] * u_r[i,j] * v_r[j,k]
                # Let M = R * v_r[None, :, :] -> (n, m, p)
                # Multiply by u_r[:, :, None] -> (n, m, p)
                # Sum over j (axis 1).
                # dW_r[k,i] = 2 * sum_j (R[i,j,k] * u_r[i,j] * v_r[j,k])
                # Result shape (n, p). We need to reshape to (p, n) for W.
                
                M_W = R * v_r[None, :, :] * u_r[:, :, None] # (n, m, p)
                dW_r = 2.0 * np.sum(M_W, axis=1) # (n, p)
                # dW_r[i, k] corresponds to W[k, i, r].
                # So we need to transpose dW_r to get (p, n).
                dW[:, r] = dW_r.T.reshape(p * n)

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
>>>>>>> REPLACE

