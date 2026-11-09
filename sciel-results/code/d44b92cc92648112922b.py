                    U_reshaped = U.reshape(nn, mm, r)
                    V_reshaped = V.reshape(mm, pp, r)
                    W_reshaped = W.reshape(pp, nn, r)
                    
                    C = np.einsum('ir,jr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
