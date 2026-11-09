<<<<<<< SEARCH
                try:
                    W_new, _, _, _ = np.linalg.lstsq(C.T, T_mat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
=======
                try:
                    W_new, _, _, _ = np.linalg.lstsq(C.T, T_mat_W.T, rcond=None)
                    W = W_new.T
                    if np.any(np.isnan(W)):
                        break
                except:
                    pass
>>>>>>> REPLACE
