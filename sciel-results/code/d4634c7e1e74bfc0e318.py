            # Solve Block 1
            M1 = np.eye(n) - h * A[0,0] * J_vals[0]
            if lu_prev is not None and iter_count == 1:
                try:
                    dY[0] = lu_solve((lu_prev[0], piv_prev[0]), -R[0])
                    # Check residual
                    res = np.max(np.abs((M1 @ dY[0]) - (-R[0])))
                    if res < 1e-8:
                        pass # Use it
                    else:
                        lu1, piv1 = lu_factor(M1)
                        dY[0] = lu_solve((lu1, piv1), -R[0])
                        lu_prev = (lu1, piv1)
                        piv_prev = piv1
                except:
                    lu1, piv1 = lu_factor(M1)
                    dY[0] = lu_solve((lu1, piv1), -R[0])
                    lu_prev = (lu1, piv1)
                    piv_prev = piv1
            else:
                lu1, piv1 = lu_factor(M1)
                dY[0] = lu_solve((lu1, piv1), -R[0])
                if iter_count == max_iter - 1: # Store for next step? No, store at end.
                    pass
