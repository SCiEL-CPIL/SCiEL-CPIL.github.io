            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(R):
                dU[:, r] += np.bincount(u_idx, weights=dU_sel[:, r], minlength=u_rows)
                dV[:, r] += np.bincount(v_idx, weights=dV_sel[:, r], minlength=v_rows)
                dW[:, r] += np.bincount(w_idx, weights=dW_sel[:, r], minlength=w_rows)
