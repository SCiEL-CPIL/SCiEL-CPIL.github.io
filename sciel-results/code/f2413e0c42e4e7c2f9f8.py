# Inside simulate_annealing
            # Fast update for U[row, col]:
            # Slice = T_calc[row, :, :]
            Slice = np.einsum('t,bt,ct->bc', U[row, :], V, W)
            Slice_Err = Slice - T_target[row, :, :]
            
            V_col = V[:, col]
            W_col = W[:, col]
            Sum1 = np.sum(Slice_Err * np.outer(V_col, W_col))
            Sum2 = np.sum(V_col**2 * W_col**2)
            
            Delta_E = 2 * d * Sum1 + d**2 * Sum2
