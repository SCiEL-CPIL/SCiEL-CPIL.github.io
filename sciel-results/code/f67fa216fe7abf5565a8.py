<<<<<<< SEARCH
    # Try to reduce rank by eliminating zero columns or finding linear dependencies
    if time.time() < deadline - 20:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check for zero columns
        non_zero_cols = []
        for c in range(U_arr.shape[1]):
            if np.any(U_arr[:, c] != 0) and np.any(V_arr[:, c] != 0) and np.any(W_arr[:, c] != 0):
                non_zero_cols.append(c)
        
        if len(non_zero_cols) < U_arr.shape[1]:
            U = [[U_arr[r, c] for c in non_zero_cols] for r in range(U_arr.shape[0])]
            V = [[V_arr[r, c] for c in non_zero_cols] for r in range(V_arr.shape[0])]
            W = [[W_arr[r, c] for c in non_zero_cols] for r in range(W_arr.shape[0])]
=======
    # Try to reduce rank by eliminating zero columns and finding exact duplicates
    if time.time() < deadline - 30:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check for zero columns
        non_zero_cols = []
        for c in range(U_arr.shape[1]):
            if np.any(U_arr[:, c] != 0) and np.any(V_arr[:, c] != 0) and np.any(W_arr[:, c] != 0):
                non_zero_cols.append(c)
        
        if len(non_zero_cols) < U_arr.shape[1]:
            U_arr = U_arr[:, non_zero_cols]
            V_arr = V_arr[:, non_zero_cols]
            W_arr = W_arr[:, non_zero_cols]
        
        # Remove exact duplicate columns (same U, V, W pattern)
        if U_arr.shape[1] > 1:
            seen = {}
            keep = []
            for c in range(U_arr.shape[1]):
                # Create a hashable key from the column vectors
                u_key = tuple(np.round(U_arr[:, c], 8))
                v_key = tuple(np.round(V_arr[:, c], 8))
                w_key = tuple(np.round(W_arr[:, c], 8))
                full_key = (u_key, v_key, w_key)
                if full_key not in seen:
                    seen[full_key] = True
                    keep.append(c)
            
            if len(keep) < U_arr.shape[1]:
                U_arr = U_arr[:, keep]
                V_arr = V_arr[:, keep]
                W_arr = W_arr[:, keep]
        
        U = U_arr.tolist()
        V = V_arr.tolist()
        W = W_arr.tolist()
>>>>>>> REPLACE
