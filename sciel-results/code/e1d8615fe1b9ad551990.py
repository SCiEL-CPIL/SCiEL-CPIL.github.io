<<<<<<< SEARCH
    def validate_add(P, S_indices, dist_sets):
        """
        Check if P can be added.
        S_indices: np.array of indices in S
        dist_sets: list of sets, dist_sets[i] = set of dist2 values from S_indices[i] to other points in S
        Returns: True if valid
        """
        if len(S_indices) == 0:
            return True
        
        # Get distances from P to all points in S
        d_vals = dist2[P, S_indices]  # shape (|S|,)
        
        # Check uniqueness: no two points in S at same distance from P
        if len(np.unique(d_vals)) != len(d_vals):
            return False
        
        # Check each d_val is not in the corresponding point's dist set
        for i, idx_Q in enumerate(S_indices):
            if int(d_vals[i]) in dist_sets[i]:
                return False
        
        return True
=======
    def validate_batch(P_batch, S_indices, dist_sets):
        """
        Check if points in P_batch can be added.
        P_batch: np.array of candidate indices
        S_indices: np.array of indices in S
        dist_sets: list of sets
        Returns: boolean array of shape (len(P_batch),)
        """
        if len(S_indices) == 0:
            return np.ones(len(P_batch), dtype=bool)
        
        # C[i, j] = dist(P_batch[i], S_indices[j])
        C = dist2[P_batch[:, None], S_indices[None, :]]
        
        # 1. Check uniqueness per row
        sorted_C = np.sort(C, axis=1)
        unique_mask = np.all(sorted_C[:, 1:] != sorted_C[:, :-1], axis=1)
        
        if not np.any(unique_mask):
            return np.zeros(len(P_batch), dtype=bool)
            
        # 2. Check conflict with existing distances
        valid_mask = unique_mask.copy()
        
        for j, idx_Q in enumerate(S_indices):
            d_vals = C[:, j]
            D_Q = dist_sets[j]
            D_Q_arr = np.array(list(D_Q), dtype=np.int32)
            conflict_mask = np.isin(d_vals, D_Q_arr)
            valid_mask &= ~conflict_mask
            
            if not np.any(valid_mask):
                break
                
        return valid_mask
>>>>>>> REPLACE

