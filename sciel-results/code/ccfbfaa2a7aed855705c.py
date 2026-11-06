    def is_55_free_numpy(A, combs):
        # A is (n, n) bool
        # combs is (M, 5) int
        # We want to check if any comb is all 1s or all 0s in the induced subgraph edges.
        
        # Extract the 10 edges for each comb
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        # Indices into combs:
        # c0 = combs[:, 0], c1 = combs[:, 1], etc.
        
        # We can create a flat array of indices
        # idx_u = np.concatenate([combs[:,0], combs[:,0], ...])
        # idx_v = np.concatenate([combs[:,1], combs[:,2], ...])
        # edges = A[idx_u, idx_v]
        # edges is shape (10*M,)
        
        # Reshape to (M, 10)
        # edge_matrix = edges.reshape(M, 10)
        
        # Check for clique: all 1s
        # clique_mask = np.all(edge_matrix == 1, axis=1)
        
        # Check for independent: all 0s
        # indep_mask = np.all(edge_matrix == 0, axis=1)
        
        # If any True, return False
        return not (np.any(np.all(edge_matrix == 1, axis=1)) or np.any(np.all(edge_matrix == 0, axis=1)))
    