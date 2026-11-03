<<<<<<< SEARCH
    # Initial solution: greedy construction
    # Sort facilities by sum of a (row sums) and locations by sum of b (col sums)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost
=======
    # Initial solution: Spectral Bisection based ordering
    # This creates a 1D embedding of the facility graph and location graph,
    # then aligns them. This often yields a much better starting point than greedy.
    
    def spectral_order(matrix, mode='flow'):
        """
        Compute a 1D ordering of nodes based on the Fiedler vector of the graph Laplacian.
        mode='flow': We want to cluster high-flow nodes together.
                     Laplacian L = D - A. Fiedler vector separates the graph.
                     Sorting by Fiedler vector values creates an ordering where
                     adjacent nodes in the order have strong connections (if we assume
                     the graph is roughly path-like or grid-like, which QAP instances often are).
        mode='dist': We want to cluster low-distance nodes together.
                     We can use the Laplacian of the distance matrix directly?
                     No, distance matrix is not a graph adjacency in the standard sense for clustering
                     unless we convert it. 
                     Standard approach for QAP: 
                     1. Compute eigenvector of L_A (facility flow).
                     2. Compute eigenvector of L_B (location distance).
                     3. Sort both by their respective eigenvectors.
                     4. The resulting permutation maps the "left" side of the facility spectrum 
                        to the "left" side of the location spectrum.
        """
        # Construct Laplacian
        # For flow matrix A, large values mean strong connection.
        # L = D - A.
        # For distance matrix B, small values mean strong connection.
        # To cluster close locations, we can treat similarity S = max(B) - B or just use -B?
        # Actually, the Fiedler vector of L_B (where B is distance) will separate the graph into
        # two parts with minimal sum of distances between them? 
        # Let's stick to a robust method:
        # 1. Compute L_A = D_A - A.
        # 2. Compute L_B = D_B - B.
        # 3. Get Fiedler vectors v_A and v_B.
        # 4. Sort indices by v_A and v_B.
        
        deg = matrix.sum(axis=1)
        L = np.diag(deg) - matrix
        # Symmetrize to handle numerical errors
        L = (L + L.T) / 2.0
        
        # Compute eigenvalues and eigenvectors
        # We only need the second smallest eigenvalue's vector.
        # scipy.sparse.linalg.eigsh is efficient, but for 256x256, dense is fine and faster to call.
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(L)
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
        # The smallest is usually 0 (constant vector) if the graph is connected.
        fiedler_vec = eigvecs[:, 1]
        
        # Sort indices by the values in the Fiedler vector
        order = np.argsort(fiedler_vec, kind='stable')
        return order

    # Get spectral orderings
    fac_order = spectral_order(a, 'flow')
    loc_order = spectral_order(b, 'dist')
    
    # Map facilities in fac_order to locations in loc_order
    # p[fac_order[i]] = loc_order[i]
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost
>>>>>>> REPLACE

I will also add a second initialization strategy using the row sums (greedy) and pick the better one, or just use the spectral one as it's generally superior for structured instances like `tai256c`. To be safe and robust, I'll compute both and start with the better one.

