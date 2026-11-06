<<<<<<< SEARCH
def _nni_refine_likelihood(newick, arr, n, seq_len, labels):
    """NNI refinement using JC likelihood. Returns best newick string."""
    global _labels_global
    _labels_global = labels
    
    if n < 4:
        return newick
    
    if _time_left() < 30.0:
        return newick
    
    # Compute initial likelihood
    tree = _parse_newick(newick)
    best_ll = _compute_jc_likelihood(tree, arr, n, seq_len)
    best_newick = newick
    
    # Iterate NNI moves
    max_iterations = 50
    for iteration in range(max_iterations):
        if _time_left() < 15.0:
            break
        
        # Find all internal edges to try NNI
        tree = _parse_newick(best_newick)
        
        # Get all internal nodes
        internal_nodes = []
        def collect_internal_nodes(node):
            if node[0] == 'internal':
                collect_internal_nodes(node[1])
                collect_internal_nodes(node[2])
                internal_nodes.append(node)
        
        collect_internal_nodes(tree)
        
        improved = False
        for node in internal_nodes:
            if _time_left() < 10.0:
                break
            
            # Try the three possible NNI rearrangements
            # Current: ((A,B),(C,D))
            # Alternative 1: ((A,C),(B,D))
            # Alternative 2: ((A,D),(B,C))
            
            left = node[1]
            right = node[2]
            
            # Check if both children are internal
            if left[0] != 'internal' or right[0] != 'internal':
                continue
            
            # Get the four subtrees
            a = left[1]
            b = left[2]
            c = right[1]
            d = right[2]
            
            # Try alternative 1: ((A,C),(B,D))
            alt1 = ('internal', ('internal', a, c), ('internal', b, d), '')
            alt1_newick = _to_newick(alt1)
            alt1_ll = _compute_jc_likelihood(alt1, arr, n, seq_len)
            
            if alt1_ll > best_ll + 1e-6:
                best_ll = alt1_ll
                best_newick = alt1_newick
                improved = True
                break
            
            # Try alternative 2: ((A,D),(B,C))
            alt2 = ('internal', ('internal', a, d), ('internal', b, c), '')
            alt2_newick = _to_newick(alt2)
            alt2_ll = _compute_jc_likelihood(alt2, arr, n, seq_len)
            
            if alt2_ll > best_ll + 1e-6:
                best_ll = alt2_ll
                best_newick = alt2_newick
                improved = True
                break
        
        if not improved:
            break
    
    return best_newick
=======
def _nni_refine_likelihood(newick, arr, n, seq_len, labels):
    """NNI refinement using JC likelihood. Returns best newick string."""
    global _labels_global
    _labels_global = labels
    
    if n < 4:
        return newick
    
    if _time_left() < 30.0:
        return newick
    
    # Compute initial likelihood
    tree = _parse_newick(newick)
    best_ll = _compute_jc_likelihood(tree, arr, n, seq_len)
    best_newick = newick
    
    # Iterate NNI moves
    max_iterations = 30
    for iteration in range(max_iterations):
        if _time_left() < 15.0:
            break
        
        # Parse current best tree
        tree = _parse_newick(best_newick)
        
        # Get all internal nodes
        internal_nodes = []
        def collect_internal_nodes(node):
            if node[0] == 'internal':
                collect_internal_nodes(node[1])
                collect_internal_nodes(node[2])
                internal_nodes.append(node)
        
        collect_internal_nodes(tree)
        
        improved = False
        for node in internal_nodes:
            if _time_left() < 10.0:
                break
            
            # Try NNI rearrangements
            left = node[1]
            right = node[2]
            
            # Check if both children are internal
            if left[0] != 'internal' or right[0] != 'internal':
                continue
            
            # Get the four subtrees
            a = left[1]
            b = left[2]
            c = right[1]
            d = right[2]
            
            # Try alternative 1: ((A,C),(B,D))
            alt1 = ('internal', ('internal', a, c, ''), ('internal', b, d, ''), '')
            alt1_ll = _compute_jc_likelihood(alt1, arr, n, seq_len)
            
            if alt1_ll > best_ll + 1e-4:
                best_ll = alt1_ll
                best_newick = _to_newick(alt1)
                improved = True
                break
            
            # Try alternative 2: ((A,D),(B,C))
            alt2 = ('internal', ('internal', a, d, ''), ('internal', b, c, ''), '')
            alt2_ll = _compute_jc_likelihood(alt2, arr, n, seq_len)
            
            if alt2_ll > best_ll + 1e-4:
                best_ll = alt2_ll
                best_newick = _to_newick(alt2)
                improved = True
                break
        
        if not improved:
            break
    
    return best_newick
>>>>>>> REPLACE

