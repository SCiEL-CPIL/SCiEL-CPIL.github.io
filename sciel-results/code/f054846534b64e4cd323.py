<<<<<<< SEARCH
def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_nj, splits_nj = _nj(d_full, labels)
    
    # Quartet-based approach with vectorized scoring
    quartet_topologies, quartet_weights, d_matrix = _quartet_ml_scores_vectorized(arr)
    split_weights = _quartet_to_splits_vectorized(quartet_topologies, quartet_weights, n)
    
    # Build tree from weighted splits
    tree_quartet = _build_tree_from_weighted_splits(split_weights, n, labels)
    
    # Compare DPP scores
    dpp_nj = _dpp_score_vectorized(splits_nj, arr, n, L)
    dpp_quartet = _dpp_score_vectorized(_get_splits_from_tree(tree_quartet, labels), arr, n, L)
    
    # Choose the better tree
    if dpp_quartet < dpp_nj:
        best_tree = tree_quartet
    else:
        best_tree = tree_nj
    
    # DPP refinement if time permits
    if _time_left() > 30.0:
        refined_tree = _dpp_refine(best_tree, arr, labels, n, L)
        dpp_refined = _dpp_score_vectorized(_get_splits_from_tree(refined_tree, labels), arr, n, L)
        dpp_best = _dpp_score_vectorized(_get_splits_from_tree(best_tree, labels), arr, n, L)
        if dpp_refined < dpp_best:
            best_tree = refined_tree
    
    return best_tree
=======
def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Precompute diff matrix (used multiple times)
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_nj, splits_nj = _nj(d_full, labels)
    
    # Quartet-based approach with vectorized scoring
    quartet_topologies, quartet_weights, d_matrix = _quartet_ml_scores_vectorized(arr)
    split_weights = _quartet_to_splits_vectorized(quartet_topologies, quartet_weights, n)
    
    # Build tree from weighted splits
    tree_quartet = _build_tree_from_weighted_splits(split_weights, n, labels)
    
    # Compare DPP scores
    dpp_nj = _dpp_score_vectorized(splits_nj, arr, n, L, diff_matrix)
    dpp_quartet = _dpp_score_vectorized(_get_splits_from_tree(tree_quartet, labels), arr, n, L, diff_matrix)
    
    # Choose the better tree
    if dpp_quartet < dpp_nj:
        best_tree = tree_quartet
    else:
        best_tree = tree_nj
    
    # DPP refinement if time permits
    if _time_left() > 20.0:
        refined_tree = _dpp_refine(best_tree, arr, labels, n, L)
        dpp_refined = _dpp_score_vectorized(_get_splits_from_tree(refined_tree, labels), arr, n, L, diff_matrix)
        dpp_best = _dpp_score_vectorized(_get_splits_from_tree(best_tree, labels), arr, n, L, diff_matrix)
        if dpp_refined < dpp_best:
            best_tree = refined_tree
    
    return best_tree
>>>>>>> REPLACE

