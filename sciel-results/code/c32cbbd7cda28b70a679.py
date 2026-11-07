remaining = deadline - time.time()
per_tree_limit = max(0.5, min(5.0, remaining / (n_trees - idx_in_order)))
