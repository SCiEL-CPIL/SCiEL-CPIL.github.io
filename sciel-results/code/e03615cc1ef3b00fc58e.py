    # Build NJ tree
    tree_tuple = _nj_build(dist, labels)
    tree = _tuple_to_tree(tree_tuple)
