<<<<<<< SEARCH
def _tree_to_newick(tree, labels):
    """Convert a parsed tree (nested tuples with string leaves) to Newick."""
    if isinstance(tree, str):
        return tree
    children = [_tree_to_newick(child, labels) for child in tree]
    return "(" + ",".join(children) + ")"
=======
def _tree_to_newick(tree, labels):
    """Convert a parsed tree (nested tuples with string leaves) to Newick."""
    if isinstance(tree, str):
        return tree
    children = [_tree_to_newick(child, labels) for child in tree]
    return "(" + ",".join(children) + ")"


def _insert_leaf_into_newick(newick, leaf_name, nearest_neighbor):
    """
    Insert a new leaf into a Newick string by attaching it as a sibling
    to the nearest_neighbor leaf.
    
    This is a simple but effective insertion: it finds the nearest_neighbor
    as a leaf in the Newick string and replaces it with (nearest_neighbor, new_leaf).
    
    Note: This only works correctly if nearest_neighbor is actually a leaf
    (not an internal node). For a more accurate insertion, one would need
    to find the best internal edge, but this is a reasonable approximation.
    """
    pattern = nearest_neighbor
    pos = 0
    while pos < len(newick):
        idx = newick.find(pattern, pos)
        if idx == -1:
            break
        
        # Check if this is a valid leaf position
        before_ok = (idx == 0 or newick[idx-1] in '([')
        after_idx = idx + len(pattern)
        after_ok = (after_idx == len(newick) or newick[after_idx] in ',)')
        
        if before_ok and after_ok:
            newick = newick[:idx] + f"({pattern},{leaf_name})" + newick[after_idx:]
            return newick
        
        pos = idx + 1
    
    # Fallback: append to the end (shouldn't happen)
    return newick
>>>>>>> REPLACE

Now let me update the FUAD function to use the proper insertion helper:

