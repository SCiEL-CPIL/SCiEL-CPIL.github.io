"""UPGMA via SciPy for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist


def _seq_to_array(sequences, labels):
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for lab in labels:
        s = sequences[lab]
        arr = np.array([char_map[c] for c in s], dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs, axis=0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    # Convert to array
    data = _seq_to_array(sequences, labels)
    
    # Compute condensed distance matrix using Hamming distance
    # pdist with metric='hamming'
    condensed_dist = pdist(data, metric='hamming')
    
    # Perform UPGMA linkage
    # method='average' is UPGMA
    Z = linkage(condensed_dist, method='average', metric='euclidean')
    
    # Convert to tree
    tree = to_tree(Z)
    
    # Convert to Newick
    # to_tree returns a scipy tree object.
    # We can convert it to Newick using tree.as_string() or similar?
    # scipy.cluster.hierarchy.to_tree returns a Tree object.
    # It has a method to get Newick?
    # Actually, `to_tree` returns a `Tree` object which can be converted to Newick using `tree.to_string()`?
    # No, it's a `scipy.cluster.hierarchy._tree.Tree` object.
    # It doesn't have a direct Newick export in all versions.
    # We can traverse it manually.
    
    def _tree_to_newick(node, labels):
        if node.is_leaf():
            return labels[node.id]
        else:
            left = _tree_to_newick(node.get_left(), labels)
            right = _tree_to_newick(node.get_right(), labels)
            return f"({left},{right})"
            
    newick = _tree_to_newick(tree, labels)
    return newick + ";"
# EVOLVE-BLOCK-END
