"""Maximum Likelihood phylogeny reconstruction under Jukes-Cantor."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj_initial_tree(sequences, labels):
    """Get initial tree using Neighbor Joining."""
    n_taxa = len(labels)
    if n_taxa <= 2:
        if n_taxa == 1:
            return labels[0]
        else:
            return f"({labels[0]},{labels[1]})"
    
    dist_matrix = np.zeros((n_taxa, n_taxa))
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    nodes = list(range(n_taxa))
    tree_dict = {i: labels[i] for i in range(n_taxa)}
    
    while len(nodes) > 2:
        n = len(nodes)
        total = {a: sum(dist_matrix[a, b] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * dist_matrix[a, b] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({tree_dict[a]},{tree_dict[b]})"
        new_node = len(nodes)
        rest = [x for x in nodes if x not in (a, b)]
        
        for x in rest:
            new_dist = 0.5 * (dist_matrix[a, x] + dist_matrix[b, x] - dist_matrix[a, b])
            dist_matrix[new_node, x] = new_dist
            dist_matrix[x, new_node] = new_dist
        
        nodes = rest + [new_node]
        tree_dict[new_node] = joined
    
    return f"({tree_dict[nodes[0]]},{tree_dict[nodes[1]]})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    return _nj_initial_tree(sequences, labels) + ";"
# EVOLVE-BLOCK-END
