"""Phylogenetic topology recovery using Neighbor Joining with potential SPR refinement under Jukes-Cantor."""
import math
import numpy as np


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    num_sites = len(seq_list[0])
    
    # Compute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = sum(1 for x, y in zip(seq_list[i], seq_list[j]) if x != y)
            p = diffs / num_sites
            d = _jc_distance(p)
            dist[i][j] = d
            dist[j][i] = d
    
    # Neighbor Joining
    active = list(range(n))
    # tree_parts maps node id to its Newick representation
    tree_parts = {i: f"t{i}" for i in range(n)}
    next_id = n
    
    # We need to maintain a distance matrix that includes internal nodes
    # Let's use a dictionary for distances
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist[i][j]
                dist_dict[(j, i)] = dist[i][j]
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        r = {}
        for a in active:
            r[a] = sum(dist_dict.get((a, b), 0.0) for b in active if b != a)
        
        # Find best pair
        best_a, best_b = None, None
        best_q = float('inf')
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                d_ab = dist_dict.get((a, b), 0.0)
                q = (m - 2) * d_ab - r[a] - r[b]
                if q < best_q:
                    best_q = q
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        d_ab = dist_dict.get((a, b), 0.0)
        
        # Create new internal node
        new_node = next_id
        next_id += 1
        
        # Update distances to other nodes
        for c in active:
            if c == a or c == b:
                continue
            d_ac = dist_dict.get((a, c), 0.0)
            d_bc = dist_dict.get((b, c), 0.0)
            new_d = 0.5 * (d_ac + d_bc - d_ab)
            dist_dict[(new_node, c)] = new_d
            dist_dict[(c, new_node)] = new_d
        
        # Update tree
        tree_parts[new_node] = f"({tree_parts[a]},{tree_parts[b]})"
        
        # Remove a and b, add new_node
        active.remove(a)
        active.remove(b)
        active.append(new_node)
    
    # Final three nodes
    remaining = active
    if len(remaining) == 3:
        final_tree = f"({tree_parts[remaining[0]]},{tree_parts[remaining[1]]},{tree_parts[remaining[2]]})"
    else:
        final_tree = ",".join(tree_parts[x] for x in remaining)
    
    return final_tree + ";"
