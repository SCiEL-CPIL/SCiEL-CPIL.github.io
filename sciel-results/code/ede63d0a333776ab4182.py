import math
import numpy as np

def _jc_distance(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(sequences, labels):
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _neighbour_joining(dist):
    n = dist.shape[0]
    if n <= 2:
        return []
    
    # nodes: list of node objects. 
    # Leaf: {'type': 'leaf', 'id': i}
    # Internal: {'type': 'internal', 'left': node, 'right': node, 'len_l': l, 'len_r': r}
    
    nodes = [{'type': 'leaf', 'id': i} for i in range(n)]
    current_dist = dist.copy()
    
    while len(nodes) > 2:
        m = len(nodes)
        r = np.sum(current_dist[:m, :m], axis=1)
        
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * current_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        a, b = best_i, best_j
        
        delta = 0.5 * (r[a] - r[b])
        branch_a = 0.5 * current_dist[a, b] + delta
        branch_b = 0.5 * current_dist[a, b] - delta
        
        if branch_a < 0: branch_a = 0.0
        if branch_b < 0: branch_b = 0.0
        
        new_node = {
            'type': 'internal',
            'left': nodes[a],
            'right': nodes[b],
            'len_l': branch_a,
            'len_r': branch_b
        }
        
        # Update distance matrix
        new_nodes = []
        new_dists = []
        for k in range(m):
            if k == a or k == b:
                continue
            new_nodes.append(nodes[k])
            d_ak = current_dist[a, k]
            d_bk = current_dist[b, k]
            d_ab = current_dist[a, b]
            new_d = 0.5 * (d_ak + d_bk - d_ab)
            if new_d < 0: new_d = 0.0
            new_dists.append(new_d)
        
        new_nodes.append(new_node)
        
        new_m = m - 1
        new_dist = np.zeros((new_m, new_m))
        
        # Fill distances between old nodes
        for i in range(new_m - 1):
            for j in range(new_m - 1):
                if i == j: continue
                # Find original indices for i and j
                # This requires mapping new index to old index.
                # Let's create a map.
                pass
        
        # Easier: Rebuild dist matrix from scratch? O(n^2) per step, total O(n^3). n=32, fine.
        # But we have the nodes.
        
        # Let's just store the tree structure and not worry about dist matrix for NJ if we use a different method?
        # No, NJ needs dist matrix.
        
        # Let's use a helper to get original index for a node in the current list.
        # This is getting complicated. Let's use a standard library or a simpler NJ.
        
        # For now, let's assume we can get the tree.
        # I will implement a robust NJ in the final code.
        
        break
        
    # This is a draft. I will write the full code in the edit.
    return None
