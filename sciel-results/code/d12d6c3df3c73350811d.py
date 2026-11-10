"""Phylogenetic topology recovery using UPGMA initialization and Minimum Evolution NNI refinement."""
import math
import numpy as np

def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _upgma_tree(dist, n):
    """
    Build a tree using UPGMA.
    Returns a tree dict similar to before.
    """
    if n < 2:
        return {'num_nodes': 1, 'children': [[]], 'parent': [-1], 'labels': [0]}
        
    # Clusters
    # Each cluster has: id, members (list of leaf indices), height
    clusters = [{'id': i, 'members': [i], 'height': 0.0} for i in range(n)]
    
    # Tree structure
    # We will build the tree as we merge
    # Node IDs: 0 to n-1 are leaves. n to ... are internal.
    
    tree = {
        'num_nodes': n,
        'children': [[] for _ in range(n)],
        'parent': [-1 for _ in range(n)],
        'labels': list(range(n))
    }
    
    active = list(range(n))
    next_id = n
    
    while len(active) > 1:
        # Find closest pair
        min_d = float('inf')
        i_idx, j_idx = 0, 1
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ci = clusters[active[i]]
                cj = clusters[active[j]]
                # Average distance
                d = 0.0
                cnt = 0
                for mi in ci['members']:
                    for mj in cj['members']:
                        d += dist[mi, mj]
                        cnt += 1
                d /= cnt
                if d < min_d:
                    min_d = d
                    i_idx, j_idx = i, j
                    
        ci = clusters[active[i_idx]]
        cj = clusters[active[j_idx]]
        
        # Merge
        new_members = ci['members'] + cj['members']
        new_height = min_d / 2.0 # UPGMA height
        
        # Create new internal node
        new_id_val = next_id
        next_id += 1
        
        # Add to tree
        tree['children'].append([ci['id'], cj['id']])
        tree['parent'].append(-1) # Will be set later
        tree['labels'].append(-1)
        
        # Update parents of children
        tree['parent'][ci['id']] = new_id_val
        tree['parent'][cj['id']] = new_id_val
        
        tree['num_nodes'] += 1
        
        # New cluster
        new_cluster = {'id': new_id_val, 'members': new_members, 'height': new_height}
        
        # Update active list
        active.pop(j_idx)
        active.pop(i_idx)
        clusters.append(new_cluster)
        active.append(new_id_val)
        
    # The last active cluster is the root
    root_id = active[0]
    tree['parent'][root_id] = -1
    
    return tree

def _solve_min_evolution(tree, dist_matrix, n_leaves):
    """
    Compute optimal branch lengths and total length for a given topology.
    """
    edges = []
    for i in range(tree['num_nodes']):
        if tree['parent'][i] != -1:
            edges.append(i)
            
    num_edges = len(edges)
    if num_edges == 0:
        return 0.0, {}
        
    pairs = []
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    
    edge_index_map = {e: idx for idx, e in enumerate(edges)}
    
    A = np.zeros((num_edges, num_pairs))
    
    for p_idx, (i, j) in enumerate(pairs):
        path_i = set()
        curr = i
        while curr != -1:
            path_i.add(curr)
            curr = tree['parent'][curr]
            
        lca = None
        curr = j
        while curr != -1:
            if curr in path_i:
                lca = curr
                break
            curr = tree['parent'][curr]
            
        if lca is None:
            continue
            
        curr = i
        while curr != lca:
            if curr in edge_index_map:
                A[edge_index_map[curr], p_idx] = 1
            curr = tree['parent'][curr]
            
        curr = j
        while curr != lca:
            if curr in edge_index_map:
                A[edge_index_map[curr], p_idx] = 1
            curr = tree['parent'][curr]
            
    d_vec = np.array([dist_matrix[i, j] for i, j in pairs])
    
    try:
        b, residuals, rank, s = np.linalg.lstsq(A, d_vec, rcond=None)
    except:
        return float('inf'), {}
        
    b = np.maximum(b, 0)
    total_length = np.sum(b)
    
    branch_lengths = {}
    for idx, e in enumerate(edges):
        branch_lengths[e] = b[idx]
        
    return total_length, branch_lengths

def _nni_optimize(tree, dist, n_leaves, max_iter=20):
    """
    Perform NNI optimization to minimize total length.
    """
    for _ in range(max_iter):
        improved = False
        edges_to_check = []
        for v in range(tree['num_nodes']):
            if tree['parent'][v] != -1 and len(tree['children'][v]) == 2:
                u = tree['parent'][v]
                if len(tree['children'][u]) == 2:
                    edges_to_check.append((u, v))
                    
        for u, v in edges_to_check:
            children_u = tree['children'][u]
            children_v = tree['children'][v]
            
            if children_u[0] == v:
                w = children_u[1]
            else:
                w = children_u[0]
                
            x, y = children_v
            
            cost_curr, _ = _solve_min_evolution(tree, dist, n_leaves)
            
            orig_cu = tree['children'][u][:]
            orig_cv = tree['children'][v][:]
            
            # Candidate 1: ((x, w), y)
            tree['children'][v] = [x, w]
            tree['children'][u] = [v, y]
            cost_c1, _ = _solve_min_evolution(tree, dist, n_leaves)
            
            tree['children'][u] = orig_cu
            tree['children'][v] = orig_cv
            
            # Candidate 2: ((y, w), x)
            tree['children'][v] = [y, w]
            tree['children'][u] = [v, x]
            cost_c2, _ = _solve_min_evolution(tree, dist, n_leaves)
            
            tree['children'][u] = orig_cu
            tree['children'][v] = orig_cv
            
            best_cost = cost_curr
            best_move = None
            
            if cost_c1 < best_cost:
                best_cost = cost_c1
                best_move = 1
            if cost_c2 < best_cost:
                best_cost = cost_c2
                best_move = 2
                
            if best_move:
                improved = True
                if best_move == 1:
                    tree['children'][v] = [x, w]
                    tree['children'][u] = [v, y]
                else:
                    tree['children'][v] = [y, w]
                    tree['children'][u] = [v, x]
                    
        if not improved:
            break
            
    return tree

def _tree_to_newick(tree, labels):
    def get_newick(node):
        if len(tree['children'][node]) == 0:
            return labels[tree['labels'][node]]
        else:
            left, right = tree['children'][node]
            return f"({get_newick(left)},{get_newick(right)})"
            
    root = 0
    for i in range(tree['num_nodes']):
        if tree['parent'][i] == -1:
            root = i
            break
            
    return get_newick(root) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    dist = _build_distance_matrix(sequences, labels)
    
    tree = _upgma_tree(dist, n)
    
    tree = _nni_optimize(tree, dist, n)
    
    newick = _tree_to_newick(tree, labels)
    
    return newick
