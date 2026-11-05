"""Reconstruct phylogenetic topology using FastNJ with NNI refinement."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin for 270s limit

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    # Convert ACGT to integers: A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create a matrix of shape (n, length)
    seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    
    # Pairwise hamming distances
    # diff[i, j] = number of positions where seqs[i] != seqs[j]
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    # Convert to JC distance
    # p = diff / length
    # d = -0.75 * ln(1 - 4p/3)
    p = diff / length
    # Clip p to avoid log(0) or negative
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _fast_nj(D):
    """
    Fast Neighbor Joining algorithm.
    D: numpy array of distances, shape (n, n)
    Returns: list of joins (i, j)
    """
    n = D.shape[0]
    if n <= 2:
        return []
    
    # Active nodes
    active = list(range(n))
    # Distance matrix (we will update this)
    # Size it to accommodate new internal nodes (n + n-2)
    total_nodes = 2 * n
    D_cur = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    D_cur[:n, :n] = D
    
    joins = []
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            break
        
        # Calculate r[i] = sum of distances from i to all other active nodes
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        # Calculate q[i, j] = (m-2)*d[i,j] - r[i] - r[j]
        # We want to minimize q[i, j]
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Mask diagonal (i != j)
        np.fill_diagonal(q, np.inf)
        
        # Find min
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        joins.append((a, b))
        
        # Create new node
        new_node = n + len(joins) - 1
        
        # Update distance matrix
        # d[new, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_x = D_cur[a, rem_active]
            db_x = D_cur[b, rem_active]
            ab = D_cur[a, b]
            
            new_d = 0.5 * (da_x + db_x - ab)
            
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
        
        # Update active list
        active = rem_active + [new_node]
        
        # Check time
        if time.time() > _DEADLINE - 5.0:
            break
            
    return joins

def _build_topology(joins, n):
    """
    Build a topology representation from NJ joins.
    Returns:
    - children: dict mapping node -> [left, right]
    - parent: dict mapping node -> parent
    - root: node ID
    """
    children = {}
    parent = {}
    
    # Initialize leaves
    for i in range(n):
        children[i] = []
        
    # Process joins
    for k, (a, b) in enumerate(joins):
        new_node = n + k
        children[new_node] = [a, b]
        parent[a] = new_node
        parent[b] = new_node
        
    # Find root
    root = None
    for i in range(n + len(joins)):
        if i not in parent:
            root = i
            break
            
    if root is None:
        # Fallback
        root = n + len(joins) - 1
        
    return children, parent, root

def _nni_refine(D, children, parent, root, labels):
    """
    Perform NNI refinement to minimize Sum of Squared Distances (SSD) or similar quartet-based metric.
    """
    n = len(labels)
    num_joins = len([k for k in children if k >= n])
    
    # Precompute clades for each node (sets of leaf indices)
    clades = {}
    def get_clade(node):
        if node in clades:
            return clades[node]
        if node < n:
            clades[node] = frozenset([node])
        else:
            l, r = children[node]
            clades[node] = get_clade(l) | get_clade(r)
        return clades[node]
        
    # Get list of internal edges
    # An internal edge is (parent[node], node) where node is internal
    internal_nodes = [i for i in range(n, n + num_joins) if i in children and len(children[i]) == 2]
    
    improved = True
    max_iter = 50  # Limit iterations to prevent long runtimes
    
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Build adjacency list for the current tree
        adj = {}
        for i in range(n + num_joins):
            adj[i] = []
        for i, chs in children.items():
            for c in chs:
                adj[i].append(c)
                adj[c].append(i)
        
        # Iterate over internal edges
        for node in internal_nodes:
            if time.time() > _DEADLINE - 1.0:
                break
                
            p = parent.get(node)
            if p is None:
                continue
                
            # The edge is (p, node)
            u = p
            v = node
            
            # Get neighbors of u excluding v
            u_neighbors = [x for x in adj[u] if x != v]
            if len(u_neighbors) != 2:
                continue # Root or non-binary?
            u1, u2 = u_neighbors
            
            # Get neighbors of v excluding u
            v_neighbors = [x for x in adj[v] if x != u]
            if len(v_neighbors) != 2:
                continue
            v1, v2 = v_neighbors
            
            # Get leaves in each subtree
            L_u1 = get_clade(u1)
            L_u2 = get_clade(u2)
            L_v1 = get_clade(v1)
            L_v2 = get_clade(v2)
            
            # Calculate average distance between the sets
            def avg_dist(set1, set2):
                if not set1 or not set2:
                    return 0
                s = 0.0
                cnt = 0
                for a in set1:
                    for b in set2:
                        s += D[a, b]
                        cnt += 1
                return s / cnt if cnt > 0 else 0
            
            # Quartet scores
            # T1 (Current): (L_u1, L_u2) | (L_v1, L_v2)
            score_t1 = avg_dist(L_u1, L_u2) + avg_dist(L_v1, L_v2)
            
            # T2: (L_u1, L_v1) | (L_u2, L_v2)
            score_t2 = avg_dist(L_u1, L_v1) + avg_dist(L_u2, L_v2)
            
            # T3: (L_u1, L_v2) | (L_u2, L_v1)
            score_t3 = avg_dist(L_u1, L_v2) + avg_dist(L_u2, L_v1)
            
            best_score = min(score_t1, score_t2, score_t3)
            
            if best_score < score_t1:
                # Perform NNI move
                if best_score == score_t2:
                    # Swap u2 and v1
                    children[u] = [x for x in children[u] if x != u2] + [v1]
                    children[v] = [x for x in children[v] if x != v1] + [u2]
                    
                    parent[u2] = v
                    parent[v1] = u
                    
                    # Invalidate clade cache for affected nodes
                    # Simplest: clear whole cache
                    clades.clear()
                    
                elif best_score == score_t3:
                    # Swap u2 and v2
                    children[u] = [x for x in children[u] if x != u2] + [v2]
                    children[v] = [x for x in children[v] if x != v2] + [u2]
                    
                    parent[u2] = v
                    parent[v2] = u
                    
                    clades.clear()
                    
                improved = True
                break  # Restart loop after modification

    return children, parent, root

def _build_newick(children, parent, root, labels):
    """Build Newick string from topology."""
    def build(node):
        if node < len(labels):
            return labels[node]
        l, r = children[node]
        return f"({build(l)},{build(r)})"
    
    return build(root) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    try:
        # 1. Distance Matrix
        D = _jc_distance_matrix(sequences, labels)
        
        # 2. FastNJ
        joins = _fast_nj(D)
        
        # 3. Build Topology
        if not joins:
             return "(" + ",".join(labels) + ");"
             
        children, parent, root = _build_topology(joins, n)
        
        # 4. NNI Refinement
        if time.time() < _DEADLINE - 5.0:
            children, parent, root = _nni_refine(D, children, parent, root, labels)
            
        # 5. Build Newick
        newick = _build_newick(children, parent, root, labels)
        
        return newick
        
    except Exception as e:
        # Fallback: Star tree
        return "(" + ",".join(labels) + ");"
