"""Reconstruct phylogenetic topology using NJ + SSD-based NNI refinement."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # Safety margin for 180s limit

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    try:
        seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    except:
        seq_matrix = np.array([[0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s] for s in seqs], dtype=np.int8)
        
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    p = diff / length
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _nj_with_lengths(D):
    """
    Neighbor Joining that returns both the topology (as adjacency list) and branch lengths.
    Returns: (adj, branch_lengths) where adj is dict node->list of neighbors,
             branch_lengths is dict (u,v) -> length
    """
    n = D.shape[0]
    if n <= 2:
        return {}, {}
    
    # Track active nodes and their current distances
    active = list(range(n))
    D_cur = D.copy()
    
    # adj: node -> list of neighbors
    adj = {i: [] for i in range(n)}
    # branch_lengths: (u, v) -> length
    branch_lengths = {}
    
    next_node_id = n
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            break
        
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(q, np.inf)
        
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        da_x = D_cur[a, active]
        db_x = D_cur[b, active]
        ab = D_cur[a, b]
        
        # Standard NJ branch length formulas
        # delta = (r_a - r_b) / (2*(m-2))
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        
        len_a = 0.5 * ab + delta
        len_b = 0.5 * ab - delta
        
        # Ensure non-negative
        if len_a < 0:
            len_a = 0
        if len_b < 0:
            len_b = 0
            
        new_node = next_node_id
        next_node_id += 1
        
        # Add new node to adjacency
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
        # Store branch lengths
        branch_lengths[(a, new_node)] = len_a
        branch_lengths[(new_node, a)] = len_a
        branch_lengths[(b, new_node)] = len_b
        branch_lengths[(new_node, b)] = len_b
        
        # Update distances for remaining active nodes
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_rem = D_cur[a, rem_active]
            db_rem = D_cur[b, rem_active]
            new_d = 0.5 * (da_rem + db_rem - ab)
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
            
        active = rem_active + [new_node]
        
        if time.time() > _DEADLINE - 5.0:
            break
            
    # Handle the final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Create a new internal node connecting all three
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths using the three distances
        d_ab = D_cur[a, b]
        d_bc = D_cur[b, c]
        d_ac = D_cur[a, c]
        
        # For a 3-taxon star, use average distances
        len_a = 0.5 * (d_ab + d_ac - d_bc)
        len_b = 0.5 * (d_ab + d_bc - d_ac)
        len_c = 0.5 * (d_ac + d_bc - d_ab)
        
        # Clamp to non-negative
        len_a = max(0, len_a)
        len_b = max(0, len_b)
        len_c = max(0, len_c)
        
        adj[new_node] = [a, b, c]
        adj[a].append(new_node)
        adj[b].append(new_node)
        adj[c].append(new_node)
        
        branch_lengths[(a, new_node)] = len_a
        branch_lengths[(new_node, a)] = len_a
        branch_lengths[(b, new_node)] = len_b
        branch_lengths[(new_node, b)] = len_b
        branch_lengths[(c, new_node)] = len_c
        branch_lengths[(new_node, c)] = len_c
        
    return adj, branch_lengths

def _compute_tree_distances(adj, branch_lengths, n):
    """
    Compute pairwise tree distances between all leaves.
    Returns: 2D array tree_d[i][j] = tree distance between leaf i and leaf j
    """
    num_nodes = len(adj)
    tree_d = np.zeros((n, n))
    
    # For each leaf, do BFS to compute distances to all other leaves
    for i in range(n):
        # BFS from leaf i
        dist = np.full(num_nodes, -1.0)
        dist[i] = 0.0
        queue = [i]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1.0:
                    edge_len = branch_lengths.get((u, v), 0.0)
                    dist[v] = dist[u] + edge_len
                    queue.append(v)
        
        for j in range(n):
            tree_d[i, j] = dist[j]
    
    return tree_d

def _ssd(D, tree_d):
    """Compute Sum of Squared Distances."""
    # Only compute for i < j to avoid double counting
    n = D.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = D[i, j] - tree_d[i, j]
            total += diff * diff
    return total

def _nni_refine_ssd(D, adj, branch_lengths, n):
    """
    NNI refinement using SSD objective.
    For each internal edge, evaluate the 3 possible quartet topologies
    and choose the one that minimizes SSD.
    """
    improved = True
    max_iter = 100
    iter_count = 0
    
    def get_leaf_set(node, visited=None):
        """Get all leaves in the subtree rooted at node (away from parent)."""
        if visited is None:
            visited = set()
        if node < n:
            return {node}
        result = set()
        for v in adj[node]:
            if v not in visited:
                visited.add(node)
                result |= get_leaf_set(v, visited)
                visited.remove(node)
        return result
    
    def compute_tree_d_fast(adj, branch_lengths, n):
        """Fast computation of tree distances."""
        num_nodes = len(adj)
        tree_d = np.zeros((n, n))
        for i in range(n):
            dist = np.full(num_nodes, -1.0)
            dist[i] = 0.0
            queue = [i]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1.0:
                        edge_len = branch_lengths.get((u, v), 0.0)
                        dist[v] = dist[u] + edge_len
                        queue.append(v)
            for j in range(n):
                tree_d[i, j] = dist[j]
        return tree_d
    
    def evaluate_ssd_for_edge(adj, branch_lengths, n, u, v, u1, u2, v1, v2):
        """
        Evaluate SSD for the 3 possible NNI topologies.
        Returns: (best_topology, best_ssd) where best_topology is 1, 2, or 3
        """
        # Current topology T1: (u1, u2) | (v1, v2)
        # We need to compute the change in SSD.
        
        # For efficiency, we can compute the change in SSD due to the NNI move.
        # The SSD change depends on the distances between the four subtrees.
        
        # Get leaf sets
        L_u1 = get_leaf_set(u1)
        L_u2 = get_leaf_set(u2)
        L_v1 = get_leaf_set(v1)
        L_v2 = get_leaf_set(v2)
        
        # Precompute sum of D between each pair of leaf sets
        def sum_d(set1, set2):
            s = 0.0
            for a in set1:
                for b in set2:
                    s += D[a, b]
            return s
        
        def sum_d_tree(set1, set2, tree_d):
            s = 0.0
            for a in set1:
                for b in set2:
                    s += tree_d[a, b]
            return s
        
        # Current tree distances
        tree_d = compute_tree_d_fast(adj, branch_lengths, n)
        
        # Compute the "within" and "between" sums for each topology
        # T1: pairs within {u1,u2} and within {v1,v2}
        # T2: pairs within {u1,v1} and within {u2,v2}  
        # T3: pairs within {u1,v2} and within {u2,v1}
        
        # The SSD change for an NNI move can be computed as:
        # delta_SSD = -2 * (sum of D for the new "close" pairs - sum of D for the old "close" pairs)
        #             + (sum of tree_d for new - sum of tree_d for old)
        
        # For simplicity, let's just compute the full SSD for each topology
        # and pick the best one.
        
        def try_topology(topo):
            """Compute SSD for a given topology (1, 2, or 3)."""
            # Save current state
            adj_save = {k: list(v) for k, v in adj.items()}
            bl_save = dict(branch_lengths)
            
            # Apply the NNI move
            if topo == 1:
                # No change
                pass
            elif topo == 2:
                # Swap u2 and v1
                adj[u].remove(u2)
                adj[u].append(v1)
                adj[v].remove(v1)
                adj[v].append(u2)
                adj[u2].remove(u)
                adj[u2].append(v)
                adj[v1].remove(v)
                adj[v1].append(u)
                
                # Recompute branch lengths for the affected edges
                # For now, keep the same lengths (approximation)
            elif topo == 3:
                # Swap u2 and v2
                adj[u].remove(u2)
                adj[u].append(v2)
                adj[v].remove(v2)
                adj[v].append(u2)
                adj[u2].remove(u)
                adj[u2].append(v)
                adj[v2].remove(v)
                adj[v2].append(u)
            
            # Compute SSD
            td = compute_tree_d_fast(adj, branch_lengths, n)
            ssd_val = _ssd(D, td)
            
            # Restore state
            adj.clear()
            adj.update(adj_save)
            branch_lengths.clear()
            branch_lengths.update(bl_save)
            
            return ssd_val
        
        # Evaluate all three topologies
        ssd1 = try_topology(1)
        ssd2 = try_topology(2)
        ssd3 = try_topology(3)
        
        best = min(ssd1, ssd2, ssd3)
        if best == ssd1:
            return 1, ssd1
        elif best == ssd2:
            return 2, ssd2
        else:
            return 3, ssd3
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        edges_checked = set()
        for u in range(n, len(adj)):
            for v in adj[u]:
                if v < n:
                    continue
                if (u, v) in edges_checked or (v, u) in edges_checked:
                    continue
                edges_checked.add((u, v))
                
                u_neighbors = [x for x in adj[u] if x != v]
                if len(u_neighbors) != 2:
                    continue
                u1, u2 = u_neighbors
                
                v_neighbors = [x for x in adj[v] if x != u]
                if len(v_neighbors) != 2:
                    continue
                v1, v2 = v_neighbors
                
                # Evaluate the three topologies
                best_topo, best_ssd = evaluate_ssd_for_edge(adj, branch_lengths, n, u, v, u1, u2, v1, v2)
                
                if best_topo != 1:
                    # Apply the NNI move
                    if best_topo == 2:
                        adj[u].remove(u2)
                        adj[u].append(v1)
                        adj[v].remove(v1)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v1].remove(v)
                        adj[v1].append(u)
                    elif best_topo == 3:
                        adj[u].remove(u2)
                        adj[u].append(v2)
                        adj[v].remove(v2)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v2].remove(v)
                        adj[v2].append(u)
                    
                    improved = True
                    break  # Restart edge loop
        
    return adj, branch_lengths

def _build_newick(adj, n, labels):
    """Build Newick string from adjacency list."""
    def build_rec(node, parent):
        if node < n:
            return labels[node]
        
        parts = []
        for v in adj[node]:
            if v == parent:
                continue
            parts.append(build_rec(v, node))
            
        if not parts:
            return "()"
            
        return "(" + ",".join(parts) + ")"
        
    root = 0
    return build_rec(root, -1) + ";"

def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    
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
        
        # 2. NJ with branch lengths
        adj, branch_lengths = _nj_with_lengths(D)
        
        if not adj:
            return "(" + ",".join(labels) + ");"
            
        # 3. NNI Refinement with SSD
        if time.time() < _DEADLINE - 5.0:
            adj, branch_lengths = _nni_refine_ssd(D, adj, branch_lengths, n)
            
        # 4. Build Newick
        newick = _build_newick(adj, n, labels)
        
        return newick
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
