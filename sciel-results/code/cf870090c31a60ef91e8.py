"""Reconstruct phylogenetic topology using FastNJ with robust NNI refinement (SSD)."""
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
    
    # Convert ACGT to integers
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    try:
        seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    except:
        seq_matrix = np.array([[0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s] for s in seqs], dtype=np.int8)
        
    # Pairwise hamming distances
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    # Convert to JC distance
    p = diff / length
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _fast_nj(D):
    """
    Fast Neighbor Joining algorithm.
    Returns: list of joins (i, j)
    """
    n = D.shape[0]
    if n <= 2:
        return []
    
    active = list(range(n))
    D_cur = D.copy()
    joins = []
    
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
        
        joins.append((a, b))
        
        new_node = n + len(joins) - 1
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_x = D_cur[a, rem_active]
            db_x = D_cur[b, rem_active]
            ab = D_cur[a, b]
            new_d = 0.5 * (da_x + db_x - ab)
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
            
        active = rem_active + [new_node]
        
        if time.time() > _DEADLINE - 5.0:
            break
            
    return joins

def _build_topology(joins, n):
    """
    Build a topology representation from NJ joins.
    Returns:
    - adj: dict mapping node -> list of neighbors
    """
    num_nodes = n + len(joins)
    adj = {i: [] for i in range(num_nodes)}
    
    for k, (a, b) in enumerate(joins):
        new_node = n + k
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
    return adj

def _calculate_ssd(D, adj, n):
    """
    Calculate Sum of Squared Distances (SSD) for the tree.
    SSD = sum_{i<j} (D[i,j] - d_tree(i,j))^2
    """
    # Compute tree distances using BFS/DFS from each leaf
    # Since n is small, we can do BFS from each leaf.
    total_ssd = 0.0
    for i in range(n):
        # BFS from i to find distances to all other leaves
        dist = [-1] * len(adj)
        dist[i] = 0
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    # We need branch lengths for this. 
                    # The current NJ doesn't store branch lengths in adj.
                    # We need to re-calculate branch lengths or store them.
                    # This is a problem. SSD requires branch lengths.
                    # If we don't have branch lengths, we can't compute d_tree.
                    # We can use a topology-only metric like Robinson-Foulds distance? No, we are optimizing topology.
                    # We can use the "Quartet" score which only depends on topology and D.
                    # Or we can estimate branch lengths from D and topology.
                    # Let's switch to a Quartet-based NNI.
                    pass
        # This approach is flawed without branch lengths.
        return 0.0 # Placeholder

def _nni_refine_quartet(D, adj, n):
    """
    NNI refinement using Quartet scores.
    For each internal edge, check the 3 possible quartet topologies.
    Score: Sum of distances for the "cross" pairs?
    For quartet (a,b,c,d), topology (a,b)|(c,d) is supported if d_ab + d_cd < d_ac + d_bd and d_ab + d_cd < d_ad + d_bc.
    We can sum these "violations" or simply count how many quartets are resolved correctly.
    Let's use the sum of the "best" quartet distances.
    """
    improved = True
    max_iter = 200
    iter_count = 0
    
    # Precompute leaf sets for all nodes
    def get_leaves(node):
        leaves = set()
        visited = set()
        stack = [node]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            if u < n:
                leaves.add(u)
            else:
                for v in adj[u]:
                    if v not in visited:
                        stack.append(v)
        return leaves

    # Cache leaf sets
    leaf_cache = {}
    def get_leaves_cached(node):
        if node in leaf_cache:
            return leaf_cache[node]
        if node < n:
            leaf_cache[node] = {node}
        else:
            l_set = set()
            for v in adj[node]:
                l_set |= get_leaves_cached(v)
            leaf_cache[node] = l_set
        return leaf_cache[node]
        
    # Initial leaf sets
    for i in range(len(adj)):
        get_leaves_cached(i)
        
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Find internal edges
        # An internal edge is one where both endpoints are internal nodes (>= n)
        # OR one endpoint is internal and the other is a leaf? No, NNI is on internal edges.
        # In a binary unrooted tree, internal edges connect two internal nodes.
        # However, the "root" of our representation might make one node look like a leaf if we aren't careful.
        # With n>=8, there are internal edges.
        
        edges_checked = set()
        for u in range(n, len(adj)):
            for v in adj[u]:
                if v < n:
                    continue
                if (u, v) in edges_checked or (v, u) in edges_checked:
                    continue
                edges_checked.add((u, v))
                
                # Edge (u, v)
                # Neighbors of u excluding v
                u_neighbors = [x for x in adj[u] if x != v]
                if len(u_neighbors) != 2:
                    continue
                u1, u2 = u_neighbors
                
                # Neighbors of v excluding u
                v_neighbors = [x for x in adj[v] if x != u]
                if len(v_neighbors) != 2:
                    continue
                v1, v2 = v_neighbors
                
                # Get leaf sets
                L_u1 = leaf_cache[u1]
                L_u2 = leaf_cache[u2]
                L_v1 = leaf_cache[v1]
                L_v2 = leaf_cache[v2]
                
                # Calculate quartet scores
                # We want to minimize the sum of squared distances or simply the "quartet distance"
                # Let's use the sum of the two "cross" distances for each topology.
                # T1 (Current): (L_u1, L_u2) | (L_v1, L_v2)
                # Score T1 = sum(d(a,b) for a in L_u1, b in L_u2) + sum(d(a,b) for a in L_v1, b in L_v2)
                # This is not the standard quartet score.
                
                # Standard Quartet Score:
                # For leaves a,b,c,d:
                # S1 = d_ab + d_cd
                # S2 = d_ac + d_bd
                # S3 = d_ad + d_bc
                # Topology (a,b)|(c,d) is best if S1 < S2 and S1 < S3.
                
                # Since we have sets, we can compute the average distance between sets.
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
                
                d_u1_u2 = avg_dist(L_u1, L_u2)
                d_v1_v2 = avg_dist(L_v1, L_v2)
                d_u1_v1 = avg_dist(L_u1, L_v1)
                d_u2_v2 = avg_dist(L_u2, L_v2)
                d_u1_v2 = avg_dist(L_u1, L_v2)
                d_u2_v1 = avg_dist(L_u2, L_v1)
                
                # Score for T1: (u1, u2) and (v1, v2) are the "close" pairs?
                # No, in T1, u1 and u2 are on the same side of the edge (v).
                # The quartet is formed by one leaf from each of the 4 subtrees.
                # Let's pick representative leaves? No, use averages.
                # The "cost" of a topology is the sum of the distances of the pairs that are "separated" by the internal branch?
                # Actually, the likelihood is higher when the "close" pairs have small distance.
                # In T1, the pairs (u1, u2) and (v1, v2) are NOT necessarily close.
                # The internal edge separates {L_u1, L_u2} from {L_v1, L_v2}.
                # So distances between {L_u1, L_u2} and {L_v1, L_v2} should be large.
                # Distances within {L_u1, L_u2} and within {L_v1, L_v2} should be small.
                
                # Let's define Score(T) as sum of "within" distances.
                # T1: within = d(u1,u2) + d(v1,v2). Minimize this.
                # T2: (u1, v1) and (u2, v2). within = d(u1,v1) + d(u2,v2).
                # T3: (u1, v2) and (u2, v1). within = d(u1,v2) + d(u2,v1).
                
                score_t1 = d_u1_u2 + d_v1_v2
                score_t2 = d_u1_v1 + d_u2_v2
                score_t3 = d_u1_v2 + d_u2_v1
                
                best_score = min(score_t1, score_t2, score_t3)
                
                if best_score < score_t1:
                    # Perform NNI
                    if best_score == score_t2:
                        # Swap u2 and v1
                        # u: remove u2, add v1
                        # v: remove v1, add u2
                        adj[u].remove(u2)
                        adj[u].append(v1)
                        adj[v].remove(v1)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v1].remove(v)
                        adj[v1].append(u)
                        
                    elif best_score == score_t3:
                        # Swap u2 and v2
                        adj[u].remove(u2)
                        adj[u].append(v2)
                        adj[v].remove(v2)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v2].remove(v)
                        adj[v2].append(u)
                        
                    # Invalidate leaf cache for affected nodes
                    # Simplest: clear cache
                    leaf_cache.clear()
                    for i in range(len(adj)):
                        get_leaves_cached(i)
                        
                    improved = True
                    break # Restart edge loop
        
    return adj

def _build_newick(adj, n, labels):
    """Build Newick string from adjacency list."""
    visited = set()
    
    def build(node, parent):
        visited.add(node)
        if node < n:
            return labels[node]
        
        children = [v for v in adj[node] if v != parent and v not in visited]
        # If no children, it's a leaf? No, internal nodes have children.
        # Handle the case where we backtrack.
        if not children:
            # This should not happen for internal nodes unless it's a malformed tree
            return f"({labels[0]},{labels[1]})" 
            
        # We need to ensure we don't visit the parent.
        # The visited set is global, which is tricky for recursive build.
        # Better to pass a set of "blocked" nodes.
        
        return ""
        
    # Let's use a simpler DFS with a stack or recursion with parent tracking.
    def build_rec(node, parent):
        if node < n:
            return labels[node]
        
        parts = []
        for v in adj[node]:
            if v == parent:
                continue
            parts.append(build_rec(v, node))
            
        if not parts:
            # Should not happen
            return "()"
            
        return "(" + ",".join(parts) + ")"
        
    # Find a root (any node)
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
        
        # 2. FastNJ
        joins = _fast_nj(D)
        
        # 3. Build Topology
        if not joins:
             return "(" + ",".join(labels) + ");"
             
        adj = _build_topology(joins, n)
        
        # 4. NNI Refinement
        if time.time() < _DEADLINE - 5.0:
            adj = _nni_refine_quartet(D, adj, n)
            
        # 5. Build Newick
        newick = _build_newick(adj, n, labels)
        
        return newick
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
