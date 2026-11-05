"""Reconstruct phylogenetic topology using NJ + quartet-score NNI refinement."""
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
    seq_matrix = np.array([[mapping.get(c, 0) for c in s] for s in seqs], dtype=np.int8)
        
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
    
    active = list(range(n))
    D_cur = D.copy()
    
    adj = {i: [] for i in range(n)}
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
        
        da_x = D_cur[a, active]
        db_x = D_cur[b, active]
        ab = D_cur[a, b]
        
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        
        len_a = 0.5 * ab + delta
        len_b = 0.5 * ab - delta
        
        if len_a < 0:
            len_a = 0
        if len_b < 0:
            len_b = 0
            
        new_node = next_node_id
        next_node_id += 1
        
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
        branch_lengths[(a, new_node)] = len_a
        branch_lengths[(new_node, a)] = len_a
        branch_lengths[(b, new_node)] = len_b
        branch_lengths[(new_node, b)] = len_b
        
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
            
    if len(active) == 3:
        a, b, c = active
        new_node = next_node_id
        next_node_id += 1
        
        d_ab = D_cur[a, b]
        d_bc = D_cur[b, c]
        d_ac = D_cur[a, c]
        
        len_a = 0.5 * (d_ab + d_ac - d_bc)
        len_b = 0.5 * (d_ab + d_bc - d_ac)
        len_c = 0.5 * (d_ac + d_bc - d_ab)
        
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

def _get_leaf_set(node, n, adj, visited=None):
    """Get all leaves in the subtree rooted at node (away from parent)."""
    if visited is None:
        visited = set()
    if node < n:
        return frozenset([node])
    result = set()
    for v in adj[node]:
        if v not in visited:
            visited.add(node)
            result |= _get_leaf_set(v, n, adj, visited)
            visited.remove(node)
    return frozenset(result)

def _compute_leaf_sets_for_edges(n, adj):
    """
    For each internal edge, precompute the four leaf sets (A, B, C, D)
    where A,B are on one side and C,D on the other.
    Returns dict: (u,v) -> (A, B, C, D) as frozensets
    """
    edge_leaf_sets = {}
    num_nodes = len(adj)
    
    for u in range(n, num_nodes):
        for v in adj[u]:
            if v < n:
                continue
            key = (min(u, v), max(u, v))
            if key in edge_leaf_sets:
                continue
                
            # Get neighbors of u other than v
            u_neighbors = [x for x in adj[u] if x != v]
            if len(u_neighbors) != 2:
                continue
            u1, u2 = u_neighbors
            
            # Get neighbors of v other than u
            v_neighbors = [x for x in adj[v] if x != u]
            if len(v_neighbors) != 2:
                continue
            v1, v2 = v_neighbors
            
            # Compute leaf sets
            A = _get_leaf_set(u1, n, adj)
            B = _get_leaf_set(u2, n, adj)
            C = _get_leaf_set(v1, n, adj)
            D = _get_leaf_set(v2, n, adj)
            
            edge_leaf_sets[key] = (A, B, C, D, u, v, u1, u2, v1, v2)
    
    return edge_leaf_sets

def _quartet_score_nni(D, n, adj, edge_leaf_sets):
    """
    Compute quartet score for all internal edges and find the best NNI move.
    Quartet score: for each quartet (a,b,c,d), the tree supports (a,b|c,d) if
    the path a-b and c-d don't cross. We count how many quartets are "correct"
    relative to the distance matrix.
    
    For NNI on edge with subtrees A,B,C,D:
    - Current tree supports A|B vs C|D (i.e., (A,B) and (C,D) are the two sides)
    - Alternative 1: (A,C) and (B,D)  
    - Alternative 2: (A,D) and (B,C)
    
    The quartet score change for switching from current to alternative can be
    computed from the sums of distances between the four sets.
    """
    best_move = None
    best_improvement = 0.0
    
    for key, (A, B, C, D, u, v, u1, u2, v1, v2) in edge_leaf_sets.items():
        if time.time() > _DEADLINE - 3.0:
            break
        
        # Compute sum of distances between each pair of sets
        def sum_dist(set1, set2):
            if not set1 or not set2:
                return 0.0
            s = 0.0
            for a in set1:
                for b in set2:
                    s += D[a, b]
            return s
        
        # Sums for current topology: pairs within {A,B} and within {C,D}
        # The "cost" of a topology is the sum of distances for the "crossing" pairs
        # For quartet score, we want to maximize the number of concordant quartets.
        
        # For each pair of leaves (a in A, b in B, c in C, d in D):
        # Current tree: a,b are close (same side), c,d are close (same side)
        # The quartet is (a,b|c,d)
        # Alternative 1: (a,c|b,d) - a,c close
        # Alternative 2: (a,d|b,c) - a,d close
        
        # The quartet score for the current tree on this edge counts quartets
        # where the tree's split agrees with the "best" quartet from distances.
        
        # For efficiency, compute the total "support" for each pairing:
        # support_AB = sum over a in A, b in B, c in C, d in D of [D(a,b)+D(c,d) < D(a,c)+D(b,d) and D(a,b)+D(c,d) < D(a,d)+D(b,c)]
        # This is O(|A|*|B|*|C|*|D|) which is too slow.
        
        # Better: use the fact that for a fixed (a,b) pair, the contribution depends
        # on sums over C and D.
        
        # Actually, let's use a simpler metric: the "quartet score" can be approximated
        # by looking at the sum of distances for the three possible pairings of the
        # four sets. The best pairing minimizes the sum of within-pair distances.
        
        # Sum for pairing (AB, CD): sum_{a in A, b in B} D(a,b) + sum_{c in C, d in D} D(c,d)
        # Sum for pairing (AC, BD): sum_{a in A, c in C} D(a,c) + sum_{b in B, d in D} D(b,d)
        # Sum for pairing (AD, BC): sum_{a in A, d in D} D(a,d) + sum_{b in B, c in C} D(b,c)
        
        # The current tree has pairing (AB, CD). We want to switch to the pairing
        # with the smallest sum (most "compatible" with the distance matrix).
        
        s_AB = sum_dist(A, B)
        s_CD = sum_dist(C, D)
        s_AC = sum_dist(A, C)
        s_BD = sum_dist(B, D)
        s_AD = sum_dist(A, D)
        s_BC = sum_dist(B, C)
        
        current_score = s_AB + s_CD
        alt1_score = s_AC + s_BD  # pairing (AC, BD)
        alt2_score = s_AD + s_BC  # pairing (AD, BC)
        
        # We want to minimize the sum, so the best is the minimum
        best_score = min(current_score, alt1_score, alt2_score)
        
        if best_score < current_score - 1e-10:
            improvement = current_score - best_score
            if improvement > best_improvement:
                best_improvement = improvement
                if alt1_score == best_score:
                    best_move = (u, v, u1, u2, v1, v2, 1)  # switch to (AC, BD)
                else:
                    best_move = (u, v, u1, u2, v1, v2, 2)  # switch to (AD, BC)
    
    return best_move

def _apply_nni(adj, branch_lengths, u, v, u1, u2, v1, v2, topo):
    """Apply an NNI move."""
    if topo == 1:
        # Switch from (AB, CD) to (AC, BD)
        # Remove edges: u-u2, v-v1
        # Add edges: u-v1, v-u2
        adj[u].remove(u2)
        adj[u].append(v1)
        adj[v].remove(v1)
        adj[v].append(u2)
        adj[u2].remove(u)
        adj[u2].append(v)
        adj[v1].remove(v)
        adj[v1].append(u)
    elif topo == 2:
        # Switch from (AB, CD) to (AD, BC)
        adj[u].remove(u2)
        adj[u].append(v2)
        adj[v].remove(v2)
        adj[v].append(u2)
        adj[u2].remove(u)
        adj[u2].append(v)
        adj[v2].remove(v)
        adj[v2].append(u)

def _nni_refine_quartet(D, n, adj, branch_lengths):
    """
    NNI refinement using quartet score.
    """
    improved = True
    max_iter = 200
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
        
        # Precompute leaf sets for all internal edges
        edge_leaf_sets = _compute_leaf_sets_for_edges(n, adj)
        
        # Find the best NNI move
        best_move = _quartet_score_nni(D, n, adj, edge_leaf_sets)
        
        if best_move is not None:
            u, v, u1, u2, v1, v2, topo = best_move
            _apply_nni(adj, branch_lengths, u, v, u1, u2, v1, v2, topo)
            improved = True
    
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
        
    # Find a root - use an internal node
    root = None
    for node in adj:
        if node >= n:
            root = node
            break
    if root is None:
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
            
        # 3. NNI Refinement with quartet score
        if time.time() < _DEADLINE - 5.0:
            adj, branch_lengths = _nni_refine_quartet(D, n, adj, branch_lengths)
            
        # 4. Build Newick
        newick = _build_newick(adj, n, labels)
        
        return newick
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
