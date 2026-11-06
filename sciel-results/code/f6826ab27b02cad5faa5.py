import time
import math
import numpy as np
import random
from collections import defaultdict

_START_TIME = None
_TIME_LIMIT = 255.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

# ============================================================================
# Distance-based NJ
# ============================================================================

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distance matrix."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels], dtype=np.int8)
    
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists

def _nj_build(dists, labels):
    """
    Standard Neighbor Joining.
    Returns a tree represented as a list of edges: [(node_a, node_b, length)]
    plus a leaf-to-index mapping.
    Tree is represented with node IDs: 0..n-1 are leaves (by label index),
    n..2n-3 are internal nodes.
    """
    n = len(labels)
    if n < 3:
        return None
    
    # Active nodes: list of node IDs
    # dist_matrix: distance between active nodes
    # node_type: 'leaf' or 'internal'
    
    active = list(range(n))  # Initially all leaves
    dist = dists.copy()
    node_count = n  # Next available node ID for internal nodes
    
    # Track the tree structure
    edges = []  # list of (node_a, node_b, length)
    
    # For each active node, track which "original" leaves it contains
    # (needed for split extraction later, but not for NJ itself)
    
    while len(active) > 2:
        m = len(active)
        if m <= 2:
            break
        
        r = np.sum(dist, axis=1)
        q = (m - 2) * dist - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q), q.shape)
        i, j = min_idx
        
        node_i = active[i]
        node_j = active[j]
        
        d_ij = dist[i, j]
        
        alpha = (r[i] - r[j]) / (2.0 * (m - 2))
        len_i = 0.5 * d_ij + alpha
        len_j = 0.5 * d_ij - alpha
        
        # Create new internal node
        new_node = node_count
        node_count += 1
        
        # Add edges
        edges.append((node_i, new_node, max(0.0, len_i)))
        edges.append((node_j, new_node, max(0.0, len_j)))
        
        # Update active list and distance matrix
        # Remove i and j, add new_node at the end
        new_active = [active[k] for k in range(m) if k != i and k != j]
        new_active.append(new_node)
        
        m_new = len(new_active)
        new_dist = np.zeros((m_new, m_new))
        
        # Map old positions to new positions
        # New node is at position m_new - 1
        for p1 in range(m_new):
            for p2 in range(p1 + 1, m_new):
                if p1 == m_new - 1 or p2 == m_new - 1:
                    # Distance to new node
                    other = p2 if p1 == m_new - 1 else p1
                    # Find the old position of 'other' in the old active list
                    other_node = new_active[other]
                    old_pos_other = None
                    for old_p in range(m):
                        if active[old_p] == other_node:
                            old_pos_other = old_p
                            break
                    
                    d_ix = dist[i, old_pos_other]
                    d_jx = dist[j, old_pos_other]
                    d_new = 0.5 * (d_ix + d_jx - d_ij)
                    d_new = max(0.0, d_new)
                    new_dist[p1, p2] = d_new
                    new_dist[p2, p1] = d_new
                else:
                    # Both are old nodes
                    node_a = new_active[p1]
                    node_b = new_active[p2]
                    old_pos_a = None
                    old_pos_b = None
                    for old_p in range(m):
                        if active[old_p] == node_a:
                            old_pos_a = old_p
                        if active[old_p] == node_b:
                            old_pos_b = old_p
                    new_dist[p1, p2] = dist[old_pos_a, old_pos_b]
                    new_dist[p2, p1] = dist[old_pos_a, old_pos_b]
        
        active = new_active
        dist = new_dist
    
    # Connect the last two
    if len(active) == 2:
        node_a = active[0]
        node_b = active[1]
        d_ab = dist[0, 1]
        # Create a final internal node to connect them
        new_node = node_count
        node_count += 1
        edges.append((node_a, new_node, 0.5 * d_ab))
        edges.append((node_b, new_node, 0.5 * d_ab))
    
    return edges, node_count

# ============================================================================
# Tree representation using adjacency list
# ============================================================================

def _build_adjacency(edges, node_count, n_leaves):
    """Build adjacency list from edge list."""
    adj = defaultdict(list)
    for a, b, length in edges:
        adj[a].append((b, length))
        adj[b].append((a, length))
    return adj

def _get_splits(adj, n_leaves):
    """
    Extract all non-trivial splits from an unrooted tree.
    Returns a set of frozensets, where each frozenset represents the smaller side of a split.
    """
    all_leaves = set(range(n_leaves))
    splits = set()
    
    def dfs(node, parent):
        """Return the set of leaf indices in the subtree rooted at 'node' (away from 'parent')."""
        if node < n_leaves:
            return {node}
        
        leaf_set = set()
        for neighbor, length in adj[node]:
            if neighbor != parent:
                leaf_set.update(dfs(neighbor, node))
        
        return leaf_set
    
    # For each internal node, compute the leaves on one side
    # We need to avoid double-counting: each edge defines a split
    # Process each edge once
    
    visited_edges = set()
    
    def get_split_for_edge(a, b):
        """Get the split defined by removing edge (a, b)."""
        # Get leaves on a's side (not going through b)
        leaves_a = _get_leaves_from(a, b, adj, n_leaves)
        return frozenset(leaves_a)
    
    for node in adj:
        for neighbor, length in adj[node]:
            edge_key = (min(node, neighbor), max(node, neighbor))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            
            leaves_side = _get_leaves_from(node, neighbor, adj, n_leaves)
            if 0 < len(leaves_side) < n_leaves:
                # Non-trivial split
                if len(leaves_side) <= n_leaves // 2:
                    splits.add(frozenset(leaves_side))
                else:
                    splits.add(frozenset(all_leaves - leaves_side))
    
    return splits

def _get_leaves_from(start, exclude, adj, n_leaves):
    """Get all leaf nodes reachable from 'start' without going through 'exclude'."""
    visited = set()
    stack = [start]
    leaves = set()
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        
        if node < n_leaves:
            leaves.add(node)
        
        for neighbor, length in adj[node]:
            if neighbor != exclude and neighbor not in visited:
                stack.append(neighbor)
    
    return leaves

# ============================================================================
# Maximum Likelihood with Felsenstein's Pruning Algorithm
# ============================================================================

class MLTree:
    """
    Tree for ML optimization.
    Uses a rooted binary tree representation internally.
    """
    def __init__(self, adj, n_leaves):
        self.n_leaves = n_leaves
        self.node_count = max(adj.keys()) + 1 if adj else n_leaves
        # Rebuild as rooted tree
        # Root at node 0 (a leaf)
        self.root = 0
        # Build parent and children from adjacency
        self.parent = {}
        self.children = {}
        self.edge_length = {}  # (node) -> length of edge from parent
        
        self._root_tree()
    
    def _root_tree(self):
        """Root the tree at node 0."""
        visited = set()
        stack = [(0, -1)]
        while stack:
            node, par = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            self.parent[node] = par
            
            if par >= 0:
                self.edge_length[node] = 0.0  # Will be set below
            
            if par not in self.children:
                self.children[par] = []
            if par >= 0:
                self.children[par].append(node)
            
            for neighbor, length in self.adj[node]:
                if neighbor != par:
                    self.edge_length[neighbor] = length
                    stack.append((neighbor, node))
        
        # Store adjacency for reference
        self.adj = adj
    
    def get_leaves(self):
        return list(range(self.n_leaves))
    
    def get_internal_nodes(self):
        return [n for n in range(self.node_count) if n >= self.n_leaves]
    
    def get_neighbors(self, node):
        return [n for n, l in self.adj[node]]
    
    def get_edge_length(self, a, b):
        for neighbor, length in self.adj[a]:
            if neighbor == b:
                return length
        return 0.0

def _felsenstein_pruning(tree_root, seqs, n_sites):
    """
    Felsenstein's pruning algorithm for Jukes-Cantor model.
    Returns log-likelihood of the tree given the sequences.
    
    tree_root: node ID of the root
    seqs: numpy array of shape (n_taxa, n_sites) with values 0-3
    n_sites: number of sites
    """
    n_leaves = len(seqs)
    
    # We need the tree structure. Let's use the adjacency list approach.
    # Actually, let's use a simpler representation: 
    # The tree is given by adjacency list with edge lengths.
    # We'll do a post-order traversal from the root.
    
    # First, build parent/children from adjacency
    parent = {}
    children = defaultdict(list)
    edge_len = {}
    
    visited = set()
    stack = [(tree_root, -1)]
    order = []
    while stack:
        node, par = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        order.append(node)
        
        if par >= 0:
            children[par].append(node)
        
        for neighbor, length in _MLTree_adj.get(node, []):
            if neighbor != par:
                edge_len[neighbor] = length
                stack.append((neighbor, node))
    
    # Process in post-order (reverse of pre-order)
    # P(node) = probability vector at node
    # For leaves: P(leaf) = one-hot for the observed base
    # For internal: P(node) = sum over children of P(child) * T(length)
    
    # JC transition matrix for time t:
    # P(AA) = P(CG) etc.
    # P_same = 1/4 + 3/4 * exp(-4t/3)
    # P_diff = 1/4 - 1/4 * exp(-4t/3)
    
    # Store probability vectors: shape (n_sites, 4)
    prob = {}
    
    for node in reversed(order):
        if node < n_leaves:
            # Leaf: set probability based on observed base
            # prob[node] is a (n_sites, 4) array
            vec = np.zeros((n_sites, 4), dtype=np.float64)
            for site in range(n_sites):
                base = seqs[node, site]
                vec[site, base] = 1.0
            prob[node] = vec
        else:
            # Internal node: multiply children's probs by transition matrices
            # P(node) = sum over children c of [P(c) * T(len(c))]
            # But we need to be careful: the transition matrix applies to the edge from node to child
            
            n_children = len(children[node])
            if n_children == 0:
                # Should not happen for internal nodes in a proper tree
                prob[node] = np.ones((n_sites, 4), dtype=np.float64) / 4.0
                continue
            
            if n_children == 1:
                child = children[node][0]
                t = edge_len.get(child, 0.0)
                p_same = 0.25 + 0.75 * math.exp(-4.0 * t / 3.0)
                p_diff = 0.25 - 0.25 * math.exp(-4.0 * t / 3.0)
                
                # P(node) = P(child) * T
                # T[i,j] = P(base j at parent | base i at child)
                # For JC, T is symmetric: T[i,i] = p_same, T[i,j] = p_diff for i != j
                # So P(node)[j] = sum_i P(child)[i] * T[i,j]
                # = p_same * P(child)[j] + p_diff * sum_{i != j} P(child)[i]
                # = p_same * P(child)[j] + p_diff * (1 - P(child)[j])
                
                pc = prob[child]
                pn = p_same * pc + p_diff * (1.0 - pc)
                prob[node] = pn
            else:
                # Multiple children (shouldn't happen in binary tree, but handle it)
                # In a binary unrooted tree rooted at a leaf, internal nodes have 2 children
                # except the root which may have 1 child (if root is leaf) or 2 children
                # Let's just handle 2 children properly
                c1 = children[node][0]
                c2 = children[node][1] if len(children[node]) > 1 else None
                
                t1 = edge_len.get(c1, 0.0)
                p_same1 = 0.25 + 0.75 * math.exp(-4.0 * t1 / 3.0)
                p_diff1 = 0.25 - 0.25 * math.exp(-4.0 * t1 / 3.0)
                pc1 = prob[c1]
                pn1 = p_same1 * pc1 + p_diff1 * (1.0 - pc1)
                
                if c2 is not None:
                    t2 = edge_len.get(c2, 0.0)
                    p_same2 = 0.25 + 0.75 * math.exp(-4.0 * t2 / 3.0)
                    p_diff2 = 0.25 - 0.25 * math.exp(-4.0 * t2 / 3.0)
                    pc2 = prob[c2]
                    pn2 = p_same2 * pc2 + p_diff2 * (1.0 - pc2)
                    
                    # P(node) = pn1 * pn2 element-wise? No.
                    # Actually, for an internal node with 2 children, the probability is:
                    # P(node)[j] = sum_{i1, i2} P(child1)[i1] * P(child2)[i2] * T1[i1,j] * T2[i2,j]
                    # This is not simply element-wise product.
                    
                    # For JC model, we can use the fact that:
                    # P(node)[j] = (1/4) * (sum_i pn1[i] * T1[i,j]) * ... 
                    
                    # Actually, let me think about this differently.
                    # The standard Felsenstein algorithm processes the tree bottom-up.
                    # At each internal node, we compute the likelihood contribution.
                    
                    # Let me use a different approach: store the likelihood vector
                    # where vector[j] = P(data below | base at node = j)
                    
                    # For a leaf: vector[j] = 1 if j == observed_base, else 0
                    # For an internal node with children c1, c2:
                    # vector[j] = sum_{i1} T1[i1,j] * pc1[i1] * sum_{i2} T2[i2,j] * pc2[i2]
                    
                    # For JC, T[i,j] = p_same if i==j, p_diff otherwise
                    # So sum_i T[i,j] * pc[i] = p_same * pc[j] + p_diff * (1 - pc[j])
                    
                    # So vector[j] = (p_same1 * pc1[j] + p_diff1 * (1-pc1[j])) * (p_same2 * pc2[j] + p_diff2 * (1-pc2[j]))
                    
                    # Wait, that's the element-wise product of pn1 and pn2!
                    # Let me verify: 
                    # vector[j] = sum_{i1,i2} T1[i1,j] * pc1[i1] * T2[i2,j] * pc2[i2]
                    # = (sum_{i1} T1[i1,j] * pc1[i1]) * (sum_{i2} T2[i2,j] * pc2[i2])
                    # = pn1[j] * pn2[j]
                    
                    # Yes! So it is element-wise product.
                    prob[node] = pn1 * pn2
                else:
                    prob[node] = pn1
    
    # Root likelihood
    # P(root) is a vector. The total likelihood is:
    # LL = sum_j P(root)[j] * (1/4)   [since base frequencies are uniform under JC]
    # Actually, the root prior is uniform: P(base) = 1/4
    # So LL = sum_j prob[root][j] * 0.25
    
    root_prob = prob[tree_root]
    # Average over sites
    site_ll = np.sum(root_prob * 0.25, axis=1)
    total_ll = np.sum(np.log(np.maximum(site_ll, 1e-300)))
    
    return total_ll

# Need to fix: _MLTree_adj is not defined. Let me restructure.

# Let me rewrite the ML part more cleanly.

def _compute_ml(adj, tree_root, seqs, n_sites):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm.
    adj: dict mapping node -> list of (neighbor, length)
    tree_root: root node ID
    seqs: numpy array (n_taxa, n_sites), values 0-3
    """
    n_leaves = len(seqs)
    
    # Build parent/children structure
    parent = {}
    children = defaultdict(list)
    edge_len = {}
    
    visited = set()
    stack = [(tree_root, -1)]
    order = []
    while stack:
        node, par = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        order.append(node)
        
        if par >= 0:
            children[par].append(node)
        
        for neighbor, length in adj.get(node, []):
            if neighbor != par:
                edge_len[neighbor] = length
                stack.append((neighbor, node))
    
    # Process in post-order
    prob = {}
    
    for node in reversed(order):
        if node < n_leaves:
            # Leaf
            vec = np.zeros((n_sites, 4), dtype=np.float64)
            for site in range(n_sites):
                base = int(seqs[node, site])
                vec[site, base] = 1.0
            prob[node] = vec
        else:
            kids = children[node]
            if len(kids) == 0:
                prob[node] = np.ones((n_sites, 4), dtype=np.float64) / 4.0
            elif len(kids) == 1:
                c = kids[0]
                t = edge_len.get(c, 0.0)
                exp_val = math.exp(-4.0 * t / 3.0)
                p_same = 0.25 + 0.75 * exp_val
                p_diff = 0.25 - 0.25 * exp_val
                pc = prob[c]
                prob[node] = p_same * pc + p_diff * (1.0 - pc)
            else:
                # Two children
                c1, c2 = kids[0], kids[1]
                t1 = edge_len.get(c1, 0.0)
                t2 = edge_len.get(c2, 0.0)
                exp1 = math.exp(-4.0 * t1 / 3.0)
                exp2 = math.exp(-4.0 * t2 / 3.0)
                ps1 = 0.25 + 0.75 * exp1
                pd1 = 0.25 - 0.25 * exp1
                ps2 = 0.25 + 0.75 * exp2
                pd2 = 0.25 - 0.25 * exp2
                pc1 = prob[c1]
                pc2 = prob[c2]
                pn1 = ps1 * pc1 + pd1 * (1.0 - pc1)
                pn2 = ps2 * pc2 + pd2 * (1.0 - pc2)
                prob[node] = pn1 * pn2
    
    root_prob = prob[tree_root]
    site_ll = np.sum(root_prob * 0.25, axis=1)
    site_ll = np.maximum(site_ll, 1e-300)
    total_ll = np.sum(np.log(site_ll))
    
    return total_ll

# ============================================================================
# NNI Refinement
# ============================================================================

def _nni_refine_ml(adj, node_count, n_leaves, seqs, n_sites, max_iters=200):
    """
    Perform NNI refinement using ML scoring.
    adj: dict mapping node -> list of (neighbor, length)
    node_count: total number of nodes
    n_leaves: number of leaves
    seqs: numpy array (n_leaves, n_sites)
    n_sites: number of sites
    """
    if n_leaves < 4:
        return adj, 0.0  # NNI not applicable
    
    current_adj = {k: list(v) for k, v in adj.items()}
    current_root = 0  # Root at leaf 0 for ML computation
    
    current_ll = _compute_ml(current_adj, current_root, seqs, n_sites)
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        
        best_gain = -1e-10
        best_move = None
        
        # Find all internal edges
        # An internal edge connects two nodes that are not leaves
        # For each internal edge (u, v), we can do NNI
        
        # Collect all edges
        processed = set()
        
        for u in range(node_count):
            for v, length in current_adj[u]:
                if u > v:
                    continue
                edge_key = (min(u, v), max(u, v))
                if edge_key in processed:
                    continue
                processed.add(edge_key)
                
                # Check if this is an internal edge (both endpoints have degree >= 3 or are internal)
                # Actually, for NNI we need both sides to have at least one leaf
                # Let's just try all edges and see if NNI is applicable
                
                if u >= n_leaves and v >= n_leaves:
                    # Both internal: standard NNI
                    # Get the other neighbors
                    u_neighbors = [n for n, l in current_adj[u] if n != v]
                    v_neighbors = [n for n, l in current_adj[v] if n != u]
                    
                    if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                        continue
                    
                    a, b = u_neighbors  # The two subtrees hanging off u (other than v)
                    c, d = v_neighbors  # The two subtrees hanging off v (other than u)
                    
                    # Original: u connects to {v, a, b}, v connects to {u, c, d}
                    # NNI move 1: swap b and c
                    # New: u connects to {v, a, c}, v connects to {u, b, d}
                    # Wait, that's not right. Let me reconsider.
                    
                    # The four subtrees are:
                    # T_a: subtree at a (away from u)
                    # T_b: subtree at b (away from u)  
                    # T_c: subtree at c (away from v)
                    # T_d: subtree at d (away from v)
                    
                    # Original topology: (T_a, T_b) - (T_c, T_d)
                    # NNI 1: (T_a, T_c) - (T_b, T_d)
                    # NNI 2: (T_a, T_d) - (T_b, T_c)
                    
                    # To perform NNI 1:
                    # Remove edges (u,a), (u,b), (v,c), (v,d), (u,v)
                    # Add edges (u,c), (u,a), (v,b), (v,d), (u,v)
                    # Wait, I'm confusing myself.
                    
                    # Let me think of it as:
                    # u is connected to: v, a, b
                    # v is connected to: u, c, d
                    
                    # For NNI 1: swap b and c
                    # u becomes connected to: v, a, c
                    # v becomes connected to: u, b, d
                    
                    # So we remove (u,b) and (v,c), add (u,c) and (v,b)
                    
                    # Let's implement this by modifying the adjacency list
                    
                    new_adj = {k: list(v2) for k, v2 in current_adj.items()}
                    
                    # Remove (u, b)
                    new_adj[u] = [(n, l) for n, l in new_adj[u] if n != b]
                    new_adj[b] = [(n, l) for n, l in new_adj[b] if n != u]
                    
                    # Remove (v, c)
                    new_adj[v] = [(n, l) for n, l in new_adj[v] if n != c]
                    new_adj[c] = [(n, l) for n, l in new_adj[c] if n != v]
                    
                    # Add (u, c) with some length
                    # And (v, b) with some length
                    # For simplicity, use the same length as the original edges
                    len_ub = current_adj[u][[n for n, l in current_adj[u] if n == b][0]][1] if b in [n for n, l in current_adj[u]] else 0.0
                    len_vc = current_adj[v][[n for n, l in current_adj[v] if n == c][0]][1] if c in [n for n, l in current_adj[v]] else 0.0
                    
                    # Actually, let's just use 0.0 for now (topology only matters for RF)
                    new_adj[u].append((c, 0.0))
                    new_adj[c].append((u, 0.0))
                    new_adj[v].append((b, 0.0))
                    new_adj[b].append((v, 0.0))
                    
                    # Compute new LL
                    new_ll = _compute_ml(new_adj, current_root, seqs, n_sites)
                    gain = new_ll - current_ll
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('nni1', u, v, a, b, c, d, new_adj, new_ll)
                    
                    # NNI 2: swap b and d
                    new_adj2 = {k: list(v2) for k, v2 in current_adj.items()}
                    
                    new_adj2[u] = [(n, l) for n, l in new_adj2[u] if n != b]
                    new_adj2[b] = [(n, l) for n, l in new_adj2[b] if n != u]
                    
                    new_adj2[v] = [(n, l) for n, l in new_adj2[v] if n != d]
                    new_adj2[d] = [(n, l) for n, l in new_adj2[d] if n != v]
                    
                    new_adj2[u].append((d, 0.0))
                    new_adj2[d].append((u, 0.0))
                    new_adj2[v].append((b, 0.0))
                    new_adj2[b].append((v, 0.0))
                    
                    new_ll2 = _compute_ml(new_adj2, current_root, seqs, n_sites)
                    gain2 = new_ll2 - current_ll
                    
                    if gain2 > best_gain:
                        best_gain = gain2
                        best_move = ('nni2', u, v, a, b, c, d, new_adj2, new_ll2)
        
        if best_move is None:
            break
        
        # Apply the best move
        _, _, _, _, _, _, _, new_adj, new_ll = best_move
        current_adj = new_adj
        current_ll = new_ll
        
        if best_gain < 1e-10:
            break
    
    return current_adj, current_ll

# ============================================================================
# Newick Output
# ============================================================================

def _adj_to_newick(adj, n_leaves, labels):
    """Convert adjacency list to Newick string."""
    # Build a rooted tree for Newick output
    # Root at leaf 0
    visited = set()
    parent = {}
    children = defaultdict(list)
    
    stack = [(0, -1)]
    while stack:
        node, par = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        
        if par >= 0:
            children[par].append(node)
        
        for neighbor, length in adj.get(node, []):
            if neighbor != par:
                stack.append((neighbor, node))
    
    # Recursively build Newick string
    def build_newick(node):
        if node < n_leaves:
            return labels[node]
        
        kids = children[node]
        if len(kids) == 0:
            return labels[node]  # Should not happen
        elif len(kids) == 1:
            child_str = build_newick(kids[0])
            # Get edge length
            length = adj[node][0][1] if node in adj else 0.0
            # Actually, need to find the edge to the child
            for neighbor, length in adj[node]:
                if neighbor == kids[0]:
                    child_str = f"{child_str}:{length:.4f}"
                    break
            return f"({child_str})"
        else:
            child_strs = []
            for kid in kids:
                cstr = build_newick(kid)
                for neighbor, length in adj[node]:
                    if neighbor == kid:
                        cstr = f"{cstr}:{length:.4f}"
                        break
                child_strs.append(cstr)
            return "(" + ",".join(child_strs) + ")"
    
    return build_newick(0)

# ============================================================================
# Main reconstruct function
# ============================================================================

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Convert sequences to integer matrix
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seqs_int = np.array([[seq_map[c] for c in sequences[label]] for label in labels], dtype=np.int8)
    
    # Compute distance matrix
    dists = _jc_distance_matrix(sequences, labels)
    
    # Run NJ to get guide tree
    nj_result = _nj_build(dists, labels)
    if nj_result is None:
        # Fallback: star tree
        if n_taxa == 3:
            return f"({labels[0]}:0.1,{labels[1]}:0.1,{labels[2]}:0.1);"
        else:
            parts = [f"{labels[i]}:0.1" for i in range(n_taxa)]
            return "(" + ",".join(parts) + ");"
    
    edges, node_count = nj_result
    
    # Build adjacency list
    adj = defaultdict(list)
    for a, b, length in edges:
        adj[a].append((b, length))
        adj[b].append((a, length))
    
    # Ensure all nodes are in adj
    for i in range(node_count):
        if i not in adj:
            adj[i] = []
    
    # Try NNI refinement with ML
    try:
        adj, final_ll = _nni_refine_ml(adj, node_count, n_taxa, seqs_int, n_sites, max_iters=100)
    except Exception:
        pass
    
    # Convert to Newick
    newick = _adj_to_newick(adj, n_taxa, labels)
    
    return newick + ";"
