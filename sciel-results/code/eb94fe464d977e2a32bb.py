"""Phylogenetic topology recovery: NJ guide + Felsenstein ML pruning + NNI refinement."""
import math
import time
import numpy as np

_DEADLINE = 260.0  # safety margin before 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two DNA strings."""
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_tree(labels, dist_matrix):
    """Build a Newick tree via Neighbour Joining. Returns (newick, parent_map, children_map).
    
    We represent the tree internally as a set of nodes with parent/children pointers
    for efficient NNI operations.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";", {}, {0: []}, {0: 0}
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # dist_matrix is a dict (a,b) -> dist
    nodes = list(range(n))  # internal node indices
    # leaf names
    names = list(labels)
    # Tree structure: each node has a 'newick' string (for leaves it's the name)
    node_names = list(labels)
    # Active nodes
    active = list(range(n))
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = dist_matrix.get((labels[i], labels[j]), 0.0)

    next_id = n
    # We'll track tree structure for building final newick
    # parent[i] = the parent node, children[i] = list of child nodes
    parent = {}
    children = {i: [] for i in range(n)}

    while len(active) > 3:
        m = len(active)
        # Compute total distances
        total = {}
        for a in active:
            total[a] = sum(dist[(a, b)] for b in active if b != a)

        best_crit = float('inf')
        best_a, best_b = -1, -1
        for i_idx, a in enumerate(active):
            for b in active[i_idx + 1:]:
                crit = (m - 2) * dist[(a, b)] - total[a] - total[b]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = a, b

        a, b = best_a, best_b
        # Create new internal node
        new = next_id
        next_id += 1
        node_names.append(f"n{new}")

        # Compute branch lengths (for the newick, though we don't need them for topology)
        # For NJ: r = 0.5 * (d(a,b) + total[a] - total[b]) / (m-2)
        # But we just need topology, so set to 0
        # Connect a and b under new
        parent[a] = new
        parent[b] = new
        children[new] = [a, b]
        children[new].append  # placeholder

        # Update distances from new to remaining active nodes
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
            dist[(new, x)] = d_new_x
            dist[(x, new)] = d_new_x

        # Remove a, b from active, add new
        active = rest + [new]

    # Final: connect remaining 3 nodes in a star (unrooted)
    # Pick one as "root" for the representation
    if len(active) == 3:
        # Create one more internal node to make it a proper binary tree
        new = next_id
        next_id += 1
        node_names.append(f"n{new}")
        a, b, c = active
        parent[a] = new
        parent[b] = new
        children[new] = [a, b]
        # c is also a child
        children[new].append(c)
        parent[c] = new
        # The root is new (but unrooted, so we just pick new as the root for representation)
        root = new
    else:
        root = active[0]

    # Build newick string
    def build_newick(node):
        if node < n:
            return node_names[node]
        ch = children[node]
        if len(ch) == 1:
            return build_newick(ch[0])
        return "(" + ",".join(build_newick(c) for c in ch) + ")"

    newick = build_newick(root) + ";"
    return newick, parent, children, root, n, node_names


def _felsenstein_loglik(tree_structure, sequences_arr, n_taxa, n_sites):
    """Compute log-likelihood under JC model using Felsenstein's pruning algorithm.
    
    tree_structure: (parent, children, root, n, node_names)
    sequences_arr: numpy array of shape (n_taxa, n_sites) with values 0,1,2,3
    Returns log-likelihood (sum over sites).
    """
    parent, children, root, n, node_names = tree_structure
    # Build list of nodes in post-order (leaves first)
    # We need to process nodes in an order where children are processed before parents
    
    # Build adjacency for traversal
    # Get all internal nodes
    all_nodes = list(range(n, len(node_names)))
    
    # Topological sort: process leaves first, then internal nodes bottom-up
    # We can do a DFS from root
    order = []
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        if node < n:
            # leaf
            order.append(node)
        else:
            for c in children[node]:
                dfs(c)
            order.append(node)
    
    dfs(root)
    
    # Now compute likelihood
    # For each site, we compute the likelihood at the root
    # JC model: transition probability P(x->y | t) = 0.25 * (1 + 3*exp(-4t/3)) if x==y
    #                                                = 0.25 * (1 - exp(-4t/3)) if x!=y
    # Since we don't have branch lengths, we'll use a fixed small value or optimize
    # Actually, for topology comparison, we can use a fixed branch length
    # Let's use t = 0.1 for all branches (this is a simplification)
    
    # Actually, let's properly compute with branch lengths. But we don't have them from NJ easily.
    # Let me re-think: for NNI refinement, we need to compare likelihoods.
    # We can use a simple approach: for each site, compute the likelihood assuming
    # all branches have length 0.1.
    
    t = 0.1
    exp_term = math.exp(-4.0 * t / 3.0)
    p_same = 0.25 * (1.0 + 3.0 * exp_term)
    p_diff = 0.25 * (1.0 - exp_term)
    
    total_loglik = 0.0
    for site in range(n_sites):
        # Compute likelihood at each node for each base (0=A,1=C,2=G,3=T)
        # L[node][base] = probability of observing data in subtree rooted at node, given node has base
        L = {}
        for node in order:
            if node < n:
                # Leaf: probability of observing the actual base
                obs = sequences_arr[node, site]
                L[node] = [0.25, 0.25, 0.25, 0.25]
                L[node][obs] = 1.0  # Actually, for leaves, the likelihood is 1 if we condition on the observed base
                # Wait, in Felsenstein's algorithm, for a leaf with observed base b:
                # L_leaf[b] = 1, L_leaf[other] = 0
                L[node] = [0.0, 0.0, 0.0, 0.0]
                L[node][obs] = 1.0
            else:
                # Internal node: sum over children
                vals = [0.0, 0.0, 0.0, 0.0]
                for c in children[node]:
                    for base in range(4):
                        s = 0.0
                        for child_base in range(4):
                            if base == child_base:
                                s += L[c][child_base] * p_same
                            else:
                                s += L[c][child_base] * p_diff
                        vals[base] += s
                L[node] = vals
        # Root likelihood
        root_vals = L[root]
        # Unrooted tree: the root is arbitrary, so we sum over all possible bases at root
        site_loglik = math.log(sum(root_vals) + 1e-300)
        total_loglik += site_loglik
    
    return total_loglik


def _nni_refine(parent, children, root, n, node_names, sequences_arr, n_sites, max_iterations=50):
    """Refine topology via NNI moves, maximizing JC log-likelihood.
    
    Returns improved (parent, children, root, n, node_names).
    """
    # We need to be able to compute likelihood quickly.
    # For NNI, we only need to recompute the affected part of the tree.
    # But for simplicity and given time constraints, let's do a full recomputation
    # for each candidate NNI move, but limit the number of moves.
    
    # First, identify all internal edges that can be NNI'd
    # An NNI move takes an internal edge (u, v) where u is parent of v,
    # and both u and v have two children (binary tree).
    # The four taxa/subtrees around the edge can be rearranged.
    
    # Let's collect all internal nodes
    internal_nodes = [nd for nd in range(n, len(node_names)) if len(children[nd]) >= 2]
    
    # For each internal edge, try NNI
    improved = True
    iteration = 0
    current_loglik = _felsenstein_loglik((parent, children, root, n, node_names), sequences_arr, n_sites)
    
    while improved and iteration < max_iterations and _elapsed() < _DEADLINE:
        improved = False
        iteration += 1
        
        # Find best NNI move
        best_gain = 1e-10
        best_move = None
        
        for u in internal_nodes:
            if u == root:
                continue
            v = parent[u]
            if v is None or v < n:
                continue
            # Check that both u and v have exactly 2 children (binary)
            if len(children[u]) != 2 or len(children[v]) != 2:
                continue
            
            # The edge is v -> u
            # Children of v: one is u, the other is w
            # Children of u: two subtrees a, b
            w = None
            for c in children[v]:
                if c != u:
                    w = c
                    break
            if w is None:
                continue
            
            a, b = children[u]
            
            # Original topology: (w, (a, b)) with edge v-u
            # NNI alternatives:
            # 1. (a, (w, b)) 
            # 2. (b, (w, a))
            
            # Try move 1: swap so that a is sibling of v, and w is sibling of b under u
            # New structure: 
            #   v has children: a and u
            #   u has children: w and b
            # Save old state
            old_children_v = list(children[v])
            old_children_u = list(children[u])
            
            # Apply NNI move 1
            children[v] = [a, u]
            children[u] = [w, b]
            # Update parents
            parent[a] = v
            parent[w] = u
            parent[b] = u
            
            new_loglik = _felsenstein_loglik((parent, children, root, n, node_names), sequences_arr, n_sites)
            gain = new_loglik - current_loglik
            if gain > best_gain:
                best_gain = gain
                best_move = (v, u, w, a, b, 1, old_children_v, old_children_u)
            
            # Revert
            children[v] = old_children_v
            children[u] = old_children_u
            parent[u] = v
            parent[w] = v
            parent[a] = u
            parent[b] = u
            
            # Try move 2: swap so that b is sibling of v, and w is sibling of a under u
            # New structure:
            #   v has children: b and u
            #   u has children: w and a
            children[v] = [b, u]
            children[u] = [w, a]
            parent[b] = v
            parent[w] = u
            parent[a] = u
            
            new_loglik = _felsenstein_loglik((parent, children, root, n, node_names), sequences_arr, n_sites)
            gain = new_loglik - current_loglik
            if gain > best_gain:
                best_gain = gain
                best_move = (v, u, w, a, b, 2, old_children_v, old_children_u)
            
            # Revert
            children[v] = old_children_v
            children[u] = old_children_u
            parent[u] = v
            parent[w] = v
            parent[a] = u
            parent[b] = u
        
        if best_move is not None:
            v, u, w, a, b, move_type, old_cv, old_cu = best_move
            # Apply the best move
            if move_type == 1:
                children[v] = [a, u]
                children[u] = [w, b]
                parent[a] = v
                parent[w] = u
                parent[b] = u
            else:
                children[v] = [b, u]
                children[u] = [w, a]
                parent[b] = v
                parent[w] = u
                parent[a] = u
            
            current_loglik += best_gain
            improved = True
    
    return parent, children, root, n, node_names


def _build_newick(parent, children, root, n, node_names):
    """Build Newick string from tree structure."""
    def build(node):
        if node < n:
            return node_names[node]
        ch = children[node]
        if len(ch) == 1:
            return build(ch[0])
        return "(" + ",".join(build(c) for c in ch) + ")"
    return build(root) + ";"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    if n_taxa <= 1:
        return labels[0] + ";"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute distance matrix
    dist_matrix = {}
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist_matrix[(labels[i], labels[j])] = d
            dist_matrix[(labels[j], labels[i])] = d
    
    # Build NJ tree
    newick, parent, children, root, n, node_names = _nj_tree(labels, dist_matrix)
    
    # Prepare sequences as numpy array for faster ML computation
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.zeros((n_taxa, n_sites), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j in range(n_sites):
            seq_arr[i, j] = base_map[s[j]]
    
    # NNI refinement
    if _elapsed() < _DEADLINE - 30:
        parent, children, root, n, node_names = _nni_refine(
            parent, children, root, n, node_names, seq_arr, n_sites, max_iterations=20
        )
    
    # Build final newick
    result = _build_newick(parent, children, root, n, node_names)
    return result
