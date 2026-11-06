"""Phylogenetic topology recovery via ML quartet-consistency + branch-length optimization."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_START_TIME = None
_TIME_LIMIT = 265.0


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _encode_sequences(sequences, labels):
    """Encode DNA sequences into one-hot binary matrix."""
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = np.zeros((n, L, 4), dtype=np.float64)
    for i, label in enumerate(labels):
        s = sequences[label]
        for pos in range(L):
            enc[i, pos, seq_map[s[pos]]] = 1.0
    return enc, L


def _log_likelihood(enc, L, n_taxa, tree, bl):
    """
    Felsenstein's pruning algorithm for JC model.
    
    tree: list of (node_id, left_child, right_child) in POST-ORDER.
    bl: dict mapping node_id -> branch length from node to its parent.
        For leaves and all non-root internal nodes.
    
    Node indices: 0..n-1 are leaves, n.. are internal nodes.
    The root is the last entry in tree.
    """
    if not tree:
        # Star tree: all leaves connected to a single internal node
        # For JC, likelihood of star tree:
        # At each site, sum over root state s of prod_i P(obs_i -> s, d_i)
        # This is complex. Let's just return a baseline.
        # Actually for n=3, a star tree is the only option.
        # Let me handle it specially.
        return -1e300
    
    total_nodes = n_taxa + len(tree)
    root = tree[-1][0]
    
    # LL[site, node, state]
    LL = np.zeros((L, total_nodes, 4), dtype=np.float64)
    LL[:, :n_taxa, :] = np.transpose(enc, (1, 0, 2))
    
    for node_id, left, right in tree:
        bl_l = bl.get(left, 0.1)
        bl_r = bl.get(right, 0.1)
        
        exp_l = math.exp(-4.0 * bl_l / 3.0)
        ps_l = 0.25 + 0.75 * exp_l
        pd_l = 0.25 * (1.0 - exp_l)
        
        exp_r = math.exp(-4.0 * bl_r / 3.0)
        ps_r = 0.25 + 0.75 * exp_r
        pd_r = 0.25 * (1.0 - exp_r)
        
        left_LL = LL[:, left, :]
        right_LL = LL[:, right, :]
        
        left_sum = np.sum(left_LL, axis=1, keepdims=True)
        right_sum = np.sum(right_LL, axis=1, keepdims=True)
        
        left_contrib = (ps_l - pd_l) * left_LL + pd_l * left_sum
        right_contrib = (ps_r - pd_r) * right_LL + pd_r * right_sum
        
        LL[:, node_id, :] = left_contrib * right_contrib
    
    root_LL = LL[:, root, :]
    site_likes = np.sum(root_LL, axis=1)
    site_likes = np.maximum(site_likes, 1e-300)
    return float(np.sum(np.log(site_likes)))


def _optimize_bl(enc, L, n_taxa, tree, max_rounds=3):
    """Coordinate descent on branch lengths."""
    # Non-root nodes
    all_children = set()
    for nid, l, r in tree:
        all_children.add(l)
        all_children.add(r)
    
    non_root = [n for n in all_children]
    
    bl = {n: 0.15 for n in non_root}
    
    candidates = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    best_ll = _log_likelihood(enc, L, n_taxa, tree, bl)
    
    for _ in range(max_rounds):
        if _time_left() < 10.0:
            break
        for node in non_root:
            best_val = bl[node]
            best_node_ll = -1e300
            for al in candidates:
                bl[node] = al
                ll = _log_likelihood(enc, L, n_taxa, tree, bl)
                if ll > best_node_ll:
                    best_node_ll = ll
                    best_val = al
            bl[node] = best_val
        # Update best
        ll = _log_likelihood(enc, L, n_taxa, tree, bl)
        if ll > best_ll:
            best_ll = ll
    
    return best_ll, bl


def _nni_refine(enc, L, n_taxa, tree, bl, rng, max_moves=30):
    """NNI (Nearest Neighbor Interchange) refinement."""
    if len(tree) < 2:
        return tree, bl, _log_likelihood(enc, L, n_taxa, tree, bl)
    
    children = {}
    for nid, l, r in tree:
        children[nid] = (l, r)
    
    all_children = set()
    for nid, l, r in tree:
        all_children.add(l)
        all_children.add(r)
    
    root = None
    for nid, _, _ in tree:
        if nid not in all_children:
            root = nid
            break
    
    internal_nodes = [nid for nid, _, _ in tree]
    current_ll = _log_likelihood(enc, L, n_taxa, tree, bl)
    best_tree = tree
    best_bl = bl
    best_ll = current_ll
    
    for _ in range(max_moves):
        if _time_left() < 10.0:
            break
        
        # Pick a random internal edge: node v with parent u
        # v is an internal node (not root), u is its parent
        candidates_v = [n for n in internal_nodes if n != root]
        if not candidates_v:
            break
        
        v = rng.choice(candidates_v)
        
        # Find parent u of v
        u = None
        for nid, l, r in tree:
            if l == v or r == v:
                u = nid
                break
        
        if u is None:
            continue
        
        # v's children
        a, b = children[v]
        # u's other child (not v)
        if children[u][0] == v:
            c = children[u][1]
        else:
            c = children[u][0]
        
        # NNI: swap to make (a, u) and (v, c) or (b, u) and (v, a)
        # New topology: u has children (a, v'), v' has children (b, c) -- wait
        # Original: u -- v, v -- (a, b), u -- c
        # NNI option 1: u -- (a, v'), v' -- (b, c)  [but v' is new]
        # Actually NNI on edge u-v: 
        #   Original: (c, (a, b)) with u as parent of v, v as parent of a,b
        #   NNI: (a, (c, b)) -- swap b with c across the edge
        #   Or: (b, (c, a))
        
        # Let me do: new children of u = (a, new_v), new_v children = (b, c)
        # But new_v needs a node ID. Let me just swap: u gets (a, c), v gets (b, ...)
        # Hmm, this changes the number of internal nodes.
        
        # Simpler: just swap the "sides". 
        # Original: u has children (v, c), v has children (a, b)
        # NNI: u has children (a, v2), v2 has children (b, c) -- but v2 is new
        # Or: u has children (b, v2), v2 has children (a, c)
        
        # For simplicity, let me just do a "rotation": 
        # Remove v from u's children, attach a (or b) directly to u,
        # and attach c (or the other) to v.
        
        # New: u has children (a, v), v has children (b, c)
        # Check: is this valid? a was a child of v, now child of u.
        # c was a child of u, now child of v.
        # This is a valid NNI if a is a leaf or internal node that doesn't create a cycle.
        
        new_tree = []
        for nid, l, r in tree:
            if nid == u:
                # u now has children (a, v)
                new_tree.append((u, a, v))
            elif nid == v:
                # v now has children (b, c)
                new_tree.append((v, b, c))
            else:
                new_tree.append((nid, l, r))
        
        # Reorder to post-order
        new_children = {}
        for nid, l, r in new_tree:
            new_children[nid] = (l, r)
        
        new_all_children = set()
        for nid, l, r in new_tree:
            new_all_children.add(l)
            new_all_children.add(r)
        
        new_root = None
        for nid, _, _ in new_tree:
            if nid not in new_all_children:
                new_root = nid
                break
        
        if new_root is None:
            continue
        
        # Post-order
        po = []
        visited = set()
        
        def po_traverse(node):
            if node in visited:
                return
            visited.add(node)
            if node in new_children:
                l, r = new_children[node]
                po_traverse(l)
                po_traverse(r)
            po.append(node)
        
        po_traverse(new_root)
        
        new_po_tree = []
        for node in po:
            if node in new_children:
                l, r = new_children[node]
                new_po_tree.append((node, l, r))
        
        # Evaluate
        new_ll = _log_likelihood(enc, L, n_taxa, new_po_tree, bl)
        
        if new_ll > best_ll:
            best_ll = new_ll
            best_tree = new_po_tree
            # Update children and internal_nodes
            children = new_children
            internal_nodes = [nid for nid, _, _ in new_po_tree]
    
    return best_tree, best_bl, best_ll


def _random_tree(n_taxa, rng):
    """Generate random binary unrooted tree via random pair-merging."""
    if n_taxa <= 3:
        return []
    
    components = list(range(n_taxa))
    rng.shuffle(components)
    
    tree_ops = []
    next_id = n_taxa
    
    while len(components) > 1:
        i = rng.randrange(len(components))
        j = rng.randrange(len(components) - 1)
        if j >= i:
            j += 1
        if i == j:
            j = (j + 1) % len(components)
            if i == j:
                continue
        
        a = components[i]
        b = components[j]
        new_node = next_id
        next_id += 1
        tree_ops.append((new_node, a, b))
        
        components.remove(a)
        components.remove(b)
        components.append(new_node)
    
    # Convert to post-order
    children = {}
    for nid, l, r in tree_ops:
        children[nid] = (l, r)
    
    all_children = set()
    for nid, l, r in tree_ops:
        all_children.add(l)
        all_children.add(r)
    
    root = None
    for nid, _, _ in tree_ops:
        if nid not in all_children:
            root = nid
            break
    
    if root is None:
        return tree_ops
    
    po = []
    visited = set()
    
    def po_traverse(node):
        if node in visited:
            return
        visited.add(node)
        if node in children:
            l, r = children[node]
            po_traverse(l)
            po_traverse(r)
        po.append(node)
    
    po_traverse(root)
    
    po_tree = []
    for node in po:
        if node in children:
            l, r = children[node]
            po_tree.append((node, l, r))
    
    return po_tree


def _to_newick(tree, n_taxa, labels):
    if not tree:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return "(" + ",".join(labels) + ")"
    
    children = {}
    for nid, l, r in tree:
        children[nid] = (l, r)
    
    all_children = set()
    for nid, l, r in tree:
        all_children.add(l)
        all_children.add(r)
    
    root = None
    for nid, _, _ in tree:
        if nid not in all_children:
            root = nid
            break
    
    if root is None:
        return "(" + ",".join(labels) + ")"
    
    def to_nw(node):
        if node < n_taxa:
            return labels[node]
        else:
            l, r = children[node]
            return f"({to_nw(l)},{to_nw(r)})"
    
    return to_nw(root) + ";"


def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return ""
    
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    enc, L = _encode_sequences(sequences, labels)
    
    best_ll = -1e300
    best_newick = None
    rng = random.Random(123)
    
    n_tries = 0
    max_tries = 300
    
    while n_tries < max_tries:
        if _time_left() < 15.0:
            break
        
        # Generate random tree
        tree = _random_tree(n_taxa, rng)
        
        if not tree:
            n_tries += 1
            continue
        
        # Optimize branch lengths
        ll, bl = _optimize_bl(enc, L, n_taxa, tree, max_rounds=2)
        
        # NNI refinement
        refined_tree, refined_bl, refined_ll = _nni_refine(
            enc, L, n_taxa, tree, bl, rng, max_moves=15
        )
        
        if refined_ll > best_ll:
            best_ll = refined_ll
            best_newick = _to_newick(refined_tree, n_taxa, labels)
        
        if ll > best_ll:
            best_ll = ll
            best_newick = _to_newick(tree, n_taxa, labels)
        
        n_tries += 1
    
    if best_newick is None:
        # Fallback
        best_newick = "(" + ",".join(labels) + ")"
    
    return best_newick

# EVOLVE-BLOCK-END
