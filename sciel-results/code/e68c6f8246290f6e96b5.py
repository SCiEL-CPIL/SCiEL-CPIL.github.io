"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Phylogenetic reconstruction using NJ guide tree + ML refinement under Jukes-Cantor.'''
import math
import time
import numpy as np


def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {(a, b): _jc_distance(sequences[a], sequences[b]) for a in labels for b in labels if a != b}
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _parse_newick(newick, labels):
    """Parse a Newick string into an adjacency list representation.
    Returns:
        adj: dict mapping node_name -> list of (neighbor, branch_length)
        leaf_set: set of leaf node names
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_set = set()
    
    # Stack-based parser
    # Each stack entry: (node_name, parent_name, parent_branch_length)
    # We'll use a different approach: recursive descent
    
    def parse(i):
        """Parse starting at position i, return (node_name, next_i)."""
        if i >= len(newick):
            return None, i
        
        if newick[i] == '(':
            # Internal node
            i += 1
            children = []
            while i < len(newick) and newick[i] != ')':
                if newick[i] == ',':
                    i += 1
                    continue
                child_name, i = parse(i)
                if child_name is None:
                    break
                children.append(child_name)
            if i < len(newick) and newick[i] == ')':
                i += 1
            
            # Read label/branch length after )
            label = ""
            branch_len = 0.0
            while i < len(newick) and newick[i] not in (',', ')'):
                if newick[i] == ':':
                    i += 1
                    break
                label += newick[i]
                i += 1
            if i < len(newick) and newick[i] == ':':
                i += 1
                branch_str = ""
                while i < len(newick) and newick[i] not in (',', ')'):
                    branch_str += newick[i]
                    i += 1
                try:
                    branch_len = float(branch_str)
                except:
                    branch_len = 0.0
            
            if not label:
                label = f"_int_{id(adj)}"
            
            return label, i
        else:
            # Leaf node
            label = ""
            branch_len = 0.0
            while i < len(newick) and newick[i] not in (',', ')'):
                if newick[i] == ':':
                    i += 1
                    break
                label += newick[i]
                i += 1
            if i < len(newick) and newick[i] == ':':
                i += 1
                branch_str = ""
                while i < len(newick) and newick[i] not in (',', ')'):
                    branch_str += newick[i]
                    i += 1
                try:
                    branch_len = float(branch_str)
                except:
                    branch_len = 0.0
            
            return label, i
    
    root_name, _ = parse(0)
    
    # Now build adjacency by re-parsing with parent tracking
    # This is complex. Let's use a simpler approach for NJ trees.
    # NJ trees have a specific structure. Let's just use the NJ result directly.
    
    # For now, return a simple representation
    return newick


def _build_tree_from_newick(newick):
    """
    Build a tree structure from Newick string.
    Returns a tree object with:
        - nodes: list where each entry is (children_list, is_leaf, taxon_index_or_None)
        - parent: list where parent[i] = parent node index, -1 for root
        - branch_lengths: list of branch lengths (branch from parent to node)
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    nodes = []  # list of [children, is_leaf, taxon_idx]
    parents = []  # parent index for each node
    branch_lens = []  # branch length from parent to this node
    
    # We'll use a stack to track current node being built
    # Stack entries: (node_index, expected_children_count, current_children_count)
    
    # First, let's identify all leaves and their order
    # Parse the Newick string to extract structure
    
    # Alternative approach: use a proper recursive parser
    pos = 0
    
    def parse_node():
        """Parse a node, return its index in the nodes list."""
        nonlocal pos
        
        if pos >= len(newick):
            return -1
        
        if newick[pos] == '(':
            pos += 1  # skip '('
            children = []
            while pos < len(newick) and newick[pos] != ')':
                if newick[pos] == ',':
                    pos += 1
                    continue
                child_idx = parse_node()
                if child_idx != -1:
                    children.append(child_idx)
            if pos < len(newick) and newick[pos] == ')':
                pos += 1  # skip ')'
            
            # Read label and branch length
            label = ""
            bl = 0.0
            while pos < len(newick) and newick[pos] not in (',', ')'):
                if newick[pos] == ':':
                    pos += 1
                    break
                label += newick[pos]
                pos += 1
            
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ""
                while pos < len(newick) and newick[pos] not in (',', ')'):
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            
            node_idx = len(nodes)
            nodes.append([children, False, None])
            parents.append(-1)  # will be set later
            branch_lens.append(bl)
            
            # Set parent for children
            for c in children:
                parents[c] = node_idx
            
            return node_idx
        else:
            # Leaf
            label = ""
            bl = 0.0
            while pos < len(newick) and newick[pos] not in (',', ')'):
                if newick[pos] == ':':
                    pos += 1
                    break
                label += newick[pos]
                pos += 1
            
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ""
                while pos < len(newick) and newick[pos] not in (',', ')'):
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            
            # Determine taxon index
            taxon_idx = -1
            if label.startswith('t'):
                try:
                    taxon_idx = int(label[1:])
                except:
                    taxon_idx = -1
            
            node_idx = len(nodes)
            nodes.append([[], True, taxon_idx])
            parents.append(-1)
            branch_lens.append(bl)
            
            return node_idx
    
    root_idx = parse_node()
    
    if root_idx == -1:
        return None
    
    # Fix parent of root
    parents[root_idx] = -1
    
    return {
        'nodes': nodes,
        'parents': parents,
        'branch_lens': branch_lens,
        'root': root_idx
    }


def _compute_likelihood(tree, seq_matrix):
    """
    Compute log-likelihood under JC model using Felsenstein's pruning.
    tree: dict with 'nodes', 'parents', 'branch_lens', 'root'
    seq_matrix: numpy array (n_taxa, n_sites) with values 0-3
    """
    nodes = tree['nodes']
    parents = tree['parents']
    branch_lens = tree['branch_lens']
    root = tree['root']
    
    n_sites = seq_matrix.shape[1]
    
    # Post-order traversal
    post_order = []
    visited = [False] * len(nodes)
    
    def dfs(nid):
        if visited[nid]:
            return
        visited[nid] = True
        node = nodes[nid]
        if not node[1]:  # internal
            for c in node[0]:
                dfs(c)
        post_order.append(nid)
    
    dfs(root)
    
    # L[node] = array of shape (n_sites, 4)
    L = {}
    
    for nid in post_order:
        node = nodes[nid]
        if node[1]:  # leaf
            taxon_idx = node[2]
            if taxon_idx < 0 or taxon_idx >= seq_matrix.shape[0]:
                # Fallback: use index based on order
                L[nid] = np.ones((n_sites, 4)) * 0.25
                continue
            
            seq = seq_matrix[taxon_idx]
            t = branch_lens[nid]
            p_same = 1.0/3.0 + 2.0/3.0 * math.exp(-3.0 * t)
            p_diff = 1.0/3.0 * (1.0 - math.exp(-3.0 * t))
            
            L_vec = np.zeros((n_sites, 4))
            for site in range(n_sites):
                obs = seq[site]
                for state in range(4):
                    if state == obs:
                        L_vec[site, state] = p_same
                    else:
                        L_vec[site, state] = p_diff
            L[nid] = L_vec
        else:
            children = node[0]
            L_vec = np.ones((n_sites, 4))
            
            for c in children:
                t = branch_lens[c]
                p_same = 1.0/3.0 + 2.0/3.0 * math.exp(-3.0 * t)
                p_diff = 1.0/3.0 * (1.0 - math.exp(-3.0 * t))
                
                Lc = L[c]
                # For each site and state s:
                # new_L[site, s] = L_vec[site, s] * (p_same * Lc[site, s] + p_diff * sum_{s' != s} Lc[site, s'])
                # = L_vec[site, s] * (p_diff * sum(Lc[site, :]) + (p_same - p_diff) * Lc[site, s])
                
                row_sum = Lc.sum(axis=1)  # shape (n_sites,)
                for s in range(4):
                    L_vec[:, s] *= (p_diff * row_sum + (p_same - p_diff) * Lc[:, s])
            
            L[nid] = L_vec
    
    root_L = L[root]
    log_lik = 0.0
    for site in range(n_sites):
        total = root_L[site].sum()
        if total > 1e-300:
            log_lik += math.log(total)
        else:
            log_lik += -1000.0
    
    return log_lik


def _optimize_branch_lengths(tree, seq_matrix, max_iter=10):
    """
    Optimize branch lengths using simple gradient-based approach.
    """
    nodes = tree['nodes']
    branch_lens = tree['branch_lens']
    
    # For each branch, we can try a few values and pick the best
    # This is a simplified approach
    
    n_sites = seq_matrix.shape[1]
    
    # Initial likelihood
    current_lik = _compute_likelihood(tree, seq_matrix)
    
    # For each leaf, adjust branch length
    for iter in range(max_iter):
        improved = False
        for nid in range(len(nodes)):
            node = nodes[nid]
            if node[1]:  # leaf
                old_bl = branch_lens[nid]
                # Try a few perturbations
                for factor in [0.5, 0.8, 1.2, 2.0, 0.1, 5.0]:
                    new_bl = max(0.001, old_bl * factor)
                    if abs(new_bl - old_bl) < 1e-6:
                        continue
                    branch_lens[nid] = new_bl
                    new_lik = _compute_likelihood(tree, seq_matrix)
                    if new_lik > current_lik + 1e-6:
                        current_lik = new_lik
                        improved = True
                    else:
                        branch_lens[nid] = old_bl
                        break
        if not improved:
            break
    
    return current_lik


def _nni_move(tree, edge_node):
    """
    Try an NNI move on the edge leading to edge_node.
    Returns a new tree dict if the move is valid, else None.
    """
    nodes = tree['nodes']
    parents = tree['parents']
    branch_lens = tree['branch_lens']
    root = tree['root']
    
    # Find the edge: parent -> edge_node
    p = parents[edge_node]
    if p == -1:
        return None  # Can't do NNI on root edge
    
    # The edge is p -> edge_node
    # NNI requires that both p and edge_node have at least 2 children (in the unrooted sense)
    # In a rooted representation, we need to be careful
    
    # For simplicity, skip NNI for now
    return None


def _nni_refine(tree, seq_matrix, deadline):
    """
    Refine tree using NNI moves.
    """
    start_time = time.time()
    current_lik = _compute_likelihood(tree, seq_matrix)
    
    # For now, just optimize branch lengths
    current_lik = _optimize_branch_lengths(tree, seq_matrix)
    
    return tree, current_lik


def _ml_reconstruct(sequences, deadline):
    """
    ML-based reconstruction:
    1. Build NJ guide tree
    2. Parse into tree structure
    3. Optimize branch lengths
    4. Try NNI refinement
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    # Convert sequences to numeric matrix
    base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seq_matrix = np.zeros((n, n_sites), dtype=np.int8)
    for i, label in enumerate(labels):
        for j, ch in enumerate(sequences[label]):
            seq_matrix[i, j] = base_to_num[ch]
    
    # Step 1: NJ guide tree
    nj_newick = _nj_reconstruct(sequences)
    
    # Step 2: Parse
    tree = _build_tree_from_newick(nj_newick)
    if tree is None:
        return nj_newick
    
    # Step 3: Optimize branch lengths
    if time.time() < deadline - 1.0:
        _optimize_branch_lengths(tree, seq_matrix)
    
    # Step 4: NNI refinement (simplified)
    if time.time() < deadline - 0.5:
        tree, _ = _nni_refine(tree, seq_matrix, deadline)
    
    # Convert back to Newick
    def to_newick(nid):
        node = tree['nodes'][nid]
        if node[1]:  # leaf
            taxon_idx = node[2]
            if taxon_idx >= 0:
                return f"t{taxon_idx}"
            else:
                return f"t{nid}"
        else:
            children = node[0]
            child_strs = [to_newick(c) for c in children]
            return "(" + ",".join(child_strs) + ")"
    
    result = "(" + to_newick(tree['root']) + ");"
    return result


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    start = time.time()
    # Budget: 180s total for 20 instances, so ~9s per instance
    # Use 8s per instance as a safe limit
    deadline = start + 8.0
    
    try:
        # Try ML-based approach
        result = _ml_reconstruct(sequences, deadline)
        if result and result.strip():
            return result
    except Exception:
        pass
    
    # Fallback to NJ
    try:
        result = _nj_reconstruct(sequences)
        if result and result.strip():
            return result
    except Exception:
        pass
    
    # Last resort: star tree
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
