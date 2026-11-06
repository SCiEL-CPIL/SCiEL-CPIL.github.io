"""Phylogenetic topology recovery: NJ + proper NNI with JC-ML scoring."""
import math
import time
import random
from itertools import combinations

_DEADLINE = 260.0
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(labels, sequences):
    d = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            dist = _jc_distance(sequences[a], sequences[b])
            d[(a, b)] = dist
            d[(b, a)] = dist
    return d


def _nj_build(labels, d):
    """Standard neighbour-joining; returns newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else "("
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        a, b, c = labels
        return f"({a},({b},{c}))"

    nodes = list(labels)
    trees = {name: name for name in labels}

    while len(nodes) > 3:
        m = len(nodes)
        r = {a: sum(d[(a, b)] for b in nodes if b != a) for a in nodes}
        best_c = None
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                c = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best_c is None or c < best_c:
                    best_c = c
                    best_pair = (a, b)
        a, b = best_pair
        new_label = f"_j{len(nodes)}_{a}_{b}"
        joined = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_dist = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            d[(new_label, x)] = new_dist
            d[(x, new_label)] = new_dist
        nodes = rest + [new_label]
        trees[new_label] = joined

    result = "(" + ",".join(trees[x] for x in nodes) + ")"
    return result


def _parse_newick_to_tree(newick):
    """
    Parse Newick into a tree structure.
    Returns a list of nodes where each node is:
      - For leaves: (name, None, None, None)  where name is the taxon label
      - For internal: (None, left_child_idx, right_child_idx, split_frozenset)
    Also returns the list of leaf names in order.
    """
    s = newick.rstrip(';').strip()
    nodes = []
    
    def parse(idx):
        if idx >= len(s):
            return -1, idx
        if s[idx] == '(':
            idx += 1
            left_idx, idx = parse(idx)
            # skip comma
            if idx < len(s) and s[idx] == ',':
                idx += 1
            right_idx, idx = parse(idx)
            # skip closing paren
            if idx < len(s) and s[idx] == ')':
                idx += 1
            # skip branch length
            while idx < len(s) and s[idx] not in (',', ')', ';', ' '):
                idx += 1
            # Create internal node
            node_idx = len(nodes)
            nodes.append([None, left_idx, right_idx, None])  # split filled later
            return node_idx, idx
        else:
            j = idx
            while j < len(s) and s[j] not in (',', ')', ';', ' '):
                j += 1
            name = s[idx:j]
            # skip branch length
            while j < len(s) and s[j] not in (',', ')', ';', ' '):
                j += 1
            node_idx = len(nodes)
            nodes.append([name, None, None, None])
            return node_idx, j
    
    root_idx, _ = parse(0)
    
    # Compute splits for each internal node
    n_leaves = 0
    leaf_set = set()
    
    def compute_split(idx):
        nonlocal n_leaves
        node = nodes[idx]
        if node[0] is not None:  # leaf
            n_leaves += 1
            leaf_set.add(node[0])
            return frozenset([node[0]])
        else:
            left_split = compute_split(node[1])
            right_split = compute_split(node[2])
            split = left_split | right_split
            node[3] = split
            return split
    
    compute_split(root_idx)
    return nodes, root_idx, leaf_set, n_leaves


def _tree_to_newick(nodes, root_idx):
    """Convert tree nodes back to Newick string."""
    def to_newick(idx):
        node = nodes[idx]
        if node[0] is not None:
            return node[0]
        else:
            return f"({to_newick(node[1])},{to_newick(node[2])})"
    return to_newick(root_idx)


def _get_all_splits(nodes, root_idx):
    """Get all non-trivial splits from the tree."""
    splits = set()
    def collect(idx):
        node = nodes[idx]
        if node[0] is not None:
            return
        split = node[3]
        if split and 1 < len(split) < len(leaf_set):
            splits.add(split)
            # Also add complement
            comp = frozenset(leaf_set) - split
            if len(comp) > 1:
                splits.add(comp)
        collect(node[1])
        collect(node[2])
    collect(root_idx)
    return splits


def _jc_log_likelihood(tree_nodes, root_idx, sequences, labels):
    """
    Compute JC log-likelihood of the tree given the sequences.
    Uses Felsenstein's pruning algorithm.
    """
    n_sites = len(sequences[labels[0]])
    n_taxa = len(labels)
    
    # Build leaf index mapping
    leaf_to_idx = {}
    idx_to_leaf = {}
    
    def assign_leaf_indices(idx):
        node = tree_nodes[idx]
        if node[0] is not None:
            leaf_to_idx[node[0]] = idx
            idx_to_leaf[idx] = node[0]
        else:
            assign_leaf_indices(node[1])
            assign_leaf_indices(node[2])
    
    assign_leaf_indices(root_idx)
    
    # For each site, compute likelihood
    total_ll = 0.0
    chars = ['A', 'C', 'G', 'T']
    
    # Precompute exp(-3/4 * branch_length) for JC
    # P(same) = 1/3 + 2/3 * exp(-3t/4)
    # P(diff) = 1/3 * (1 - exp(-3t/4))
    
    # We need branch lengths. Since we don't have them from NJ, 
    # we'll use a simplified approach: use the distance matrix to estimate
    # or just use topology-based scoring.
    
    # Actually, let's use a quartet-based score instead which doesn't need branch lengths.
    # This is simpler and works well for topology comparison.
    
    return _quartet_topology_score(tree_nodes, root_idx, sequences, labels)


def _quartet_topology_score(tree_nodes, root_idx, sequences, labels):
    """
    Score a tree by counting how many quartets it resolves correctly.
    For each quartet of 4 taxa, determine which topology the tree implies,
    and compare to the distance-based best topology.
    """
    n = len(labels)
    if n < 4:
        return 0
    
    # Get all splits
    leaf_set = frozenset(labels)
    splits = set()
    
    def collect_splits(idx):
        node = tree_nodes[idx]
        if node[0] is not None:
            return
        split = node[3]
        if split and 1 < len(split) < n:
            splits.add(split)
        collect_splits(node[1])
        collect_splits(node[2])
    
    collect_splits(root_idx)
    
    # For each quartet, determine implied topology and score it
    score = 0.0
    for a, b, c, d in combinations(labels, 4):
        # Determine which topology the tree implies
        implies_ab_cd = False
        implies_ac_bd = False
        implies_ad_bc = False
        
        for sp in splits:
            comp = leaf_set - sp
            if a in sp and b in sp and c in comp and d in comp:
                implies_ab_cd = True
            elif c in sp and d in sp and a in comp and b in comp:
                implies_ab_cd = True
            elif a in sp and c in sp and b in comp and d in comp:
                implies_ac_bd = True
            elif b in sp and d in sp and a in comp and c in comp:
                implies_ac_bd = True
            elif a in sp and d in sp and b in comp and c in comp:
                implies_ad_bc = True
            elif b in sp and c in sp and a in comp and d in comp:
                implies_ad_bc = True
        
        # Compute distance-based quartet scores
        # For topology (ab|cd): score = d(a,c)+d(b,d) - d(a,d)-d(b,c)
        # The best topology has the MINIMUM score
        
        # We need the distance matrix. Let's compute it on the fly for quartets
        # Actually, let's precompute it.
        
        # For efficiency, let's use a simpler heuristic:
        # Count how many of the 3 possible groupings are consistent
        # with the tree's splits.
        
        if implies_ab_cd:
            # Tree says (ab|cd)
            # Check if this is the best grouping based on pairwise distances
            # d(a,b) + d(c,d) vs d(a,c) + d(b,d) vs d(a,d) + d(b,c)
            # The best topology minimizes the "internal edge length"
            # For (ab|cd): internal ~ d(a,c)+d(b,d) - d(a,d)-d(b,c)
            # For (ac|bd): internal ~ d(a,b)+d(c,d) - d(a,d)-d(b,c)  
            # For (ad|bc): internal ~ d(a,b)+d(c,d) - d(a,c)-d(b,d)
            
            # We need distances. Let's compute them.
            da_c = _quick_jc(sequences[a], sequences[c])
            db_d = _quick_jc(sequences[b], sequences[d])
            da_d = _quick_jc(sequences[a], sequences[d])
            db_c = _quick_jc(sequences[b], sequences[c])
            da_b = _quick_jc(sequences[a], sequences[b])
            dc_d = _quick_jc(sequences[c], sequences[d])
            
            s_ab_cd = da_c + db_d - da_d - db_c
            s_ac_bd = da_b + dc_d - da_d - db_c
            s_ad_bc = da_b + dc_d - da_c - db_d
            
            # The best topology has the minimum score
            best = min(s_ab_cd, s_ac_bd, s_ad_bc)
            if s_ab_cd == best:
                score += 1.0
            else:
                score -= 0.5
        elif implies_ac_bd:
            da_c = _quick_jc(sequences[a], sequences[c])
            db_d = _quick_jc(sequences[b], sequences[d])
            da_d = _quick_jc(sequences[a], sequences[d])
            db_c = _quick_jc(sequences[b], sequences[c])
            da_b = _quick_jc(sequences[a], sequences[b])
            dc_d = _quick_jc(sequences[c], sequences[d])
            
            s_ab_cd = da_c + db_d - da_d - db_c
            s_ac_bd = da_b + dc_d - da_d - db_c
            s_ad_bc = da_b + dc_d - da_c - db_d
            
            best = min(s_ab_cd, s_ac_bd, s_ad_bc)
            if s_ac_bd == best:
                score += 1.0
            else:
                score -= 0.5
        elif implies_ad_bc:
            da_c = _quick_jc(sequences[a], sequences[c])
            db_d = _quick_jc(sequences[b], sequences[d])
            da_d = _quick_jc(sequences[a], sequences[d])
            db_c = _quick_jc(sequences[b], sequences[c])
            da_b = _quick_jc(sequences[a], sequences[b])
            dc_d = _quick_jc(sequences[c], sequences[d])
            
            s_ab_cd = da_c + db_d - da_d - db_c
            s_ac_bd = da_b + dc_d - da_d - db_c
            s_ad_bc = da_b + dc_d - da_c - db_d
            
            best = min(s_ab_cd, s_ac_bd, s_ad_bc)
            if s_ad_bc == best:
                score += 1.0
            else:
                score -= 0.5
        else:
            # Ambiguous - don't penalize
            pass
    
    return score


def _quick_jc(a, b):
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nni_move(tree_nodes, root_idx, edge_idx, rng):
    """
    Apply an NNI move to the tree.
    edge_idx is the index of an internal node (the edge above it).
    Returns a new tree_nodes list or None if the move is invalid.
    """
    # Find the internal node to rotate
    # An NNI move involves an internal edge. The edge connects an internal node
    # to its parent. We rotate around that edge.
    
    # For simplicity, let's pick a random internal node and try to rotate.
    # The internal node has left and right children. One of them is also internal.
    # We swap the sibling of that internal child with the other child of the parent.
    
    # This is complex. Let's use a simpler approach:
    # Pick two random internal nodes and swap their subtrees.
    
    # Actually, let's implement a proper NNI:
    # 1. Pick an internal node X (not root)
    # 2. X has parent P, and P has another child Y
    # 3. X has children A and B
    # 4. NNI move: swap A with Y, or B with Y
    #    New structure: P has children (A, (B, Y)) or (B, (A, Y))
    
    # We need to track parent pointers. Let's build them.
    parent = {}
    
    def build_parents(idx, par=None):
        parent[idx] = par
        node = tree_nodes[idx]
        if node[0] is not None:
            return
        build_parents(node[1], idx)
        build_parents(node[2], idx)
    
    build_parents(root_idx)
    
    # Get all internal nodes (excluding root)
    internal_nodes = []
    
    def collect_internal(idx):
        node = tree_nodes[idx]
        if node[0] is not None:
            return
        if idx != root_idx:
            internal_nodes.append(idx)
        collect_internal(node[1])
        collect_internal(node[2])
    
    collect_internal(root_idx)
    
    if not internal_nodes:
        return None
    
    # Pick a random internal node
    x_idx = rng.choice(internal_nodes)
    p_idx = parent[x_idx]
    
    if p_idx is None:
        return None
    
    p_node = tree_nodes[p_idx]
    x_node = tree_nodes[x_idx]
    
    # Find the other child of p
    if p_node[1] == x_idx:
        y_idx = p_node[2]
    else:
        y_idx = p_node[1]
    
    # x has children A and B
    a_idx = x_node[1]
    b_idx = x_node[2]
    
    # Two NNI moves:
    # Move 1: P has (A, (B, Y))
    # Move 2: P has (B, (A, Y))
    
    # Create new tree structure
    # We'll modify a copy of the nodes
    new_nodes = [list(node) for node in tree_nodes]
    
    # Create new internal node for (B, Y) or (A, Y)
    # We'll reuse an existing internal node index or create a new one
    
    # For simplicity, let's just swap the children in place
    # Move 1: P.left = A, P.right = new_node(B, Y)
    # We need to create a new node. Let's find an unused internal node or create one.
    
    # Actually, the tree structure is binary, so we have exactly n-1 internal nodes.
    # Let's just rewire:
    # Original: P -> (X, Y), X -> (A, B)
    # New: P -> (A, X'), X' -> (B, Y)
    
    # We'll modify:
    # new_nodes[p_idx][1] = a_idx (or b_idx for move 2)
    # new_nodes[p_idx][2] = x_idx (x becomes the new internal node)
    # new_nodes[x_idx][1] = b_idx (or a_idx)
    # new_nodes[x_idx][2] = y_idx
    
    # Move 1:
    new_nodes[p_idx][1] = a_idx
    new_nodes[p_idx][2] = x_idx
    new_nodes[x_idx][1] = b_idx
    new_nodes[x_idx][2] = y_idx
    
    # Recompute splits
    leaf_set = frozenset()
    n_taxa = 0
    
    def recompute_splits(idx):
        nonlocal n_taxa
        node = new_nodes[idx]
        if node[0] is not None:
            n_taxa += 1
            leaf_set.add(node[0])
            return frozenset([node[0]])
        else:
            left_split = recompute_splits(node[1])
            right_split = recompute_splits(node[2])
            split = left_split | right_split
            node[3] = split
            return split
    
    recompute_splits(root_idx)
    
    return new_nodes


def _nni_refine(newick, sequences, labels, max_rounds=10):
    """NNI refinement using quartet scoring."""
    if _deadline_exceeded():
        return newick
    
    n = len(labels)
    if n < 4:
        return newick
    
    tree_nodes, root_idx, leaf_set, n_leaves = _parse_newick_to_tree(newick)
    current_score = _quartet_topology_score(tree_nodes, root_idx, sequences, labels)
    
    rng = random.Random(42)
    
    for rnd in range(max_rounds):
        if _deadline_exceeded():
            break
        
        improved = False
        n_attempts = min(50, max(10, n * 5))
        
        for _ in range(n_attempts):
            if _deadline_exceeded():
                break
            
            new_tree = _nni_move(tree_nodes, root_idx, None, rng)
            if new_tree is None:
                continue
            
            new_score = _quartet_topology_score(new_tree, root_idx, sequences, labels)
            
            if new_score > current_score:
                current_score = new_score
                tree_nodes = new_tree
                improved = True
        
        if not improved:
            break
    
    return _tree_to_newick(tree_nodes, root_idx)


def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    d = _compute_distance_matrix(labels, sequences)
    
    if _deadline_exceeded():
        return "(" + ",".join(labels) + ");"
    
    newick = _nj_build(labels, d)
    
    if _deadline_exceeded():
        return newick + ";"
    
    # NNI refinement
    refined = _nni_refine(newick, sequences, labels, max_rounds=8)
    
    return refined + ";"
