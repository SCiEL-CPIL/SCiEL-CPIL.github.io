import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    """
    Compute Jukes-Cantor distance matrix.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n numpy array.
    """
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    
    # Hamming distances using broadcasting
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    
    # JC correction
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    
    # Symmetrize and zero diagonal
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

class _TreeNode:
    """Internal node in an unrooted binary tree represented as a rooted binary tree."""
    __slots__ = ['left', 'right', 'label', 'parent']
    def __init__(self):
        self.left = None
        self.right = None
        self.label = None
        self.parent = None

def _run_nj(dist, labels):
    """
    Neighbor Joining returning a _TreeNode structure.
    dist: n x n numpy array.
    labels: list of n labels.
    Returns: root _TreeNode (unrooted tree represented as rooted).
    """
    n = len(labels)
    if n == 1:
        node = _TreeNode()
        node.label = labels[0]
        return node
    if n == 2:
        root = _TreeNode()
        l = _TreeNode(); l.label = labels[0]
        r = _TreeNode(); r.label = labels[1]
        root.left = l; root.right = r
        l.parent = root; r.parent = root
        return root
    
    # Map original indices to current active nodes
    nodes = list(range(n))
    node_objs = {}
    for i in range(n):
        obj = _TreeNode()
        obj.label = labels[i]
        node_objs[i] = obj
    
    # Distance dict
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    new_id = n
    while len(nodes) > 2:
        m = len(nodes)
        # Totals
        tot = {}
        for x in nodes:
            tot[x] = 0.0
            for y in nodes:
                if y != x:
                    tot[x] += D[(x, y)]
        
        # Find best pair
        best = float('inf')
        a, b = None, None
        for i_idx, x in enumerate(nodes):
            for y in nodes[i_idx + 1:]:
                crit = (m - 2) * D[(x, y)] - tot[x] - tot[y]
                if crit < best:
                    best = crit
                    a, b = x, y
        
        # Create new node
        node = new_id
        new_id += 1
        obj = _TreeNode()
        obj.left = node_objs[a]
        obj.right = node_objs[b]
        node_objs[a].parent = obj
        node_objs[b].parent = obj
        node_objs[node] = obj
        
        # Update distances to new node
        rest = [x for x in nodes if x != a and x != b]
        d_ab = D[(a, b)]
        for x in rest:
            d_ax = D[(a, x)]
            d_bx = D[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            D[(node, x)] = d_new
            D[(x, node)] = d_new
        
        # Remove a, b, add node
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(node)
    
    a, b = nodes
    root = _TreeNode()
    root.left = node_objs[a]
    root.right = node_objs[b]
    node_objs[a].parent = root
    node_objs[b].parent = root
    
    return root

def _node_to_newick(node):
    """Convert _TreeNode to Newick string."""
    if node.label is not None:
        return node.label
    return f"({_node_to_newick(node.left)},{_node_to_newick(node.right)})"

def _newick_from_tree(root):
    return _node_to_newick(root) + ";"

def _get_internal_edges(root):
    """
    Get all internal edges of an unrooted tree represented as rooted.
    For an unrooted tree with n leaves, there are n-3 internal edges.
    Each internal edge connects two internal nodes.
    In the rooted representation, internal edges are:
    - Edges between an internal node and its child that is also internal
    """
    edges = []
    
    def walk(node):
        if node.label is not None:
            return
        if node.parent is not None and node.parent.label is None:
            # Edge from parent to this node (both internal)
            edges.append((node.parent, node))
        if node.left and node.left.label is None:
            edges.append((node, node.left))
            walk(node.left)
        if node.right and node.right.label is None:
            edges.append((node, node.right))
            walk(node.right)
    
    walk(root)
    return edges

def _fitch_cost_for_split(A, split_indices, n, L):
    """
    Compute Fitch parsimony cost for a split.
    A: n x L array of ints (0-3).
    split_indices: set of indices in the split.
    Returns: int cost.
    """
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    A_int = A.astype(np.int32)
    
    split_set = set(split_indices)
    split_bits = np.zeros(L, dtype=np.int32)
    other_bits = np.zeros(L, dtype=np.int32)
    
    for i in range(n):
        if i in split_set:
            split_bits |= bit_map[A_int[i]]
        else:
            other_bits |= bit_map[A_int[i]]
    
    overlap = (split_bits & other_bits)
    cost = np.sum(overlap == 0)
    return int(cost)

def _fitch_cost_tree(root, A, label_to_idx, n, L):
    """
    Compute total Fitch parsimony cost for a tree.
    """
    total_cost = 0
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    A_int = A.astype(np.int32)
    
    def compute_bitsets(node):
        """Returns bitset for subtree rooted at node."""
        if node.label is not None:
            idx = label_to_idx[node.label]
            return bit_map[A_int[idx]]
        left_bs = compute_bitsets(node.left)
        right_bs = compute_bitsets(node.right)
        # Fitch: if intersection non-empty, use it; else union, add cost
        inter = left_bs & right_bs
        if inter != 0:
            return inter
        else:
            nonlocal total_cost
            total_cost += 1
            return left_bs | right_bs
    
    compute_bitsets(root)
    return total_cost

def _nni_search(root, A, label_to_idx, n, L, deadline):
    """
    Perform NNI search on the tree.
    """
    total_cost = _fitch_cost_tree(root, A, label_to_idx, n, L)
    
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    A_int = A.astype(np.int32)
    
    def compute_bitsets(node, cache):
        """Returns bitset for subtree rooted at node, caching results."""
        if node in cache:
            return cache[node]
        if node.label is not None:
            idx = label_to_idx[node.label]
            cache[node] = bit_map[A_int[idx]]
            return cache[node]
        left_bs = compute_bitsets(node.left, cache)
        right_bs = compute_bitsets(node.right, cache)
        inter = left_bs & right_bs
        if inter != 0:
            cache[node] = inter
        else:
            cache[node] = left_bs | right_bs
        return cache[node]
    
    def fitch_cost_with_bitsets(root_node, cache):
        """Compute Fitch cost using precomputed bitsets."""
        cost = 0
        def walk(node):
            nonlocal cost
            if node.label is not None:
                return
            left_bs = cache[node.left]
            right_bs = cache[node.right]
            inter = left_bs & right_bs
            if inter == 0:
                cost += 1
            walk(node.left)
            walk(node.right)
        walk(root_node)
        return cost
    
    # Precompute bitsets
    cache = {}
    compute_bitsets(root, cache)
    current_cost = fitch_cost_with_bitsets(root, cache)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        edges = _get_internal_edges(root)
        
        for edge in edges:
            if time.time() > deadline - 5.0:
                break
            u, v = edge
            # u is parent of v in rooted representation
            # The four subtrees around this edge:
            # - subtree_v (v and its children)
            # - sibling of v (u's other child)
            # - parent of u (or the other side)
            # - sibling of u
            
            # For NNI, we need the four subtrees connected to the internal edge
            # In rooted representation, internal edge u-v where u is parent of v:
            # The four subtrees are:
            # 1. v's left subtree
            # 2. v's right subtree  
            # 3. u's other child (sibling of v)
            # 4. u's parent side (everything else)
            
            # Actually, let me think about this more carefully.
            # In an unrooted binary tree, an internal edge splits the tree into two parts.
            # For NNI, we swap the subtrees on one side.
            
            # Let's use a different approach: represent the tree as a set of splits
            # and do NNI in split space.
            
            # For now, let's do a simpler local search: try all possible reconnections.
            # This is getting complex. Let me use a different strategy.
            pass
    
    return root, current_cost

def _tree_to_splits(root, all_labels):
    """
    Convert a _TreeNode tree to a set of canonical splits.
    """
    all_set = set(all_labels)
    min_label = min(all_labels)
    splits = set()
    
    def get_leaf_set(node):
        if node.label is not None:
            return {node.label}
        return get_leaf_set(node.left) | get_leaf_set(node.right)
    
    def walk(node):
        if node.label is not None:
            return
        leaf_set = get_leaf_set(node)
        if 0 < len(leaf_set) < len(all_set):
            if min_label in leaf_set:
                splits.add(frozenset(leaf_set))
            else:
                splits.add(frozenset(all_set - leaf_set))
        if node.left:
            walk(node.left)
        if node.right:
            walk(node.right)
    
    walk(root)
    return splits

def _splits_to_newick(splits, all_labels):
    """
    Build Newick string from a set of compatible splits.
    """
    all_set = set(all_labels)
    
    def build(subset):
        if len(subset) == 1:
            return list(subset)[0]
        
        for split in splits:
            A = split & subset
            if 0 < len(A) < len(subset):
                B = subset - A
                return f"({build(A)},{build(B)})"
        
        return "(" + ",".join(sorted(subset)) + ")"
    
    return build(all_set) + ";"

def _nni_in_split_space(splits, A, label_to_idx, all_labels, n, L, deadline):
    """
    Perform NNI search in split space.
    splits: set of frozensets (canonical splits)
    Returns improved splits.
    """
    all_set = set(all_labels)
    min_label = min(all_labels)
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    A_int = A.astype(np.int32)
    
    def fitch_cost_split(split):
        """Compute Fitch cost for a single split."""
        split_set = set(split)
        split_bits = np.zeros(L, dtype=np.int32)
        other_bits = np.zeros(L, dtype=np.int32)
        
        for i in range(n):
            if label_to_idx[all_labels[i]] in split_set:
                split_bits |= bit_map[A_int[i]]
            else:
                other_bits |= bit_map[A_int[i]]
        
        overlap = (split_bits & other_bits)
        return int(np.sum(overlap == 0))
    
    def total_fitch_cost(split_list):
        """Compute total Fitch cost for a set of splits."""
        # For a set of compatible splits forming a tree, the total cost is
        # the sum of costs of each split (each internal edge contributes its cost)
        total = 0
        for s in split_list:
            total += fitch_cost_split(s)
        return total
    
    current_splits = list(splits)
    current_cost = total_fitch_cost(current_splits)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try removing each split and see if we can add a better one
        # Or try NNI moves
        
        # NNI: for each pair of adjacent splits, try swapping
        # In split space, NNI corresponds to replacing two splits with two different compatible splits
        
        # Simpler approach: for each split, try replacing it with a neighboring split
        # A neighboring split is one that differs by moving one leaf
        
        for i in range(len(current_splits)):
            if time.time() > deadline - 5.0:
                break
            s = current_splits[i]
            # Try moving one leaf from s to the other side
            leaf_list = list(s)
            for leaf in leaf_list:
                if len(s) <= 2:
                    break
                new_s = frozenset(s - {leaf})
                # Canonicalize
                if min_label not in new_s:
                    new_s = frozenset(all_set - new_s)
                
                if new_s in splits:
                    continue
                
                # Check compatibility with all other splits
                compatible = True
                for j in range(len(current_splits)):
                    if j == i:
                        continue
                    s2 = current_splits[j]
                    i1 = new_s & s2
                    i2 = new_s & (all_set - s2)
                    i3 = (all_set - new_s) & s2
                    i4 = (all_set - new_s) & (all_set - s2)
                    if len(i1) > 0 and len(i2) > 0 and len(i3) > 0 and len(i4) > 0:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Compute new cost
                new_cost = current_cost - fitch_cost_split(s) + fitch_cost_split(new_s)
                if new_cost < current_cost:
                    current_splits[i] = new_s
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    
    return set(current_splits), current_cost

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    A = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            A[i, j] = char_map[c]
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    all_set = set(labels)
    min_label = min(labels)
    
    start_time = time.time()
    deadline = 250.0
    
    # 1. NJ tree
    D = _jc_distance_matrix(A)
    root = _run_nj(D, labels)
    tree_nj = _newick_from_tree(root)
    splits_nj = _tree_to_splits(root, labels)
    
    # 2. NNI search in split space
    if time.time() < deadline - 10.0:
        splits_refined, cost = _nni_in_split_space(splits_nj, A, label_to_idx, labels, n, L, deadline)
    else:
        splits_refined = splits_nj
    
    # 3. Build final tree
    tree = _splits_to_newick(splits_refined, labels)
    return tree

# EVOLVE-BLOCK-END
