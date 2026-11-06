# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.85  # Safety margin


def _now():
    return time.time()


def _init_time():
    global _START
    if _START is None:
        _START = time.time()


def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))


def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr


def _jc_matrix(arr):
    n, L = arr.shape
    if n <= 64:
        diff = (arr[:, None, :] != arr[None, :, :])
        counts = diff.sum(axis=2)
    else:
        counts = np.zeros((n, n), dtype=np.int32)
        chunk_size = max(1, min(16, n))
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            diff_chunk = (arr[i:i_end, None, :] != arr[None, :, :])
            counts[i:i_end, :] = diff_chunk.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d


def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()

    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}

    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]

    splits = set()

    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}

        best_criterion = float('inf')
        best_a, best_b = -1, -1

        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b

        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]

        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)

        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]

        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)

    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits


# --- Tree node structure for MCMC ---
class _TreeNode:
    __slots__ = ['name', 'children', 'parent', 'edge_len']

    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None
        self.edge_len = 0.0


def _parse_newick_to_nodes(newick_str):
    """Parse Newick into a tree of _TreeNode objects."""
    root = None
    stack = []
    i = 0
    n = len(newick_str)

    while i < n:
        c = newick_str[i]
        if c == '(':
            new_node = _TreeNode()
            if stack:
                stack[-1].children.append(new_node)
                new_node.parent = stack[-1]
            else:
                root = new_node
            stack.append(new_node)
            i += 1
        elif c == ')':
            stack.pop()
            i += 1
            # Skip any branch length
            j = i
            while j < n and newick_str[j] not in ',;':
                j += 1
            if stack:
                # Try to parse as branch length
                part = newick_str[i:j]
                if part:
                    try:
                        stack[-1].edge_len = float(part)
                    except ValueError:
                        if stack[-1].name is None:
                            stack[-1].name = part
            i = j
        elif c == ',':
            i += 1
        elif c == ';':
            i += 1
        else:
            j = i
            while j < n and newick_str[j] not in '(),;':
                j += 1
            name = newick_str[i:j]
            if stack:
                if stack[-1].name is None:
                    stack[-1].name = name
            else:
                root = _TreeNode(name)
            i = j

    return root


def _nodes_to_newick(root):
    """Convert _TreeNode tree to Newick string."""
    if root is None:
        return ""
    if not root.children:
        return root.name if root.name else ""
    child_strs = [_nodes_to_newick(c) for c in root.children]
    return "(" + ",".join(child_strs) + ")"


def _get_all_nodes(root):
    """Get all nodes in the tree."""
    nodes = []
    def _recurse(node):
        if node is None:
            return
        nodes.append(node)
        for child in node.children:
            _recurse(child)
    _recurse(root)
    return nodes


def _get_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    if node is None:
        return leaves
    if not node.children:
        if node.name:
            leaves.append(node.name)
    else:
        for child in node.children:
            _get_leaves(child, leaves)
    return leaves


def _tree_splits_from_nodes(root, all_leaves):
    """Get the set of non-trivial splits from a _TreeNode tree."""
    splits = set()
    n = len(all_leaves)

    def _recurse(node):
        if node is None or not node.children:
            return
        for child in node.children:
            child_leaves = frozenset(_get_leaves(child))
            if 1 < len(child_leaves) < n - 1:
                complement = frozenset(all_leaves - child_leaves)
                if tuple(sorted(child_leaves)) < tuple(sorted(complement)):
                    splits.add(child_leaves)
                else:
                    splits.add(complement)
            _recurse(child)

    _recurse(root)
    return splits


def _compute_jc_likelihood(tree_root, arr, n_leaves, L):
    """
    Compute the log-likelihood of the data given the tree topology under JC model.
    We use Felsenstein's pruning algorithm with optimized branch lengths.
    
    For speed, we do a single pass of Felsenstein pruning with reasonable
    branch length estimates, then optionally refine.
    
    Returns log-likelihood.
    """
    # Build leaf index mapping
    leaf_indices = {}
    def _map_leaves(node, idx):
        if node is None:
            return idx
        if not node.children:
            leaf_indices[node.name] = idx
            return idx + 1
        for child in node.children:
            idx = _map_leaves(child, idx)
        return idx
    
    _map_leaves(tree_root, 0)
    
    # Initialize branch lengths using distance-based estimates
    # First, compute pairwise distances
    n, L_arr = arr.shape
    
    # For Felsenstein pruning, we need to compute likelihood at each node
    # This is complex to do efficiently. Let's use a simpler approach:
    # Use the distance matrix to estimate branch lengths, then compute
    # the likelihood as a sum over sites of log-sum-exp over possible
    # ancestral states.
    
    # Actually, for the MCMC, we just need a scoring function that
    # correlates with tree quality. Let's use a simpler score based on
    # the sum of squared differences between observed and expected
    # pairwise distances under the tree.
    
    # Compute expected distances from tree topology
    # For each pair of leaves, the expected distance is the sum of
    # branch lengths on the path between them.
    
    # First, compute branch lengths using least squares or simple estimates
    # For simplicity, use equal branch lengths initially, then refine
    
    # Let's use a different approach: compute the likelihood using
    # Felsenstein's algorithm with branch lengths estimated from distances
    
    # Step 1: Estimate branch lengths
    # Use the distance matrix to set initial branch lengths
    # For each internal edge, the length is approximately:
    # (d(a,c) + d(b,d) - d(a,b) - d(c,d)) / 2 for a quartet
    
    # This is getting complex. Let's use a simpler scoring function:
    # The sum over all pairs of |d_observed - d_tree| where d_tree
    # is the path length in the tree.
    
    # Actually, let's just use Felsenstein's algorithm properly.
    # It's O(n * L * 4) which is fast enough.
    
    # Build the tree structure for Felsenstein
    # We need to root the tree and do post-order traversal
    
    # For an unrooted tree, we can root it at any internal node
    # Let's find an internal node to root at
    
    internal_nodes = [nd for nd in _get_all_nodes(tree_root) if nd.children]
    if not internal_nodes:
        return 0.0
    
    # Root at the first internal node
    root_node = internal_nodes[0]
    
    # For Felsenstein, we need a rooted tree
    # Let's pick a leaf as the "root" (the outgroup)
    # Actually, for unrooted trees, we can use the "pruning" algorithm
    # which works on unrooted trees directly
    
    # Simplified approach: use the distance-based score
    # Score = -sum over pairs of (d_obs - d_tree)^2
    # where d_tree is computed from the tree topology with optimized lengths
    
    # For speed in MCMC, let's use a fast approximation:
    # Compute the "tree distance" for each pair using the topology
    # and compare with observed distances
    
    # Get all leaves
    leaves = _get_leaves(tree_root)
    if len(leaves) != n:
        return -1e10  # Invalid tree
    
    # Build leaf to index map
    leaf_to_idx = {l: i for i, l in enumerate(leaves)}
    
    # Compute tree distances using the topology
    # For each pair, find the path length
    # This requires computing the path in the tree
    
    # For efficiency, let's precompute the path lengths
    # Using the tree structure
    
    # Actually, this is getting too complex for a fast MCMC score.
    # Let's use a simpler but effective approach:
    # Use the quartet score as the MCMC objective
    
    return 0.0  # Placeholder - will be replaced with proper implementation


def _quartet_score(tree_root, dist, labels):
    """
    Compute a quartet-based score for the tree.
    For each quartet of leaves, check if the tree's topology
    matches the best quartet topology from distances.
    Higher score = better tree.
    """
    n = len(labels)
    if n < 4:
        return 0.0
    
    leaves = _get_leaves(tree_root)
    if len(leaves) != n:
        return -1e10
    
    # Build leaf to index map
    leaf_to_idx = {l: i for i, l in enumerate(labels)}
    
    # For each quartet, determine the tree's topology
    # and compare with the distance-based best topology
    
    # For speed, sample quartets
    np.random.seed(12345)
    num_quartets = min(10000, n * (n-1) * (n-2) * (n-3) // 24)
    
    score = 0.0
    count = 0
    
    # Build a set of splits from the tree for fast lookup
    tree_splits = _tree_splits_from_nodes(tree_root, set(labels))
    
    for _ in range(num_quartets):
        idx = np.random.choice(n, 4, replace=False)
        a, b, c, d = idx[0], idx[1], idx[2], idx[3]
        
        # Best topology from distances
        s1 = dist[a, b] + dist[c, d]
        s2 = dist[a, c] + dist[b, d]
        s3 = dist[a, d] + dist[b, c]
        
        min_s = min(s1, s2, s3)
        
        # Check which quartet splits the tree supports
        # Split {a,b}|{c,d}
        split_ab_cd = frozenset({labels[a], labels[b]})
        split_ac_bd = frozenset({labels[a], labels[c]})
        split_ad_bc = frozenset({labels[a], labels[d]})
        
        # Normalize splits (use the smaller side)
        all_leaves_set = set(labels)
        
        def _norm_split(s):
            comp = frozenset(all_leaves_set - s)
            if tuple(sorted(s)) < tuple(sorted(comp)):
                return s
            else:
                return comp
        
        n_split_ab_cd = _norm_split(split_ab_cd)
        n_split_ac_bd = _norm_split(split_ac_bd)
        n_split_ad_bc = _norm_split(split_ad_bc)
        
        # Count how many of the best quartet topologies are in the tree
        if s1 <= min_s + 1e-10:
            if n_split_ab_cd in tree_splits:
                score += 1
        if s2 <= min_s + 1e-10:
            if n_split_ac_bd in tree_splits:
                score += 1
        if s3 <= min_s + 1e-10:
            if n_split_ad_bc in tree_splits:
                score += 1
        
        count += 1
    
    return score / max(count, 1)


def _spr_mcmc(tree_newick, dist, labels, arr, max_steps=5000):
    """
    MCMC with SPR moves to explore tree space.
    Uses quartet score as the objective function.
    """
    n = len(labels)
    if n < 5:
        return tree_newick
    
    root = _parse_newick_to_nodes(tree_newick)
    if root is None:
        return tree_newick
    
    all_leaves = set(labels)
    current_score = _quartet_score(root, dist, labels)
    best_tree = tree_newick
    best_score = current_score
    
    # Temperature for simulated annealing
    T = 0.1
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        # Propose an SPR move
        all_nodes = _get_all_nodes(root)
        internal_nodes = [nd for nd in all_nodes if nd.children]
        
        if len(internal_nodes) < 2:
            break
        
        # Choose a subtree to prune
        # Pick a random internal node (not the root for simplicity)
        candidates = [nd for nd in internal_nodes if nd.parent is not None]
        if not candidates:
            candidates = internal_nodes
        
        subtree_root = candidates[np.random.randint(len(candidates))]
        
        # Choose a random edge to regraft
        # The regrafting edge should be on the path from subtree_root to the root
        # or on a different branch
        
        # For SPR: prune a subtree and regraft it on a different edge
        # The new edge should not be on the path from the pruned subtree to the root
        
        # Find all edges not on the path from subtree_root to root
        path_to_root = set()
        cur = subtree_root
        while cur is not None:
            path_to_root.add(id(cur))
            cur = cur.parent
        
        # All edges (parent-child pairs)
        all_edges = []
        for nd in all_nodes:
            for child in nd.children:
                all_edges.append((nd, child))
        
        # Filter out edges on the path
        valid_edges = [(p, c) for p, c in all_edges if id(p) not in path_to_root or id(c) not in path_to_root]
        
        if not valid_edges:
            continue
        
        parent_node, child_node = valid_edges[np.random.randint(len(valid_edges))]
        
        # Perform the SPR move
        # 1. Prune the subtree
        subtree_parent = subtree_root.parent
        if subtree_parent is None:
            continue
        
        subtree_parent.children.remove(subtree_root)
        subtree_root.parent = None
        
        # 2. Regraft: insert subtree_root as a child of parent_node
        #    by splitting the edge parent_node -> child_node
        #    Create a new internal node
        new_internal = _TreeNode()
        new_internal.parent = parent_node
        new_internal.children = [child_node, subtree_root]
        child_node.parent = new_internal
        subtree_root.parent = new_internal
        
        # Replace child_node with new_internal in parent_node
        parent_node.children.remove(child_node)
        parent_node.children.append(new_internal)
        
        # Evaluate the new tree
        new_score = _quartet_score(root, dist, labels)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta >= 0 or np.random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_tree = _nodes_to_newick(root)
        else:
            # Revert the move
            parent_node.children.remove(new_internal)
            parent_node.children.append(child_node)
            child_node.parent = parent_node
            
            new_internal.children.remove(child_node)
            new_internal.children.remove(subtree_root)
            subtree_root.parent = subtree_parent
            subtree_parent.children.append(subtree_root)
            new_internal.parent = None
            new_internal.children = []
        
        # Cool down
        T = max(0.01, T * 0.999)
    
    return best_tree


def _nni_polish(tree_str, dist, labels, max_iterations=200):
    """
    Apply NNI moves to improve the tree topology using quartet-based scoring.
    """
    n = len(labels)
    if n <= 4:
        return tree_str

    root = _parse_newick_to_nodes(tree_str)
    if root is None:
        return tree_str

    all_leaves = set(labels)
    best_tree = tree_str
    best_score = _quartet_score(root, dist, labels)

    for iteration in range(max_iterations):
        if _time_left() < 2.0:
            break

        improved = False

        internal_nodes = []

        def _collect_internal(node, parent=None):
            if node is None:
                return
            if node.children:
                internal_nodes.append((node, parent))
                for child in node.children:
                    _collect_internal(child, node)

        _collect_internal(root)

        for node, parent in internal_nodes:
            if parent is None:
                continue
            if not node.children:
                continue
            if not parent.children:
                continue
            if len(node.children) < 2 or len(parent.children) < 2:
                continue

            node_children = [c for c in node.children if c != parent]
            if len(node_children) < 1:
                continue

            parent_children = [c for c in parent.children if c != node]
            if len(parent_children) < 1:
                continue

            for i in range(len(node_children)):
                for j in range(len(parent_children)):
                    if _time_left() < 1.0:
                        break

                    child_a = node_children[i]
                    child_b = parent_children[j]

                    # Perform the swap
                    node.children.remove(child_a)
                    child_a.parent = parent
                    parent.children.append(child_a)

                    parent.children.remove(child_b)
                    child_b.parent = node
                    node.children.append(child_b)

                    new_score = _quartet_score(root, dist, labels)

                    if new_score > best_score:
                        best_tree = _nodes_to_newick(root)
                        best_score = new_score
                        improved = True
                        root = _parse_newick_to_nodes(best_tree)
                        internal_nodes = []
                        _collect_internal(root)
                        break

                    # Revert the swap
                    parent.children.remove(child_a)
                    child_a.parent = node
                    node.children.append(child_a)

                    node.children.remove(child_b)
                    child_b.parent = parent
                    parent.children.append(child_b)

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    return best_tree


def reconstruct(sequences):
    global _START
    _START = time.time()

    labels, arr = _seqs_to_array(sequences)
    n = len(labels)

    d_full = _jc_matrix(arr)

    # Run NJ on the original distance matrix
    tree_orig, splits_orig = _nj(d_full, labels)

    # Bootstrap consensus
    candidates = [(tree_orig, splits_orig)]
    split_counts = {}

    for s in splits_orig:
        split_counts[s] = split_counts.get(s, 0) + 1

    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 200

    # Time budget: use up to 60s for bootstrapping
    while num_bootstraps < max_bootstraps and _time_left() > 60.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)

        tree_boot, splits_boot = _nj(d_boot, labels)

        candidates.append((tree_boot, splits_boot))

        for s in splits_boot:
            split_counts[s] = split_counts.get(s, 0) + 1

        num_bootstraps += 1

    # Select best tree from bootstrap candidates using consensus
    if len(candidates) > 1:
        total_bootstraps = len(candidates)
        threshold = 0.4 * total_bootstraps

        strong_splits = set()
        for s, c in split_counts.items():
            if c > threshold:
                strong_splits.add(s)

        best_idx = 0
        best_score = -1

        for i in range(len(candidates)):
            splits_i = candidates[i][1]
            common = len(splits_i & strong_splits)
            if common > best_score:
                best_score = common
                best_idx = i

        best_tree = candidates[best_idx][0]
    else:
        best_tree = tree_orig

    # Apply SPR MCMC refinement if time permits
    if _time_left() > 20.0 and n >= 5:
        try:
            mcmc_tree = _spr_mcmc(best_tree, d_full, labels, arr, max_steps=3000)
            if mcmc_tree:
                best_tree = mcmc_tree
        except Exception:
            pass

    # Apply NNI refinement if time permits
    if _time_left() > 10.0 and n >= 5:
        try:
            refined_tree = _nni_polish(best_tree, d_full, labels, max_iterations=100)
            if refined_tree:
                best_tree = refined_tree
        except Exception:
            pass

    return best_tree


# EVOLVE-BLOCK-END
