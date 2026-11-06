# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys

_START = None
_DEADLINE_FRACTION = 0.85 # Stop at 85% of 270s to be safe

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
    # Using uint8 for compactness
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    # Calculate pairwise differences efficiently using broadcasting
    # arr shape: (n, L), we want (n, n, L) comparison
    # Use chunked approach for memory efficiency with large n
    if n <= 64:
        diff = (arr[:, None, :] != arr[None, :, :])
        counts = diff.sum(axis=2)
    else:
        # Chunked computation to avoid O(n^2 * L) memory
        counts = np.zeros((n, n), dtype=np.int32)
        chunk_size = max(1, min(16, n))
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            diff_chunk = (arr[i:i_end, None, :] != arr[None, :, :])
            counts[i:i_end, :] = diff_chunk.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
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
    
    # Copy dist to dict for easy access/modification
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
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        # Record split
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        # Canonical form: lexicographically smaller frozenset
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

def _parse_newick(newick_str):
    """Parse Newick string into a tree structure with Node objects."""
    class Node:
        def __init__(self, name=None):
            self.name = name
            self.children = []
            self.parent = None
    
    root = None
    stack = []
    i = 0
    n = len(newick_str)
    
    while i < n:
        c = newick_str[i]
        if c == '(':
            new_node = Node()
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
            # After closing paren, might have a name or branch length
            j = i
            while j < n and newick_str[j] not in ',;':
                j += 1
            if i < j:
                # Could be name or branch length
                part = newick_str[i:j]
                if stack:
                    # This is a branch length or name for the node we just closed
                    # Check if it's a number (branch length)
                    try:
                        float(part)
                        # It's a branch length, ignore for topology
                    except ValueError:
                        # It's a name
                        stack[-1].name = part
                i = j
        elif c == ',':
            i += 1
        elif c == ';':
            i += 1
        else:
            # Read a name
            j = i
            while j < n and newick_str[j] not in '(),;':
                j += 1
            name = newick_str[i:j]
            if stack:
                if stack[-1].name is None:
                    stack[-1].name = name
                else:
                    # This might be a leaf name after a closing paren
                    pass
            else:
                root = Node(name)
            i = j
    
    return root

def _get_leaves(node, leaves=None):
    """Get all leaf nodes in the subtree."""
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

def _tree_splits(node, all_leaves):
    """Extract all non-trivial splits from a tree."""
    splits = set()
    n = len(all_leaves)
    
    def _recurse(node, parent_side=None):
        if node is None or not node.children:
            return
        for child in node.children:
            child_leaves = frozenset(_get_leaves(child))
            if 1 < len(child_leaves) < n - 1:
                complement = frozenset(all_leaves - child_leaves)
                # Canonical form
                if tuple(sorted(child_leaves)) < tuple(sorted(complement)):
                    splits.add(child_leaves)
                else:
                    splits.add(complement)
            _recurse(child, parent_side)
    
    _recurse(node)
    return splits

def _spr_refine(tree_str, dist, labels, max_iterations=100):
    """Apply SPR moves to improve the tree topology."""
    n = len(labels)
    if n <= 4:
        return tree_str
    
    # Parse the tree
    root = _parse_newick(tree_str)
    if root is None:
        return tree_str
    
    all_leaves = set(labels)
    
    # Build a proper tree representation for SPR
    # We'll work with a simpler representation
    
    best_tree = tree_str
    best_splits = _tree_splits(root, all_leaves)
    
    for iteration in range(max_iterations):
        if _time_left() < 1.0:
            break
        
        # Try SPR moves
        improved = False
        # For each internal edge, try pruning and regrafting
        # This is a simplified SPR - we'll try all possible regrafting positions
        
        # Get all internal nodes
        internal_nodes = []
        def _collect_internal(node):
            if node is None:
                return
            if node.children:
                internal_nodes.append(node)
                for child in node.children:
                    _collect_internal(child)
        _collect_internal(root)
        
        for parent in internal_nodes:
            for child in list(parent.children):
                if not child.children:
                    continue  # Skip leaf pruning for simplicity
                
                # Prune child from parent
                parent.children.remove(child)
                child.parent = None
                
                # Try regrafting at all other internal edges
                for reg_parent in internal_nodes:
                    if reg_parent == parent:
                        continue
                    for reg_child in reg_parent.children:
                        if reg_child == child:
                            continue
                        
                        # Regraft child between reg_parent and reg_child
                        reg_parent.children.append(child)
                        child.parent = reg_parent
                        
                        # Check if this creates a valid tree (no cycles, etc.)
                        # Simple check: make sure we can reach all leaves
                        test_leaves = _get_leaves(reg_parent)
                        if len(test_leaves) == len(set(test_leaves)):
                            # Compute new splits
                            test_root = _parse_newick(_tree_to_newick_simple(root))
                            if test_root:
                                test_splits = _tree_splits(test_root, all_leaves)
                                # Count common splits with best
                                common = len(test_splits & best_splits)
                                if common > len(best_splits):
                                    improved = True
                                    best_tree = _tree_to_newick_simple(root)
                                    best_splits = test_splits
                                    break
                        
                        # Revert
                        reg_parent.children.remove(child)
                        child.parent = None
                
                # Restore original
                parent.children.append(child)
                child.parent = parent
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_tree

def _tree_to_newick_simple(node):
    """Convert a Node tree to Newick string."""
    if node is None:
        return ""
    if not node.children:
        return node.name if node.name else ""
    child_strs = [_tree_to_newick_simple(c) for c in node.children]
    return "(" + ",".join(child_strs) + ")"

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    split_counts = {}
    
    # Count splits in the first tree (NJ)
    for s in splits_full:
        split_counts[s] = split_counts.get(s, 0) + 1
        
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 500  # Reduced to leave time for SPR
    
    # Time budget: 270s total.
    # Leave time for SPR refinement.
    while num_bootstraps < max_bootstraps and _time_left() > 10.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        
        # Update split counts
        for s in splits_boot:
            split_counts[s] = split_counts.get(s, 0) + 1
            
        num_bootstraps += 1
        
    if len(candidates) > 1:
        total_bootstraps = len(candidates)
        threshold = 0.5 * total_bootstraps
        
        # Identify strong splits
        strong_splits = set()
        for s, c in split_counts.items():
            if c > threshold:
                strong_splits.add(s)
                
        best_idx = 0
        best_score = -1
        
        for i in range(len(candidates)):
            splits_i = candidates[i][1]
            # Count how many splits in splits_i are in strong_splits
            common = len(splits_i & strong_splits)
            
            if common > best_score:
                best_score = common
                best_idx = i
                    
        best_tree = candidates[best_idx][0]
    else:
        best_tree = tree_full
    
    # Apply SPR refinement if time permits
    if _time_left() > 5.0 and n >= 5:
        try:
            refined_tree = _spr_refine(best_tree, d_full, labels, max_iterations=50)
            if refined_tree:
                best_tree = refined_tree
        except Exception:
            pass
    
    return best_tree

# EVOLVE-BLOCK-END
