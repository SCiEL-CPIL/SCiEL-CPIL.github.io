"""Phylogenetic topology recovery: NJ initialization + ML-based NNI local search."""
import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255  # safety margin before 270s kill

def _check_deadline():
    return time.time() < _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _parse_newick(newick):
    """Parse a Newick string into a nested tuple structure."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    def parse(s):
        if s[0] == '(':
            depth = 0
            i = 1
            parts = []
            current = ''
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                    current += s[i]
                elif s[i] == ')':
                    depth -= 1
                    if depth == -1:
                        break
                    current += s[i]
                elif s[i] == ',' and depth == 0:
                    parts.append(current)
                    current = ''
                else:
                    current += s[i]
                i += 1
            if i < len(s):
                i += 1
            rest = s[i:]
            if rest.startswith(':'):
                j = 0
                while j < len(rest) and rest[j] not in (',', '(', ')', ';'):
                    j += 1
            children = [parse(p) for p in parts]
            return ('internal', children)
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label)
    
    return parse(newick)


def _build_adjacency(tree):
    """Build adjacency list from tree structure."""
    adj = {}
    node_types = {}
    counter = [0]
    
    def assign_id(node, parent_id=None):
        if node[0] == 'leaf':
            nid = ('leaf', node[1])
            node_types[nid] = ('leaf', node[1])
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            return nid
        else:
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            for child in node[1]:
                cid = assign_id(child, nid)
            return nid
    
    root = assign_id(tree)
    node_ids = list(adj.keys())
    return adj, node_ids, node_types


def _root_tree(adj, node_ids, node_types):
    """Root the tree and build parent/children structure."""
    root = None
    for nid in node_ids:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = node_ids[0]
    
    parent = {root: None}
    children = {nid: [] for nid in node_ids}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)
    
    return root, parent, children


def _postorder(root, children):
    """Get postorder traversal."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order


def _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, base_length):
    """Compute ML log-likelihood using vectorized NumPy operations.
    
    seq_array: (n_taxa, n_sites) int array with values 0-3
    node_types: maps node_id -> ('leaf', label) or ('internal', None)
    """
    # Build rooted structure
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Map leaf labels to indices in seq_array
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    # Precompute transition probabilities
    exp_term = math.exp(-4.0 * base_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    
    prob = {}
    
    # Initialize leaves
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            p = np.zeros((n_sites, 4), dtype=np.float64)
            obs = seq_array[idx]
            p[np.arange(n_sites), obs] = 1.0
            prob[nid] = p
        else:
            prob[nid] = None
    
    # Process internal nodes in postorder
    for nid in order:
        if node_types[nid][0] == 'leaf':
            continue
        
        child_list = children[nid]
        if not child_list:
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            cp = prob[child]
            row_sum = cp.sum(axis=1, keepdims=True)
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll


def _ml_score_with_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, branch_lengths):
    """Compute ML log-likelihood with per-branch lengths.
    
    branch_lengths: dict mapping (node_a, node_b) -> length, where node_a < node_b
    """
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    prob = {}
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            p = np.zeros((n_sites, 4), dtype=np.float64)
            obs = seq_array[idx]
            p[np.arange(n_sites), obs] = 1.0
            prob[nid] = p
        else:
            prob[nid] = None
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            continue
        
        child_list = children[nid]
        if not child_list:
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            # Get branch length between nid and child
            key = tuple(sorted([nid, child]))
            bl = branch_lengths.get(key, 0.1)
            exp_term = math.exp(-4.0 * bl / 3.0)
            p_same = 0.25 + 0.75 * exp_term
            p_diff = (1.0 - exp_term) / 3.0
            
            cp = prob[child]
            row_sum = cp.sum(axis=1, keepdims=True)
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll


def _estimate_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels):
    """Estimate branch lengths using distance-based method.
    
    Returns dict mapping (node_a, node_b) -> length
    """
    # For each edge, estimate length as half the difference of distances
    # We use a simple approach: for each internal edge, estimate based on
    # the average distance to leaves on each side
    
    branch_lengths = {}
    
    # Get leaf set
    leaves = [nid for nid in node_ids if node_types[nid][0] == 'leaf']
    
    # For each edge in the tree, compute the set of leaves on each side
    # and estimate branch length
    
    for u in node_ids:
        for v in adj[u]:
            if u >= v:
                continue
            key = (u, v)
            
            # Temporarily remove edge to find connected components
            # Simpler: use the parent-child structure
            pass
    
    # Alternative: use the distance matrix to estimate
    # For each pair of leaves, the path length should match the distance
    # We'll use a least-squares approach
    
    # Build edge list
    edges = []
    for u in node_ids:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    # For each edge, determine which leaves are on each side
    # Use BFS/DFS to find connected components when edge is removed
    
    leaf_indices = {nid: i for i, nid in enumerate(leaves)}
    leaf_labels = {nid: node_types[nid][1] for nid in leaves}
    
    # For each edge, compute the set of leaves reachable from each endpoint
    # without going through the edge
    edge_leaf_sets = {}
    for u, v in edges:
        # Find leaves reachable from u without going through v
        visited_u = set()
        stack = [u]
        while stack:
            node = stack.pop()
            if node in visited_u:
                continue
            visited_u.add(node)
            for nbr in adj[node]:
                if nbr != v and nbr not in visited_u:
                    stack.append(nbr)
        
        # Find leaves reachable from v without going through u
        visited_v = set()
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited_v:
                continue
            visited_v.add(node)
            for nbr in adj[node]:
                if nbr != u and nbr not in visited_v:
                    stack.append(nbr)
        
        leaves_u = [nid for nid in visited_u if nid in leaf_indices]
        leaves_v = [nid for nid in visited_v if nid in leaf_indices]
        
        edge_leaf_sets[(u, v)] = (leaves_u, leaves_v)
    
    # For each edge, estimate length as:
    # avg distance from leaves_u to leaves_v - (avg distance within leaves_u + avg distance within leaves_v) / 2
    # This is a rough estimate
    
    for u, v in edges:
        leaves_u, leaves_v = edge_leaf_sets[(u, v)]
        
        if not leaves_u or not leaves_v:
            branch_lengths[(u, v)] = 0.1
            continue
        
        # Average distance from leaves_u to leaves_v
        cross_dists = []
        for lu in leaves_u:
            for lv in leaves_v:
                label_u = leaf_labels[lu]
                label_v = leaf_labels[lv]
                d = distance_matrix.get((label_u, label_v), 0.0)
                cross_dists.append(d)
        
        avg_cross = sum(cross_dists) / len(cross_dists) if cross_dists else 0.1
        
        # Average distance within leaves_u
        if len(leaves_u) > 1:
            within_u = []
            for i, lu in enumerate(leaves_u):
                for lv in leaves_u[i+1:]:
                    label_u = leaf_labels[lu]
                    label_v = leaf_labels[lv]
                    d = distance_matrix.get((label_u, label_v), 0.0)
                    within_u.append(d)
            avg_within_u = sum(within_u) / len(within_u) if within_u else 0.0
        else:
            avg_within_u = 0.0
        
        # Average distance within leaves_v
        if len(leaves_v) > 1:
            within_v = []
            for i, lv in enumerate(leaves_v):
                for lv2 in leaves_v[i+1:]:
                    label_v = leaf_labels[lv]
                    label_v2 = leaf_labels[lv2]
                    d = distance_matrix.get((label_v, label_v2), 0.0)
                    within_v.append(d)
            avg_within_v = sum(within_v) / len(within_v) if within_v else 0.0
        else:
            avg_within_v = 0.0
        
        # Estimate branch length
        bl = max(0.01, (avg_cross - avg_within_u - avg_within_v) / 2.0)
        bl = min(bl, 1.0)  # Cap at 1.0
        branch_lengths[(u, v)] = bl
    
    return branch_lengths


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    For internal edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    NNI1: swap b and c -> changes split from (a,b)|(c,d) to (a,c)|(b,d)
    NNI2: swap b and d -> changes split from (a,b)|(c,d) to (a,d)|(b,c)
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        # Swap b and c
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        # Swap b and d
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _spr_local_search(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels, max_iterations=30):
    """Perform SPR local search to improve tree topology using ML scoring with per-branch lengths."""
    # Estimate initial branch lengths
    branch_lengths = _estimate_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels)
    current_ll = _ml_score_with_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, branch_lengths)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_ll = current_ll
        best_adj = None
        best_bl = None
        
        # Try all SPR moves: prune a subtree and regraft elsewhere
        # For each node u (not root), prune it and try regrafting on all edges
        root, parent, children = _root_tree(adj, node_ids, node_types)
        
        for u in node_ids:
            if u == root or node_types[u][0] == 'leaf':
                continue
            
            # Get parent of u
            p = parent[u]
            if p is None:
                continue
            
            # Prune u: remove edge (u, p)
            # New subtree rooted at u, and the rest of the tree
            
            # Try regrafting u on each edge in the remaining tree
            # The remaining tree has all nodes except u's subtree
            # For simplicity, try regrafting on edges adjacent to p
            
            # Get neighbors of p excluding u
            p_nbrs = [x for x in adj[p] if x != u]
            
            for v in p_nbrs:
                # Reattach u to v: create new edge (u, v) and adjust
                # This is a simple SPR move
                new_adj = {k: set(v) for k, v in adj.items()}
                
                # Remove edge (u, p)
                new_adj[u].discard(p)
                new_adj[p].discard(u)
                
                # Add edge (u, v)
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                # Check if this is a valid tree (no cycles)
                # For now, assume it's valid and score it
                new_bl = _estimate_branch_lengths(new_adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels)
                ll = _ml_score_with_branch_lengths(new_adj, node_ids, node_types, seq_array, n_sites, new_bl)
                
                if ll > best_ll + 1e-6:
                    best_ll = ll
                    best_adj = new_adj
                    best_bl = new_bl
        
        if best_adj is not None:
            adj = best_adj
            branch_lengths = best_bl
            current_ll = best_ll
            improved = True
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    # Find a leaf to start from
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    # Build Newick by traversing from root
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in sorted(adj[nid], key=lambda x: str(x)):
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute distance matrix using JC correction
    matrix = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                d = _jc_distance(sequences[a], sequences[b])
                matrix[(a, b)] = d
                matrix[(b, a)] = d
    
    # Step 2: Fast Up-And-Down (FUAD) pruning to remove noisy taxa
    # This improves NJ initialization by removing long/noisy branches first
    nodes = list(labels)
    pruned_taxa = []
    prune_threshold = 0.15  # fraction of mean distance to consider "long"
    
    while len(nodes) > max(6, int(0.6 * n_taxa)) and _check_deadline():
        # Compute mean pairwise distance
        if len(nodes) < 2:
            break
        total_d = sum(matrix[(a, b)] for i, a in enumerate(nodes) for b in nodes[i+1:])
        mean_d = total_d / (len(nodes) * (len(nodes) - 1) / 2)
        
        # Find the taxon with highest mean distance to others
        worst_taxon = None
        worst_mean = -1
        for a in nodes:
            mean_to_others = sum(matrix[(a, b)] for b in nodes if b != a) / (len(nodes) - 1)
            if mean_to_others > worst_mean:
                worst_mean = mean_to_others
                worst_taxon = a
        
        # Only prune if the worst taxon is significantly longer than average
        if worst_taxon is not None and worst_mean > mean_d * (1 + prune_threshold):
            pruned_taxa.append(worst_taxon)
            nodes.remove(worst_taxon)
        else:
            break
    
    # Step 3: Neighbor-Joining on pruned set
    nj_nodes = list(nodes)
    trees = {name: name for name in nj_nodes}
    
    while len(nj_nodes) > 3:
        if not _check_deadline():
            break
        n = len(nj_nodes)
        total = {a: sum(matrix[(a, b)] for b in nj_nodes if b != a) for a in nj_nodes}
        best = None
        for i, a in enumerate(nj_nodes):
            for b in nj_nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nj_nodes)}_{a}_{b}"
        rest = [x for x in nj_nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nj_nodes = rest + [new]
        trees[new] = joined
    
    # Step 4: Reinsert pruned taxa using nearest-neighbor join
    # For each pruned taxon, find its best attachment point in the current tree
    # We do this by computing its distance to all remaining taxa and attaching to the closest
    if pruned_taxa:
        # Rebuild full node list with internal nodes
        all_current_nodes = list(nj_nodes)
        
        for pt in pruned_taxa:
            if not _check_deadline():
                break
            
            # Find the taxon (or internal node) closest to pt
            best_target = None
            best_dist = float('inf')
            
            for candidate in all_current_nodes:
                if candidate in trees and trees[candidate].startswith('('):
                    # Internal node: estimate distance as min distance to its leaf descendants
                    # Extract leaf labels from the subtree
                    subtree = trees[candidate]
                    # Simple extraction: find all t\d+ labels in subtree
                    import re
                    leaf_labels = re.findall(r't\d+', subtree)
                    if leaf_labels:
                        min_d = min(matrix.get((pt, ll), float('inf')) for ll in leaf_labels)
                        if min_d < best_dist:
                            best_dist = min_d
                            best_target = candidate
                else:
                    # Leaf node
                    d = matrix.get((pt, candidate), float('inf'))
                    if d < best_dist:
                        best_dist = d
                        best_target = candidate
            
            if best_target is not None:
                # Attach pt to best_target
                # This creates a new internal node
                new_internal = f"ins_{pt}_{best_target}"
                trees[new_internal] = f"({trees[best_target]},{pt})"
                # Replace best_target in the tree structure
                # We need to update the parent reference
                # For simplicity, we'll rebuild the tree string
                trees[best_target] = trees[new_internal]
                # Add new_internal to nodes if it's a root-level node
                if best_target in nj_nodes:
                    nj_nodes[nj_nodes.index(best_target)] = new_internal
                all_current_nodes.append(new_internal)
    
    # Build initial Newick from the current tree structure
    # Find the root (the node that contains all taxa)
    # The root should be the last internal node created or the one with all leaves
    root_node = None
    for node in nj_nodes:
        subtree = trees[node]
        leaf_count = len(set(re.findall(r't\d+', subtree)))
        if leaf_count == n_taxa:
            root_node = node
            break
    
    if root_node is None:
        # Fallback: use the first node
        root_node = nj_nodes[0]
    
    newick = trees[root_node] + ";"
    
    # Ensure all taxa are present
    for label in labels:
        if label not in newick:
            # Fallback to simple NJ without pruning
            nodes = list(labels)
            trees = {name: name for name in labels}
            while len(nodes) > 3:
                if not _check_deadline():
                    break
                n = len(nodes)
                total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
                best = None
                for i, a in enumerate(nodes):
                    for b in nodes[i + 1:]:
                        criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                        if best is None or criterion < best[0]:
                            best = (criterion, a, b)
                if best is None:
                    break
                _, a, b = best
                joined = f"({trees[a]},{trees[b]})"
                new = f"n{len(nodes)}_{a}_{b}"
                rest = [x for x in nodes if x not in (a, b)]
                for x in rest:
                    matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
                nodes = rest + [new]
                trees[new] = joined
            newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: Prepare sequence data for vectorized ML scoring
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seq_array = np.zeros((n_taxa, n_sites), dtype=np.int32)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(n_sites):
            seq_array[i, j] = state_map[seq[j]]
    
    # Estimate branch length
    total_d = 0.0
    count = 0
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            total_d += matrix.get((a, b), 0.0)
            count += 1
    mean_d = total_d / max(1, count)
    base_length = max(0.01, min(0.5, mean_d / max(1, n_taxa - 2)))
    
    # Step 5: SPR local search with per-branch length ML scoring
    if _check_deadline():
        # Compute distance matrix for branch length estimation
        dist_matrix = {}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    d = _jc_distance(sequences[a], sequences[b])
                    dist_matrix[(a, b)] = d
                    dist_matrix[(b, a)] = d
        
        # Use SPR search which is more powerful than NNI
        adj = _spr_local_search(adj, node_ids, node_types, seq_array, n_sites, dist_matrix, labels, max_iterations=20)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
