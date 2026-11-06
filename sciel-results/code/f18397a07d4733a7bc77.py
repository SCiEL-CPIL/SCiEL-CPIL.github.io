import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255

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
    """Build adjacency list and node types from tree structure."""
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
    """Root the tree arbitrarily at an internal node for DP calculation."""
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
    """Get postorder traversal for DP."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order

def _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, branch_lengths):
    """
    Compute ML log-likelihood.
    seq_array: (n_taxa, n_sites) int array 0-3.
    branch_lengths: dict mapping (u, v) or node->parent to length? 
    For unrooted trees with NNI, it's better to store lengths on edges.
    Let's store branch lengths in a dict keyed by frozenset({u, v}).
    """
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Map leaf labels to indices in seq_array
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    # Precompute transition probabilities for each branch length?
    # No, branch lengths vary. We compute on the fly.
    
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
            # Get branch length between nid and child
            # The edge is (nid, child)
            bl = branch_lengths.get(frozenset((nid, child)), 0.1) # Default 0.1
            
            # Compute transition matrix probabilities
            exp_term = math.exp(-4.0 * bl / 3.0)
            p_same = 0.25 + 0.75 * exp_term
            p_diff = (1.0 - exp_term) / 3.0
            
            cp = prob[child]  # (n_sites, 4)
            row_sum = cp.sum(axis=1, keepdims=True)  # (n_sites, 1)
            # child_contrib[s] = p_same * cp[s] + p_diff * (row_sum - cp[s])
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll

def _apply_nni(adj, u, v, a, b, c, d, move_type, branch_lengths):
    """
    Apply NNI move.
    u-v is the internal edge.
    u neighbors: v, a, b
    v neighbors: u, c, d
    
    Move 1: swap b and c.
    New edges: u-c, v-b.
    Removed edges: u-b, v-c.
    
    Move 2: swap b and d.
    New edges: u-d, v-b.
    Removed edges: u-b, v-d.
    """
    new_adj = {k: set(v2) for k, v2 in adj.items()}
    new_bl = branch_lengths.copy()
    
    # Remove old edges
    old_edge_1 = frozenset((u, b))
    old_edge_2 = frozenset((v, c) if move_type == 1 else (v, d))
    
    if old_edge_1 in new_bl:
        del new_bl[old_edge_1]
    if old_edge_2 in new_bl:
        del new_bl[old_edge_2]
    
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
        
        # Assign branch lengths: keep average or specific logic?
        # For simplicity, assign the length of the removed edge to the new one?
        # Or 0.1?
        # Let's try to preserve total length.
        len_b = branch_lengths.get(old_edge_1, 0.1)
        len_c = branch_lengths.get(old_edge_2, 0.1)
        new_bl[frozenset((u, c))] = len_c
        new_bl[frozenset((v, b))] = len_b
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
        
        len_b = branch_lengths.get(old_edge_1, 0.1)
        len_d = branch_lengths.get(old_edge_2, 0.1)
        new_bl[frozenset((u, d))] = len_d
        new_bl[frozenset((v, b))] = len_b
        
    return new_adj, new_bl

def _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, branch_lengths, max_iterations=20):
    """
    Perform NNI local search.
    """
    current_ll = _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, branch_lengths)
    
    improved = True
    iteration = 0
    
    # Pre-calculate internal edges
    def get_internal_edges(adj, node_types):
        edges = []
        seen = set()
        for u in node_ids:
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u: continue
                if node_types[v][0] != 'internal':
                    continue
                edge = frozenset((u, v))
                if edge not in seen:
                    seen.add(edge)
                    edges.append((u, v))
        return edges

    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_ll = current_ll
        best_adj = None
        best_bl = None
        
        edges = get_internal_edges(adj, node_types)
        
        for u, v in edges:
            u_nbrs = [x for x in adj[u] if x != v]
            v_nbrs = [x for x in adj[v] if x != u]
            
            if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                continue
            
            a = u_nbrs[0]
            b = u_nbrs[1]
            c = v_nbrs[0]
            d = v_nbrs[1]
            
            # Try NNI1
            new_adj1, new_bl1 = _apply_nni(adj, u, v, a, b, c, d, 1, branch_lengths)
            ll1 = _ml_score_tree_fast(new_adj1, node_ids, node_types, seq_array, n_sites, new_bl1)
            if ll1 > best_ll + 1e-6:
                best_ll = ll1
                best_adj = new_adj1
                best_bl = new_bl1
                
            # Try NNI2
            new_adj2, new_bl2 = _apply_nni(adj, u, v, a, b, c, d, 2, branch_lengths)
            ll2 = _ml_score_tree_fast(new_adj2, node_ids, node_types, seq_array, n_sites, new_bl2)
            if ll2 > best_ll + 1e-6:
                best_ll = ll2
                best_adj = new_adj2
                best_bl = new_bl2
                
        if best_adj is not None:
            adj = best_adj
            branch_lengths = best_bl
            current_ll = best_ll
            improved = True
    
    return adj, branch_lengths

def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    # Pick an arbitrary leaf as the "root" for Newick generation
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in adj[nid]:
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # 1. Compute JC Distance Matrix
    matrix = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                d = _jc_distance(sequences[a], sequences[b])
                matrix[(a, b)] = d
                matrix[(b, a)] = d
                
    # 2. Neighbor Joining Initialization
    # Returns Newick string and branch lengths?
    # Standard NJ implementation
    nodes = list(labels)
    trees = {name: name for name in nodes}
    # Branch lengths storage: dict mapping node to length? 
    # For NJ, we calculate lengths when joining.
    # We'll store lengths in a dict: key = node_name, value = length from parent.
    # But the tree changes. Let's just get the Newick and parse branch lengths if any, 
    # or estimate them later.
    # Standard NJ often returns a tree with branch lengths.
    
    # Simple NJ without explicit branch length tracking for the final tree structure,
    # but we can estimate branch lengths from the distances.
    
    current_d = { (a, b): matrix[(a, b)] for a in nodes for b in nodes if a != b }
    # Make it a dict of dicts for easier access?
    D = {}
    for a in nodes:
        D[a] = {}
        for b in nodes:
            if a != b:
                D[a][b] = matrix[(a, b)]
    
    while len(nodes) > 3:
        n = len(nodes)
        # r(i) = sum_{k != i} D[i][k]
        r = {a: sum(D[a][k] for k in nodes if k != a) for a in nodes}
        
        best_c = None
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                c = (n - 2) * D[a][b] - r[a] - r[b]
                if best_c is None or c < best_c:
                    best_c = c
                    best_pair = (a, b)
        
        a, b = best_pair
        # Calculate branch lengths for a and b
        # delta_a = 0.5 * D[a][b] + (r[a] - r[b]) / (2 * (n - 2))
        delta_a = 0.5 * D[a][b] + (r[a] - r[b]) / (2.0 * (n - 2))
        delta_b = 0.5 * D[a][b] - (r[a] - r[b]) / (2.0 * (n - 2))
        
        # Clamp lengths to non-negative
        if delta_a < 0: delta_a = 0.0
        if delta_b < 0: delta_b = 0.0
        
        new_node = f"n_{a}_{b}"
        
        # Update distance matrix
        for x in nodes:
            if x not in (a, b):
                # D[new_node][x] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
                D[new_node][x] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
                D[x][new_node] = D[new_node][x]
        
        # Update tree
        # trees[new_node] = (trees[a], trees[b]) with lengths?
        # For now, just topology. We'll estimate branch lengths for ML later.
        trees[new_node] = f"({trees[a]}:{delta_a},{trees[b]}:{delta_b})"
        
        # Remove a, b, add new_node
        nodes = [x for x in nodes if x not in (a, b)] + [new_node]
        
    # Final tree
    # Reconstruct the full Newick with lengths
    # We need to assemble the tree from the `trees` dict.
    # The root is the last node added or the one that contains all leaves.
    # Actually, `trees` maps node names to strings.
    # The final node in `nodes` is the root.
    final_node = nodes[0]
    final_newick = trees[final_node] + ";"
    
    # Parse the Newick to get the structure and branch lengths
    tree = _parse_newick(final_newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Extract branch lengths from the Newick string
    # We need to associate lengths with edges in the adjacency list.
    # Since _build_adjacency doesn't preserve lengths, we need a modified parser or
    # we can just re-estimate branch lengths using a heuristic or optimize them.
    
    # For simplicity and robustness, let's just use a single global branch length estimate
    # or optimize branch lengths for the initial tree.
    
    # Estimate global branch length
    total_d = sum(matrix[(a, b)] for i, a in enumerate(labels) for b in labels[i+1:] if (a,b) in matrix)
    count = len(labels) * (len(labels) - 1) / 2.0
    mean_d = total_d / count if count > 0 else 0.1
    # Heuristic: mean distance ~ 2 * (N-2) * avg_bl?
    # In a star, d = 2*bl.
    # In a balanced tree, it's complex.
    # Let's just use mean_d / (N-1) as a rough scale.
    global_bl = max(0.01, mean_d / max(1, n_taxa - 1))
    
    # Initialize branch lengths dict for all edges
    branch_lengths = {}
    for u in node_ids:
        for v in adj[u]:
            if u < v: # Avoid duplicates
                branch_lengths[frozenset((u, v))] = global_bl
                
    # Prepare sequence data
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seq_array = np.zeros((n_taxa, n_sites), dtype=np.int32)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(n_sites):
            seq_array[i, j] = state_map[seq[j]]
            
    # NNI Search
    if _check_deadline():
        adj, branch_lengths = _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, branch_lengths, max_iterations=50)
        
    # Convert to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate
    for label in labels:
        if label not in result:
            return final_newick
            
    return result
