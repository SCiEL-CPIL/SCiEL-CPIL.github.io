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
    
    # For vectorized computation, we process all sites at once
    # prob[node] will be a (n_sites, 4) array
    
    # Initialize: for leaves, prob is one-hot at the observed state
    # For internal nodes, we'll compute in postorder
    
    # We need to map each node to its position in order for efficient access
    # prob[node_id] = (n_sites, 4) array
    
    prob = {}
    
    # Initialize leaves
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            # seq_array[idx] is (n_sites,) with values 0-3
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
            # Shouldn't happen for internal nodes, but handle gracefully
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        # For each state s at this node, compute:
        # P(s) = prod over children c of [sum over c_state of P_trans(s->c_state) * prob[c][c_state]]
        
        # Build the combined probability
        # For each child, compute the sum over c_states
        # result[s] = prod_c [ p_same * prob[c][s] + p_diff * sum_{c'!=s} prob[c][c'] ]
        
        # Efficient: for each child, compute:
        #   child_sum[s] = p_same * prob[c][s] + p_diff * (total_prob[c] - prob[c][s])
        # where total_prob[c] = sum over all 4 states of prob[c]
        
        # Initialize result as ones (n_sites, 4)
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            cp = prob[child]  # (n_sites, 4)
            # total = sum over states = row sum
            row_sum = cp.sum(axis=1, keepdims=True)  # (n_sites, 1)
            # child_sum[s] = p_same * cp[s] + p_diff * (row_sum - cp[s])
            #              = p_diff * row_sum + (p_same - p_diff) * cp[s]
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp  # (n_sites, 4)
            result *= child_contrib
        
        prob[nid] = result
    
    # Root probability: average over all states, then average over sites (or sum for total LL)
    root_prob = prob[root].mean(axis=1)  # (n_sites,) - average over 4 states per site
    
    # Total log-likelihood = sum of log(root_prob[site]) for each site
    # Clip to avoid log(0)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll


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


def _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, base_length, max_iterations=50):
    """Perform NNI local search to improve tree topology using ML scoring."""
    current_ll = _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, base_length)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        # Find all internal edges
        best_ll = current_ll
        best_adj = None
        
        for u in node_ids:
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u:  # avoid duplicates
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                # Get neighbors of u and v (excluding each other)
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a = u_nbrs[0]
                b = u_nbrs[1]
                c = v_nbrs[0]
                d = v_nbrs[1]
                
                # Try NNI1: swap b and c
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                ll1 = _ml_score_tree_fast(new_adj1, node_ids, node_types, seq_array, n_sites, base_length)
                if ll1 > best_ll + 1e-6:
                    best_ll = ll1
                    best_adj = new_adj1
                
                # Try NNI2: swap b and d
                new_adj2 = _apply_nni(adj, node_types, u, v, a, b, c, d, 2)
                ll2 = _ml_score_tree_fast(new_adj2, node_ids, node_types, seq_array, n_sites, base_length)
                if ll2 > best_ll + 1e-6:
                    best_ll = ll2
                    best_adj = new_adj2
        
        if best_adj is not None:
            adj = best_adj
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
    
    # Step 2: Neighbor-Joining to get initial tree
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
    
    # Build initial Newick
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
    
    # Step 5: NNI local search with vectorized ML scoring
    if _check_deadline():
        adj = _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, base_length, max_iterations=50)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
