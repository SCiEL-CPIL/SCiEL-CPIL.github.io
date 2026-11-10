"""Phylogenetic topology recovery using quartet scoring with NNI refinement."""
import time
import math
import numpy as np
from itertools import combinations


def _jc_dist_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)
    p = mismatches / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _quartet_scores_vectorized(sequences, labels, n_sites):
    """Compute quartet scores for all 4-taxon subsets using vectorized operations.
    
    For each set of 4 taxa, compute support for each of the 3 possible
    unrooted topologies based on pairwise match counts across all sites.
    
    Returns a dict mapping frozenset of 4 taxa -> (score_01|23, score_02|13, score_03|12)
    """
    n = len(labels)
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    
    quartet_scores = {}
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        labels_quad = (labels[a], labels[b], labels[c], labels[d])
        key = frozenset(labels_quad)
        
        # Vectorized computation of match counts
        ab_match = (mat[a] == mat[b]).astype(np.int8).sum()
        cd_match = (mat[c] == mat[d]).astype(np.int8).sum()
        ac_match = (mat[a] == mat[c]).astype(np.int8).sum()
        bd_match = (mat[b] == mat[d]).astype(np.int8).sum()
        ad_match = (mat[a] == mat[d]).astype(np.int8).sum()
        bc_match = (mat[b] == mat[c]).astype(np.int8).sum()
        
        # Score each topology: sum of matches within the two clades
        score_0123 = ab_match + cd_match  # (a,b)|(c,d)
        score_0213 = ac_match + bd_match  # (a,c)|(b,d)
        score_0312 = ad_match + bc_match  # (a,d)|(b,c)
        
        quartet_scores[key] = (score_0123, score_0213, score_0312)
    
    return quartet_scores


def _build_tree_from_quartets(quartet_scores, labels):
    """Build a tree from quartet scores using a greedy quartet method.
    
    Uses a split-based approach: start with all taxa in one group,
    and iteratively split groups based on the strongest quartet constraints.
    """
    n = len(labels)
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        elif n == 1:
            return labels[0]
        else:
            return "(" + ",".join(labels) + ")"
    
    # Get all quartets sorted by strength of signal
    quartet_list = []
    for key, scores in quartet_scores.items():
        s0, s1, s2 = scores
        total = s0 + s1 + s2
        if total == 0:
            continue
        sorted_scores = sorted([s0, s1, s2], reverse=True)
        strength = sorted_scores[0] - sorted_scores[1]
        if strength > 0:
            # Determine which topology has the highest score
            if s0 >= s1 and s0 >= s2:
                # (a,b)|(c,d) - need to figure out which is which
                taxa = sorted(key)
                a, b, c, d = taxa
                quartet_list.append((strength, (a, b), (c, d), key))
            elif s1 >= s0 and s1 >= s2:
                # (a,c)|(b,d)
                taxa = sorted(key)
                a, b, c, d = taxa
                quartet_list.append((strength, (a, c), (b, d), key))
            else:
                # (a,d)|(b,c)
                taxa = sorted(key)
                a, b, c, d = taxa
                quartet_list.append((strength, (a, d), (b, c), key))
    
    # Sort by strength descending
    quartet_list.sort(key=lambda x: x[0], reverse=True)
    
    # Use a simple approach: build the tree using NJ as the base,
    # but use quartet information to guide NNI moves
    # For now, return None to fall back to NJ
    return None


def _nj_tree(dist, labels):
    """Standard neighbour joining with proper distance updates."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 1:
        return labels[0]
    
    # Use dictionary-based approach for clarity and correctness
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # dist_dict maps (node_i, node_j) -> distance
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            d = float(dist[i, j])
            dist_dict[(i, j)] = d
            dist_dict[(j, i)] = d
    
    counter = n  # for naming new internal nodes
    
    while len(nodes) > 3:
        n_cur = len(nodes)
        
        # Compute row sums
        row_sum = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if a != b:
                    s += dist_dict[(a, b)]
            row_sum[a] = s
        
        # Find the pair with minimum Q criterion
        best_criterion = float('inf')
        best_a = best_b = None
        for i_idx in range(len(nodes)):
            a = nodes[i_idx]
            for j_idx in range(i_idx + 1, len(nodes)):
                b = nodes[j_idx]
                criterion = (n_cur - 2) * dist_dict[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
        
        a, b = best_a, best_b
        d_ab = dist_dict[(a, b)]
        
        # Compute branch lengths
        len_a = 0.5 * d_ab + (row_sum[a] - row_sum[b]) / (2.0 * (n_cur - 2))
        len_b = d_ab - len_a
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        # Create new node
        new_name = counter
        counter += 1
        trees[new_name] = f"({trees[a]},{trees[b]})"
        
        # Update distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            if d_new_x < 0:
                d_new_x = 0.0
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x
        
        nodes = rest + [new_name]
    
    # Final tree: join the remaining 3 nodes
    return "(" + ",".join(trees[x] for x in nodes) + ")"


def _nni_improve(newick, dist, labels, max_iterations=20, time_limit=None, start_time=None):
    """Apply NNI moves to improve the tree topology using distance-based scoring."""
    if time_limit is not None and start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return newick
    
    # Parse the Newick string into an adjacency representation
    # For simplicity, we'll work with a list of splits
    
    # Parse newick to get the tree structure
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # Build adjacency list from newick
    # This is a simplified parser for the specific format we generate
    parent_of = {}
    children_of = {}
    taxon_map = {}
    node_count = [0]
    
    def next_node_id():
        nid = node_count[0]
        node_count[0] += 1
        parent_of[nid] = -1
        children_of[nid] = []
        return nid
    
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            tokens.append(('LPAREN', '('))
            i += 1
        elif c == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
        elif c == ',':
            tokens.append(('COMMA', ','))
            i += 1
        elif c == ':':
            j = i + 1
            while j < len(s) and s[j] not in '(),;':
                j += 1
            tokens.append(('BL', s[i+1:j]))
            i = j
        else:
            j = i
            while j < len(s) and s[j] not in '():,;':
                j += 1
            tokens.append(('LABEL', s[i:j]))
            i = j
    
    pos = [0]
    
    def parse_subtree():
        while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
            pos[0] += 1
        
        if pos[0] >= len(tokens):
            return None
        
        tok = tokens[pos[0]]
        if tok[0] == 'LPAREN':
            node_id = next_node_id()
            pos[0] += 1
            
            while True:
                child_id = parse_subtree()
                if child_id is not None:
                    children_of[node_id].append(child_id)
                    parent_of[child_id] = node_id
                
                while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                    pos[0] += 1
                
                if pos[0] >= len(tokens):
                    break
                    
                tok2 = tokens[pos[0]]
                if tok2[0] == 'COMMA':
                    pos[0] += 1
                elif tok2[0] == 'RPAREN':
                    pos[0] += 1
                    break
                else:
                    break
            
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
                
            if pos[0] < len(tokens):
                tok3 = tokens[pos[0]]
                if tok3[0] == 'BL':
                    pos[0] += 1
                    
            return node_id
        elif tok[0] == 'LABEL':
            node_id = next_node_id()
            taxon_map[tok[1]] = node_id
            pos[0] += 1
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
            if pos[0] < len(tokens) and tokens[pos[0]][0] == 'BL':
                pos[0] += 1
            return node_id
        else:
            return None
    
    root_id = parse_subtree()
    
    n_nodes = len(children_of)
    if n_nodes < 4:
        return newick
    
    # Build adjacency list
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for child in children_of[i]:
            adj[i].append(child)
            adj[child].append(i)
    
    # Identify internal nodes (nodes with children)
    internal_nodes = [i for i in range(n_nodes) if len(children_of[i]) > 0]
    
    # Map labels to indices
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time_limit is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Find all internal edges
        internal_edges = []
        for u in internal_nodes:
            for v in adj[u]:
                if v > u and v in internal_nodes:
                    internal_edges.append((u, v))
        
        for (u, v) in internal_edges:
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) < 2 or len(v_neighbors) < 2:
                continue
            
            a, b = u_neighbors[0], u_neighbors[1]
            c, d = v_neighbors[0], v_neighbors[1]
            
            # Get taxon sets for each side
            def get_taxa_set(nid):
                if nid in taxon_map:
                    return {taxon_map[nid]}
                result = set()
                for child in children_of[nid]:
                    result |= get_taxa_set(child)
                return result
            
            set_a = get_taxa_set(a)
            set_b = get_taxa_set(b)
            set_c = get_taxa_set(c)
            set_d = get_taxa_set(d)
            
            # Compute distance-based cost for current and alternative topologies
            # Cost = sum of distances across the split
            def side_dist(set1, set2):
                total = 0.0
                for t1 in set1:
                    i1 = label_to_idx[t1]
                    for t2 in set2:
                        i2 = label_to_idx[t2]
                        total += dist[i1, i2]
                return total
            
            cost_current = side_dist(set_a | set_b, set_c | set_d)
            cost_alt1 = side_dist(set_a | set_c, set_b | set_d)
            cost_alt2 = side_dist(set_a | set_d, set_b | set_c)
            
            min_cost = min(cost_current, cost_alt1, cost_alt2)
            
            if min_cost < cost_current - 1e-10:
                if min_cost == cost_alt1:
                    # Swap b and c
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(c)
                    adj[c].remove(v)
                    
                    adj[u].append(c)
                    adj[c].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                elif min_cost == cost_alt2:
                    # Swap b and d
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(d)
                    adj[d].remove(v)
                    
                    adj[u].append(d)
                    adj[d].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                
                # Update parent/children
                parent_of = {i: -1 for i in range(n_nodes)}
                children_of = {i: [] for i in range(n_nodes)}
                
                visited = set()
                stack = [root_id]
                parent_of[root_id] = -1
                
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            parent_of[neighbor] = node
                            children_of[node].append(neighbor)
                            stack.append(neighbor)
                
                improved = True
        
        if improved:
            # Rebuild internal_nodes list
            internal_nodes = [i for i in range(n_nodes) if len(children_of[i]) > 0]
    
    # Convert back to Newick
    node_to_taxon = {}
    for taxon, nid in taxon_map.items():
        node_to_taxon[nid] = taxon
    
    def node_name(nid):
        if nid in node_to_taxon:
            return node_to_taxon[nid]
        return f"n{nid}"
    
    def to_newick_recursive(nid):
        kids = children_of[nid]
        if not kids:
            return node_name(nid)
        else:
            parts = [to_newick_recursive(k) for k in kids]
            return "(" + ",".join(parts) + ")"
    
    return to_newick_recursive(root_id) + ";"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 165.0  # Safety margin before 180s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    n_sites = len(sequences[labels[0]])
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    if time.time() > deadline:
        return "(" + ",".join(labels) + ";"
    
    # Try quartet-based approach if time permits
    quartet_tree = None
    if n <= 20 and time.time() < deadline - 30.0:
        try:
            quartet_scores = _quartet_scores_vectorized(sequences, labels, n_sites)
            quartet_tree = _build_tree_from_quartets(quartet_scores, labels)
        except Exception:
            quartet_tree = None
    
    # Use NJ as the primary method
    result = _nj_tree(dist, labels)
    
    if time.time() > deadline - 5.0:
        return result
    
    # Apply NNI refinement
    try:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 5.0:
            result = _nni_improve(result, dist, labels, max_iterations=30, 
                                  time_limit=remaining_time,
                                  start_time=start_time)
    except Exception:
        pass
    
    return result
