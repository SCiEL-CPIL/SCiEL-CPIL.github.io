"""Neighbour-joining with NNI refinement, vectorized JC distances, and a deadline guard."""
import time
import math
import numpy as np


def _parse_newick(newick):
    """Parse a Newick string into a tree structure (list of nodes).
    Returns (nodes, edges, taxon_map) where:
      nodes: list of node indices (0..n_nodes-1)
      edges: list of (parent, child) tuples
      taxon_map: dict taxon_name -> node_index
    Also returns a parent dict: parent[i] = parent node index or -1 for root.
    And a children list: children[i] = list of child node indices.
    """
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # Tokenize
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
        elif c == ':' :
            # Branch length
            j = i + 1
            while j < len(s) and s[j] not in '(),;':
                j += 1
            tokens.append(('BL', s[i+1:j]))
            i = j
        else:
            # Label
            j = i
            while j < len(s) and s[j] not in '():,;':
                j += 1
            tokens.append(('LABEL', s[i:j]))
            i = j
    
    # Parse recursively
    nodes = []  # list of node ids
    parent_of = {}  # node_id -> parent_node_id
    children_of = {}  # node_id -> [child_node_ids]
    taxon_map = {}  # taxon_name -> node_id
    node_count = [0]
    
    def next_node_id():
        nid = node_count[0]
        node_count[0] += 1
        parent_of[nid] = -1
        children_of[nid] = []
        return nid
    
    pos = [0]
    
    def parse_subtree():
        """Parse a subtree starting at current position. Returns node_id."""
        # Skip whitespace
        while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
            pos[0] += 1
        
        if pos[0] >= len(tokens):
            return None
        
        tok = tokens[pos[0]]
        if tok[0] == 'LPAREN':
            # Internal node
            node_id = next_node_id()
            pos[0] += 1  # consume '('
            
            # Parse children
            while True:
                child_id = parse_subtree()
                if child_id is not None:
                    children_of[node_id].append(child_id)
                    parent_of[child_id] = node_id
                
                # Skip whitespace
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
            
            # After ')', there might be a label or branch length
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
                
            if pos[0] < len(tokens):
                tok3 = tokens[pos[0]]
                if tok3[0] == 'LABEL':
                    # This shouldn't happen for internal nodes in standard Newick
                    # But some formats allow it
                    pass
                elif tok3[0] == 'BL':
                    pos[0] += 1
                    
            return node_id
        elif tok[0] == 'LABEL':
            # Leaf node
            node_id = next_node_id()
            taxon_map[tok[1]] = node_id
            pos[0] += 1
            # Skip branch length if present
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
            if pos[0] < len(tokens) and tokens[pos[0]][0] == 'BL':
                pos[0] += 1
            return node_id
        else:
            return None
    
    root_id = parse_subtree()
    
    return nodes, parent_of, children_of, taxon_map, root_id


def _to_newick(parent_of, children_of, taxon_map, root_id, labels):
    """Convert tree structure back to Newick string."""
    # Build reverse map: node_id -> taxon_name for leaves
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


def _get_splits(parent_of, children_of, root_id, taxon_map, labels):
    """Get all non-trivial splits from an unrooted tree.
    For each edge, the split is the set of taxa on one side vs the other.
    Returns a set of frozensets (each split represented as frozenset of taxa on one side, 
    with the convention that the smaller side is used, or we use a canonical representation).
    """
    n_taxa = len(labels)
    label_set = set(labels)
    
    # For each edge (parent, child), compute the set of taxa in the child's subtree
    splits = set()
    
    def get_subtree_taxa(nid, visited):
        """Get all taxa in the subtree rooted at nid (in the rooted sense)."""
        if nid in visited:
            return set()
        visited.add(nid)
        taxa = set()
        if nid in taxon_map:
            taxa.add(taxon_map[nid])
        for child in children_of[nid]:
            taxa |= get_subtree_taxa(child, visited)
        return taxa
    
    # For each edge, compute the split
    for nid in range(len(children_of)):
        if parent_of[nid] == -1:
            continue  # skip root
        # The split is: taxa in subtree of nid vs taxa not in subtree of nid
        # We need to be careful about the rooting
        pass
    
    # Alternative approach: for each edge, compute the two sides
    # Since the tree is unrooted, we can root it arbitrarily and for each edge,
    # the two sides are the subtree below the edge and the complement
    
    # Let's use a different approach: for each node (except root), 
    # the edge to its parent defines a split
    # The subtree below that edge contains all taxa in that node's subtree
    
    # But we need to handle the unrooted nature properly.
    # For an unrooted tree with n taxa, there are n-3 internal edges.
    # Each internal edge defines a split.
    
    # Let me use a cleaner approach:
    # Root the tree at root_id
    # For each edge (u, v) where u is parent of v:
    #   The split is: {taxa in subtree of v} vs {taxa not in subtree of v}
    # We represent each split as a frozenset of the smaller side (or canonical form)
    
    def compute_subtree_taxa(nid):
        """Compute set of taxa in subtree rooted at nid."""
        if nid in taxon_map:
            return {taxon_map[nid]}
        result = set()
        for child in children_of[nid]:
            result |= compute_subtree_taxa(child)
        return result
    
    all_taxa = set(labels)
    
    for nid in range(len(children_of)):
        if parent_of[nid] == -1:
            continue
        subtree_taxa = compute_subtree_taxa(nid)
        complement = all_taxa - subtree_taxa
        
        # Skip trivial splits (size 1 or n-1)
        if len(subtree_taxa) <= 1 or len(subtree_taxa) >= n_taxa - 1:
            continue
        
        # Canonical representation: use the smaller side, or if equal, use lexicographic
        if len(subtree_taxa) < len(complement):
            split_key = frozenset(subtree_taxa)
        else:
            split_key = frozenset(complement)
        
        splits.add(split_key)
    
    return splits


def _nni_improve(newick, dist, labels, max_iterations=10, time_limit=None, start_time=None):
    """Apply NNI moves to improve the tree topology.
    
    For each internal edge, consider the two alternative topologies
    and keep the one with smaller sum of pairwise distances (as a proxy).
    
    Returns the improved Newick string.
    """
    if time_limit is not None and start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return newick
    
    # Parse the tree
    nodes, parent_of, children_of, taxon_map, root_id = _parse_newick(newick)
    n_nodes = len(children_of)
    
    if n_nodes < 4:
        return newick
    
    # We'll work with the tree structure
    # For NNI, we need to find internal edges and consider swaps
    
    # Let's use a simpler approach: represent the tree as an adjacency list
    # and perform NNI on the unrooted tree
    
    # Build adjacency list (undirected)
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for child in children_of[i]:
            adj[i].append(child)
            adj[child].append(i)
    
    # Identify internal nodes (non-leaf)
    internal_nodes = [i for i in range(n_nodes) if len(children_of[i]) > 0]
    
    # For each internal edge, try NNI
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
            # For edge (u, v), the four subtrees are:
            # - The part of u's side excluding v
            # - The part of v's side excluding u
            # - The other two neighbors of u
            # - The other two neighbors of v
            
            # Get neighbors of u (excluding v)
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) < 2 or len(v_neighbors) < 2:
                continue  # Not a valid internal edge for NNI
            
            # The current topology is: (A, B) connected to u, (C, D) connected to v
            # where A, B are u_neighbors and C, D are v_neighbors
            # Actually, u has degree 3 (for internal edge in binary tree), so u_neighbors has 2 elements
            # Same for v
            
            a, b = u_neighbors[0], u_neighbors[1]
            c, d = v_neighbors[0], v_neighbors[1]
            
            # Current: ... - u - ... with u connected to a, b, v
            #         ... - v - ... with v connected to c, d, u
            # NNI alternatives:
            # Alt 1: u connected to a, c, v; v connected to b, d, u
            # Alt 2: u connected to a, d, v; v connected to b, c, u
            
            # To evaluate, we can use the sum of distances criterion:
            # For the current topology, the "cost" is related to the distances
            # between the four subtrees.
            
            # A simple heuristic: use the sum of distances between taxa in different subtrees
            # But that's expensive. Instead, use a simpler proxy:
            # The NNI that minimizes the sum of pairwise distances across the split
            
            # Let's compute the "length" of each subtree as the sum of distances from the root of that subtree to all taxa in it
            # Actually, let's use a simpler approach: just count the number of taxa in each subtree
            # and use the distance matrix to estimate the cost
            
            # For now, let's use a simpler criterion: 
            # Compute the sum of distances for the three possible pairings of the four subtrees
            # Current: (a,b) and (c,d) are paired through the edge u-v
            # Alt 1: (a,c) and (b,d)
            # Alt 2: (a,d) and (b,c)
            
            # The cost for a pairing (X,Y) and (Z,W) is roughly:
            # dist(X, Z) + dist(X, W) + dist(Y, Z) + dist(Y, W)
            # where dist(X, Z) is the average distance between taxa in subtree X and subtree Z
            
            # But computing this is expensive. Let's use a simpler proxy:
            # For each subtree, compute the "center" (mean position) and use that
            
            # Actually, let's just use the number of taxa as a simple weight
            # and the average distance between the subtree roots
            
            # For simplicity, let's compute the average distance between the four subtree roots
            # for each of the three pairings
            
            # Get the representative node for each subtree (the neighbor itself)
            # a, b, c, d are the subtree roots
            
            # Compute average distances
            def avg_dist(nid1, nid2):
                """Average distance between taxa in subtree nid1 and subtree nid2."""
                # This is expensive; use a simpler proxy for now
                # Just return the distance between the root nodes as a proxy
                # But we don't have a direct distance between internal nodes
                # Let's use the number of taxa as a simple weight
                return 0.0  # Placeholder
            
            # For now, let's use a simpler approach: 
            # Just try the NNI and keep it if it doesn't increase the RF distance
            # But we don't have the true tree to compare against
            
            # Alternative: use the sum of pairwise distances as a cost function
            # For the current topology, the cost is:
            # sum of all pairwise distances weighted by the split structure
            
            # Let's implement a proper cost function:
            # For a given tree topology, the cost is the sum over all pairs of taxa
            # of their distance in the tree (which is the sum of branch lengths on the path)
            # But we don't have branch lengths, so let's use the distance matrix as a proxy
            
            # Actually, for NNI, a common approach is to use the "quartet score":
            # For the four taxa a, b, c, d (representing the four subtrees),
            # the three possible quartet topologies are:
            # 1. (a,b)|(c,d) - current
            # 2. (a,c)|(b,d) - alt 1
            # 3. (a,d)|(b,c) - alt 2
            # The quartet score for a topology is the sum of distances for the split
            # For (a,b)|(c,d): dist(a,c) + dist(a,d) + dist(b,c) + dist(b,d)
            # For (a,c)|(b,d): dist(a,b) + dist(a,d) + dist(c,b) + dist(c,d)
            # For (a,d)|(b,c): dist(a,b) + dist(a,c) + dist(d,b) + dist(d,c)
            
            # But a, b, c, d are subtrees, not single taxa.
            # Let's use the average distance between the subtrees as a proxy
            
            # Compute average distance between each pair of subtrees
            def subtree_avg_dist(nid1, nid2):
                """Average distance between taxa in subtree nid1 and subtree nid2."""
                # Get all taxa in each subtree
                def get_taxa(nid):
                    if nid in taxon_map:
                        return {taxon_map[nid]}
                    result = set()
                    for child in children_of[nid]:
                        result |= get_taxa(child)
                    return result
                
                taxa1 = get_taxa(nid1)
                taxa2 = get_taxa(nid2)
                
                if not taxa1 or not taxa2:
                    return 0.0
                
                total = 0.0
                count = 0
                for t1 in taxa1:
                    for t2 in taxa2:
                        # Find indices in labels
                        i1 = labels.index(t1)
                        i2 = labels.index(t2)
                        total += dist[i1, i2]
                        count += 1
                return total / count if count > 0 else 0.0
            
            # This is too expensive for large trees. Let's use a simpler approach.
            # For now, let's just use the distance between the root nodes as a proxy
            # and accept that this is a heuristic
            
            # Actually, let's implement a proper but efficient NNI using the 
            # "length" of each subtree as the sum of distances from the root to all leaves
            
            def subtree_length(nid):
                """Sum of distances from nid to all leaves in its subtree."""
                if nid in taxon_map:
                    return 0.0
                total = 0.0
                for child in children_of[nid]:
                    total += subtree_length(child)
                return total
            
            # This is still not quite right. Let's use a different approach.
            
            # For simplicity, let's just implement NNI with a simple cost function
            # and see if it helps. If it's too slow, we'll skip it.
            
            # Cost for current topology:
            # The four subtrees are a, b, c, d
            # Current: a and b are on u's side, c and d are on v's side
            # The "cost" is the sum of distances between the two sides
            cost_current = 0.0
            cost_alt1 = 0.0
            cost_alt2 = 0.0
            
            # For efficiency, let's precompute the taxa sets for each subtree
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
            
            # Current: (a,b) vs (c,d)
            # Alt 1: (a,c) vs (b,d)
            # Alt 2: (a,d) vs (b,c)
            
            # Cost is the sum of distances between the two sides
            def side_dist(set1, set2):
                total = 0.0
                for t1 in set1:
                    i1 = labels.index(t1)
                    for t2 in set2:
                        i2 = labels.index(t2)
                        total += dist[i1, i2]
                return total
            
            cost_current = side_dist(set_a | set_b, set_c | set_d)
            cost_alt1 = side_dist(set_a | set_c, set_b | set_d)
            cost_alt2 = side_dist(set_a | set_d, set_b | set_c)
            
            # Choose the minimum cost
            min_cost = min(cost_current, cost_alt1, cost_alt2)
            
            if min_cost < cost_current - 1e-10:
                # Perform the NNI
                if min_cost == cost_alt1:
                    # Swap: u connected to a, c, v; v connected to b, d, u
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
                    # Swap: u connected to a, d, v; v connected to b, c, u
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
                
                improved = True
        
        # Rebuild the tree structure from the adjacency list
        # Root the tree at root_id
        if improved:
            # Rebuild parent_of and children_of from adj
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
            
            # Rebuild taxon_map (should be unchanged)
            # taxon_map is already correct
    
    # Convert back to Newick
    return _to_newick(parent_of, children_of, taxon_map, root_id, labels)


def _jc_dist_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    # Build a matrix of nucleotide codes: 0=A,1=C,2=G,3=T
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    # Pairwise mismatches
    # For each pair (i,j), count positions where mat[i]!=mat[j]
    # Use broadcasting: mat[:, None, :] vs mat[None, :, :] -> (n,n,n_sites)
    # That's n^2 * n_sites which for n=32, n_sites=2000 is 32^2*2000 = 2,048,000 -> fine
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)  # (n, n)
    p = mismatches / n_sites
    # JC correction: -3/4 ln(1 - 4p/3)
    np.fill_diagonal(p, 0.0)
    # Clip p to avoid log(0) or negative
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """Standard neighbour joining. dist is (n,n) numpy array, labels is list of str."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 1:
        return labels[0]

    # trees[i] = newick string for current node i
    trees = list(labels)
    # active nodes
    active = list(range(n))
    # We'll maintain a (n,n) distance matrix; when we merge, we add a new node
    # For simplicity, use a dict-based approach or extend the array.
    # Use a list of dicts? No, let's use a growing array.
    # Actually, let's use a simpler approach: maintain dist as a 2D list of lists, 
    # and a list of active node indices. When merging a,b, create new node index = len(active)+... 
    # Let's use a different strategy: keep dist as numpy array of size (n+n-3, n+n-3) but that's wasteful.
    # Better: use a list-based adjacency.
    
    # Let's use a clean implementation:
    # nodes: list of current node names (strings)
    # trees: dict name -> newick
    # dist: dict (name1, name2) -> distance
    # This is O(n^2) per iteration, total O(n^3) which is fine for n<=32.
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    # Build dist dict from numpy array
    dist_dict = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            d = float(dist[i, j])
            dist_dict[(nodes[i], nodes[j])] = d
            dist_dict[(nodes[j], nodes[i])] = d
    
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
        
        # Find best pair
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
        # Compute branch lengths
        d_ab = dist_dict[(a, b)]
        len_a = 0.5 * d_ab + (row_sum[a] - row_sum[b]) / (2.0 * (n_cur - 2))
        len_b = d_ab - len_a
        # Clamp to non-negative
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        # Create new node
        new_name = f"n{len(nodes)}_{a}_{b}"
        trees[new_name] = f"({trees[a]},{trees[b]})"
        
        # Update distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x
        
        nodes = rest + [new_name]
    
    # Final three nodes
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 20 instances in 270s wall-clock
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    result = _nj_tree(dist, labels)
    
    # Check deadline before NNI
    if time.time() > deadline - 5.0:
        return result
    
    # Apply NNI refinement
    try:
        result = _nni_improve(result, dist, labels, max_iterations=5, 
                              time_limit=deadline - time.time() - 2.0,
                              start_time=start_time)
    except Exception:
        # If NNI fails, return the NJ tree
        pass
    
    return result
