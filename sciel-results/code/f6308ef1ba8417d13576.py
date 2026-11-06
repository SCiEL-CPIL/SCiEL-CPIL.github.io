import math
import time
import numpy as np

_DEADLINE = 260.0  # Safety margin before 270s kill
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    
    conv = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seqs = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        for j, c in enumerate(seq):
            num_seqs[i, j] = conv[c]
    
    L = len(sequences[labels[0]])
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(num_seqs[i] != num_seqs[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_build(D, labels):
    """Run Neighbor Joining, return Newick string."""
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    active = list(range(n))
    tree_strs = {i: labels[i] for i in range(n)}
    
    dist = {}
    for i in range(n):
        for j in range(i+1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
    
    counter = n
    while len(active) > 3:
        m = len(active)
        
        total = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    total[i] += dist[(i, j)]
        
        best_criterion = float('inf')
        best_pair = None
        
        for ki in range(m):
            i = active[ki]
            for kj in range(ki + 1, m):
                j = active[kj]
                criterion = (m - 2) * dist[(i, j)] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        i, j = best_pair
        joined_str = f"({tree_strs[i]},{tree_strs[j]})"
        new_idx = counter
        counter += 1
        
        for x in active:
            if x != i and x != j:
                d_new = 0.5 * (dist[(i, x)] + dist[(j, x)] - dist[(i, j)])
                dist[(new_idx, x)] = d_new
                dist[(x, new_idx)] = d_new
        
        active = [x for x in active if x not in (i, j)]
        active.append(new_idx)
        tree_strs[new_idx] = joined_str
    
    parts = [tree_strs[x] for x in active]
    return "(" + ",".join(parts) + ");"


def _parse_newick(newick):
    """
    Parse Newick string into an adjacency list representation.
    Returns:
        nodes: list of node IDs
        adj: dict mapping node ID to list of neighbor IDs
        leaves: dict mapping node ID to label (for leaves)
    """
    s = newick.rstrip(';').strip()
    
    nodes = []
    adj = {}
    leaves = {}
    node_counter = 0
    
    def parse(s):
        nonlocal node_counter
        s = s.strip()
        if not s:
            return None
        
        node_id = node_counter
        node_counter += 1
        nodes.append(node_id)
        adj[node_id] = []
        
        if s.startswith('('):
            depth = 0
            end_idx = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            inner = s[1:end_idx]
            parts = []
            depth = 0
            current = []
            for c in inner:
                if c == '(':
                    depth += 1
                    current.append(c)
                elif c == ')':
                    depth -= 1
                    current.append(c)
                elif c == ',' and depth == 0:
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(c)
            if current:
                parts.append(''.join(current))
            
            for part in parts:
                child_id = parse(part)
                if child_id is not None:
                    adj[node_id].append(child_id)
                    adj[child_id].append(node_id)
            
            rest = s[end_idx+1:].strip()
            # branch length ignored for topology
        else:
            label = s
            if ':' in s:
                label = s.split(':')[0].strip()
            leaves[node_id] = label
        
        return node_id

    parse(s)
    return nodes, adj, leaves


def _get_leaves_subtree(node, adj, leaves):
    """Get set of leaf labels in subtree rooted at node (away from parent)."""
    if node in leaves:
        return {leaves[node]}
    result = set()
    visited = set()
    stack = [node]
    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        if curr in leaves:
            result.add(leaves[curr])
        else:
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    stack.append(neighbor)
    return result


def _compute_splits(adj, leaves):
    """Compute all non-trivial splits from the tree."""
    n_leaves = len(leaves)
    if n_leaves <= 3:
        return set()
    
    splits = set()
    # For each internal edge, compute the split it induces
    for u in adj:
        if u in leaves:
            continue
        for v in adj[u]:
            if v in leaves:
                continue
            if u < v:  # avoid double counting
                # Get leaves on u side (excluding v side)
                # Do BFS from u not going through v
                visited = set()
                stack = [u]
                leaves_u = set()
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    if curr in leaves:
                        leaves_u.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor != v and neighbor not in visited:
                            stack.append(neighbor)
                
                if 0 < len(leaves_u) < n_leaves:
                    # Normalize split: always have smallest label first
                    if min(leaves_u) > min(set(leaves) - leaves_u):
                        splits.add(frozenset(set(leaves) - leaves_u))
                    else:
                        splits.add(frozenset(leaves_u))
    
    return splits


def _compute_nni_moves(adj, leaves):
    """
    Find all NNI moves. For each internal edge (u,v) where both u and v are internal,
    the NNI move swaps one of the four subtrees.
    Returns list of (u, v, child_to_move, new_parent) tuples.
    """
    moves = []
    for u in adj:
        if u in leaves:
            continue
        for v in adj[u]:
            if v in leaves:
                continue
            if u >= v:
                continue
            # u and v are internal nodes connected by an edge
            # u's neighbors: v and up to 2 others (in unrooted binary tree)
            # v's neighbors: u and up to 2 others
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Four NNI moves:
            # Current: u connects to {a, b, v}, v connects to {u, c, d}
            # Move 1: swap b and c -> u connects to {a, c, v'}, v' connects to {u', b, d}
            # Actually let me think more carefully.
            # The four subtrees hanging off the internal edge (u,v) are:
            # From u side: a, b (two subtrees)
            # From v side: c, d (two subtrees)
            # 
            # Current topology: (a,b) joined at u, (c,d) joined at v, u-v edge
            # NNI alternatives:
            # 1. (a,c) at u', (b,d) at v', u'-v' edge
            # 2. (a,d) at u'', (b,c) at v'', u''-v'' edge
            # 
            # So 3 possible topologies total, 2 NNI moves from current.
            
            # Move 1: pair (a,c) and (b,d)
            moves.append((u, v, a, c, b, d))
            # Move 2: pair (a,d) and (b,c)  
            moves.append((u, v, a, d, b, c))
    
    return moves


def _apply_nni(adj, u, v, a, c, b, d):
    """
    Apply NNI move: change from (a,b)@u - (c,d)@v to (a,c)@u' - (b,d)@v'
    This is done by modifying the adjacency list.
    Returns new adj dict (modified in place for efficiency, but we copy first).
    """
    # We need to modify adj in place
    # Remove edges: u-a, u-b, u-v, v-c, v-d
    # Add edges: u-c, u-a (keep), v-b, v-d (keep), u-v (keep)
    # Wait, let me reconsider.
    # Current: u connected to {a, b, v}, v connected to {u, c, d}
    # After NNI (a,c) and (b,d):
    #   u connected to {a, c, v}, v connected to {u, b, d}
    # So we swap: remove u-b and v-c, add u-c and v-b
    # But wait, that's only one of the two NNI moves.
    # The other is: remove u-b and v-d, add u-d and v-b -> (a,d) and (b,c)
    
    # For move (a,c) and (b,d):
    # Remove: u-b, v-c
    # Add: u-c, v-b
    
    adj[u].remove(b)
    adj[b].remove(u)
    adj[v].remove(c)
    adj[c].remove(v)
    
    adj[u].append(c)
    adj[c].append(u)
    adj[v].append(b)
    adj[b].append(v)
    
    return adj


def _nni_polish(newick, D, labels):
    """
    Perform NNI polish to improve topology.
    Uses a simple cost: for each NNI move, compute the change in the
    sum of squared residuals of the distance matrix fit.
    """
    nodes, adj, leaves = _parse_newick(newick)
    n = len(nodes)
    
    if n <= 3:
        return newick
    
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    node_to_label = {node: leaf for leaf, node in label_to_node.items()}
    
    # Precompute: for each leaf pair, which internal edges are on the path
    # This is expensive, so we'll use a simpler cost.
    # Cost = sum over all leaf pairs of |D[i][j] - path_length(i,j)|^2
    # But computing path lengths requires branch lengths.
    # 
    # Alternative simpler cost: For each internal edge, compute the "balance"
    # of the split it induces. A good tree has splits that are well-supported
    # by the distance matrix.
    #
    # Even simpler: use the NJ criterion. For each potential edge, compute
    # the NJ score contribution.
    #
    # Let's use a different approach: compute the QP cost.
    # For a given topology, fit branch lengths by least squares, then compute
    # the residual sum of squares.
    
    def fit_branch_lengths_and_cost(adj, D, label_to_node, node_to_label):
        """Fit branch lengths by least squares and return RSS."""
        # Build edge list
        edges = []
        edge_idx = {}
        for u in adj:
            for v in adj[u]:
                if u < v:
                    idx = len(edges)
                    edges.append((u, v))
                    edge_idx[(u, v)] = idx
                    edge_idx[(v, u)] = idx
        
        num_edges = len(edges)
        if num_edges == 0:
            return 0.0
        
        leaf_nodes = list(label_to_node.values())
        num_leaves = len(leaf_nodes)
        num_pairs = num_leaves * (num_leaves - 1) // 2
        
        if num_pairs == 0:
            return 0.0
        
        # Build path incidence matrix
        # A[i, k] = 1 if edge k is on path between leaf pair i
        # b[i] = D[leaf_i, leaf_j]
        
        # For efficiency, precompute paths using BFS from each leaf
        # paths_from_leaf[l] = {leaf_j: [list of edge indices on path]}
        
        # Actually, let's just compute RSS directly without building the full system.
        # Use the fact that for a tree, the distance between two leaves is the sum
        # of edge lengths on the path. We can set up the linear system.
        
        A = np.zeros((num_pairs, num_edges))
        b = np.zeros(num_pairs)
        
        # Precompute path from each leaf to all others
        # Use BFS
        leaf_to_idx = {node: i for i, node in enumerate(leaf_nodes)}
        
        for li, leaf_node in enumerate(leaf_nodes):
            # BFS from leaf_node
            visited = {leaf_node: []}
            queue = [leaf_node]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        # Path to neighbor = path to curr + edge (curr, neighbor)
                        curr_path = visited[curr]
                        e_key = (min(curr, neighbor), max(curr, neighbor))
                        if e_key in edge_idx:
                            new_path = curr_path + [edge_idx[e_key]]
                        else:
                            new_path = curr_path
                        visited[neighbor] = new_path
                        queue.append(neighbor)
            
            for lj in range(li + 1, num_leaves):
                leaf_j = leaf_nodes[lj]
                if leaf_j not in visited:
                    continue
                path = visited[leaf_j]
                pair_idx = li * (num_leaves - 1) - li * (li + 1) // 2 + (lj - li - 1)
                # Hmm, this indexing is getting complex. Let me use a simpler approach.
        
        # Let me redo with simpler indexing
        A = np.zeros((num_pairs, num_edges))
        b = np.zeros(num_pairs)
        
        pair_count = 0
        for li in range(num_leaves):
            leaf_node_i = leaf_nodes[li]
            # BFS from leaf_node_i
            dist_edges = {leaf_node_i: []}
            queue = [leaf_node_i]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in dist_edges:
                        curr_path = dist_edges[curr]
                        e_key = (min(curr, neighbor), max(curr, neighbor))
                        if e_key in edge_idx:
                            new_path = curr_path + [edge_idx[e_key]]
                        else:
                            new_path = curr_path
                        dist_edges[neighbor] = new_path
                        queue.append(neighbor)
            
            for lj in range(li + 1, num_leaves):
                leaf_node_j = leaf_nodes[lj]
                if leaf_node_j not in dist_edges:
                    continue
                path = dist_edges[leaf_node_j]
                for e_idx in path:
                    A[pair_count, e_idx] = 1.0
                
                # Get D value
                label_i = node_to_label[leaf_node_i]
                label_j = node_to_label[leaf_node_j]
                # Find indices in labels
                idx_i = labels.index(label_i)
                idx_j = labels.index(label_j)
                b[pair_count] = D[idx_i, idx_j]
                
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        A = A[:pair_count]
        b = b[:pair_count]
        
        # Least squares
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Clamp negative branch lengths to 0
            x = np.maximum(x, 0)
            predicted = A @ x
            rss = np.sum((b - predicted) ** 2)
            return rss
        except:
            return float('inf')
    
    # Compute initial cost
    initial_cost = fit_branch_lengths_and_cost(adj, D, label_to_node, node_to_label)
    
    # Try NNI moves
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if _time_left() < 5.0:
            break
        
        # Find all NNI moves
        moves = _compute_nni_moves(adj, leaves)
        
        for move in moves:
            u, v, a, c, b_node, d_node = move
            
            # Save state
            adj_copy = {k: list(v) for k, v in adj.items()}
            
            # Apply NNI move: swap b_node and c_node
            # Remove u-b_node and v-c_node, add u-c_node and v-b_node
            if b_node in adj[u] and c_node in adj[v]:
                adj[u].remove(b_node)
                adj[b_node].remove(u)
                adj[v].remove(c_node)
                adj[c_node].remove(v)
                
                adj[u].append(c_node)
                adj[c_node].append(u)
                adj[v].append(b_node)
                adj[b_node].append(v)
                
                # Compute new cost
                new_cost = fit_branch_lengths_and_cost(adj, D, label_to_node, node_to_label)
                
                if new_cost < initial_cost - 1e-10:
                    initial_cost = new_cost
                    improved = True
                    # Keep the move
                    break
                else:
                    # Revert
                    adj.clear()
                    adj.update(adj_copy)
        
        # If no improvement in this pass, done
        if not improved:
            break
    
    # Rebuild Newick from adj
    # Find a leaf to root the tree
    if not leaves:
        return newick
    
    root_leaf = next(iter(leaves))
    
    # Build Newick via DFS
    def build_newick(node, parent):
        nonlocal adj, leaves
        if node in leaves:
            return leaves[node]
        
        children = [x for x in adj[node] if x != parent]
        if not children:
            return leaves.get(node, f"n{node}")
        
        child_strs = []
        for child in children:
            child_strs.append(build_newick(child, node))
        
        return "(" + ",".join(child_strs) + ")"
    
    result = build_newick(root_leaf, -1)
    return f"({result});"


def _dpp_build(D, labels):
    """
    Distance-Proportional Parsimony (DPP) tree construction.
    
    DPP works by:
    1. Starting with a star tree (all taxa connected to a central node).
    2. Iteratively adding taxa by finding the position that minimizes the
       increase in total branch length.
    3. For each potential position, compute the implied branch lengths
       using the distance matrix.
    
    This is similar to NJ but uses a different criterion that may be
    more robust to distance estimation errors.
    """
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Start with the two closest taxa
    # Find the pair with minimum distance
    min_dist = float('inf')
    min_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < min_dist:
                min_dist = D[i, j]
                min_pair = (i, j)
    
    # Initialize tree with these two taxa
    # We'll maintain the tree as a set of nodes and edges
    # Each node has a label (for leaves) or is internal
    # Edges have lengths
    
    # Use a different representation: maintain the tree as adjacency with lengths
    # nodes: list of node IDs
    # adj: dict node -> list of (neighbor, length)
    # leaf_labels: dict node -> label
    
    nodes = [0, 1]
    adj = {0: [(1, min_dist / 2.0)], 1: [(0, min_dist / 2.0)]}
    leaf_labels = {0: labels[min_pair[0]], 1: labels[min_pair[1]]}
    next_node_id = 2
    
    # Remaining taxa
    remaining = set(range(n)) - set(min_pair)
    
    while remaining:
        if _time_left() < 5.0:
            break
        
        best_taxon = None
        best_position = None
        best_score = float('inf')
        
        for taxon in remaining:
            # Try inserting taxon at each edge in the current tree
            for u in adj:
                for v, length in adj[u]:
                    if v > u:  # avoid double counting
                        # Try inserting taxon on edge (u, v)
                        # The new configuration: u - taxon - v
                        # We need to determine the branch lengths.
                        # Let d(u, taxon) = x, d(taxon, v) = y
                        # We want x + y = D[u, v] approximately
                        # And we want to minimize some cost.
                        
                        # DPP criterion: minimize the increase in total tree length
                        # or equivalently, find x, y that best fit the distances.
                        
                        # For each other leaf w, the path from w to taxon goes through
                        # either u or v (whichever is closer in the tree).
                        # Actually, in a tree, the path from w to taxon is unique.
                        # If taxon is inserted on edge (u,v), then for any leaf w:
                        #   d(w, taxon) = d(w, u) + x  if the path from w to v goes through u
                        #   d(w, taxon) = d(w, v) + y  if the path from w to u goes through v
                        # But in a tree, all paths from w to the edge (u,v) go through
                        # exactly one of u or v.
                        
                        # Let's compute: for each leaf w, what is d(w, u) and d(w, v)?
                        # We can compute these from the current tree.
                        
                        # This is getting complex. Let me use a simpler approach:
                        # For each potential insertion point, compute the "score" as
                        # the sum of squared differences between observed and implied distances.
                        
                        # Actually, for DPP, the standard approach is:
                        # Given a tree T and a new leaf x, for each edge (u,v) in T,
                        # compute the optimal branch lengths x_u and x_v such that:
                        #   x_u + x_v = D[u,v]  (approximately)
                        #   and the distances from x to all other leaves are consistent.
                        
                        # The DPP score for inserting x on edge (u,v) is:
                        #   sum over all leaves w of [D[x,w] - (d(w,u) + x_u or d(w,v) + x_v)]^2
                        # minimized over x_u, x_v.
                        
                        # This is a 2-variable least squares problem.
                        
                        # Let's compute d(w, u) and d(w, v) for all leaves w.
                        # We need to know which side of the edge each leaf is on.
                        
                        # In the current tree, cutting edge (u,v) splits the tree into
                        # two components: one containing u, one containing v.
                        # Let S_u = set of leaves in u's component, S_v = set of leaves in v's component.
                        
                        # For w in S_u: d(w, x) = d(w, u) + x_u
                        # For w in S_v: d(w, x) = d(w, v) + x_v
                        
                        # We want to find x_u, x_v to minimize:
                        #   sum_{w in S_u} [D[x,w] - d(w,u) - x_u]^2 + sum_{w in S_v} [D[x,w] - d(w,v) - x_v]^2
                        
                        # This separates into two independent 1-variable least squares problems:
                        #   x_u = mean_{w in S_u} [D[x,w] - d(w,u)]
                        #   x_v = mean_{w in S_v} [D[x,w] - d(w,v)]
                        
                        # And the score is the sum of squared residuals.
                        
                        # But we also have the constraint that x_u + x_v should be close to D[u,v]
                        # (the distance between u and v in the distance matrix, if u and v are leaves)
                        # or the implied distance if u and v are internal.
                        
                        # For now, let's ignore the constraint and just use the mean formula.
                        
                        # First, determine S_u and S_v by BFS from u not going through v, and vice versa.
                        
                        # Get all leaves in the current tree
                        all_leaves = list(leaf_labels.keys())
                        
                        # BFS from u, not going through v
                        visited_u = set()
                        stack = [u]
                        while stack:
                            curr = stack.pop()
                            if curr in visited_u:
                                continue
                            visited_u.add(curr)
                            for neighbor, _ in adj[curr]:
                                if neighbor != v and neighbor not in visited_u:
                                    stack.append(neighbor)
                        
                        S_u = [leaf for leaf in all_leaves if leaf in visited_u]
                        S_v = [leaf for leaf in all_leaves if leaf not in visited_u]
                        
                        if not S_u or not S_v:
                            continue
                        
                        # Compute d(w, u) for w in S_u and d(w, v) for w in S_v
                        # using the current tree with branch lengths.
                        
                        def path_length(node1, node2, adj):
                            """Compute path length between two nodes in the tree."""
                            if node1 == node2:
                                return 0.0
                            visited = {node1}
                            queue = [(node1, 0.0)]
                            while queue:
                                curr, dist = queue.pop(0)
                                if curr == node2:
                                    return dist
                                for neighbor, length in adj[curr]:
                                    if neighbor not in visited:
                                        visited.add(neighbor)
                                        queue.append((neighbor, dist + length))
                            return float('inf')
                        
                        # Compute x_u
                        if S_u:
                            residuals_u = []
                            for w in S_u:
                                d_wu = path_length(w, u, adj)
                                residuals_u.append(D[taxon, w] - d_wu)
                            x_u = np.mean(residuals_u)
                            ss_u = np.sum((np.array(residuals_u) - x_u) ** 2)
                        else:
                            x_u = 0.0
                            ss_u = 0.0
                        
                        # Compute x_v
                        if S_v:
                            residuals_v = []
                            for w in S_v:
                                d_wv = path_length(w, v, adj)
                                residuals_v.append(D[taxon, w] - d_wv)
                            x_v = np.mean(residuals_v)
                            ss_v = np.sum((np.array(residuals_v) - x_v) ** 2)
                        else:
                            x_v = 0.0
                            ss_v = 0.0
                        
                        # Clamp to non-negative
                        x_u = max(0.0, x_u)
                        x_v = max(0.0, x_v)
                        
                        # Total score
                        score = ss_u + ss_v
                        
                        if score < best_score:
                            best_score = score
                            best_taxon = taxon
                            best_position = (u, v, x_u, x_v)
        
        if best_taxon is None:
            # Fallback: add to a random leaf
            best_taxon = next(iter(remaining))
            # Attach to first leaf with a small branch length
            first_leaf = next(iter(leaf_labels.keys()))
            best_position = (first_leaf, first_leaf, 0.1, 0.0)
        
        u, v, x_u, x_v = best_position
        
        # Insert best_taxon on edge (u, v)
        new_node = next_node_id
        next_node_id += 1
        
        # Remove edge (u, v)
        adj[u].remove((v, x_u))  # This won't work because the stored length is the original
        # Let me redo this properly.
        
        # Find and remove the edge (u, v)
        for i, (neighbor, length) in enumerate(adj[u]):
            if neighbor == v:
                adj[u].pop(i)
                break
        for i, (neighbor, length) in enumerate(adj[v]):
            if neighbor == u:
                adj[v].pop(i)
                break
        
        # Add new node
        nodes.append(new_node)
        adj[new_node] = [(u, x_u), (v, x_v)]
        adj[u].append((new_node, x_u))
        adj[v].append((new_node, x_v))
        leaf_labels[new_node] = labels[best_taxon]
        
        remaining.discard(best_taxon)
    
    # Rebuild Newick from the tree
    if not leaf_labels:
        return f"({','.join(labels)});"
    
    root = next(iter(leaf_labels.keys()))
    
    def build_newick(node, parent):
        if node in leaf_labels and len(adj[node]) == 1:
            # It's a leaf
            return leaf_labels[node]
        
        children = [x for x, _ in adj[node] if x != parent]
        if not children:
            return leaf_labels.get(node, f"n{node}")
        
        child_strs = []
        for child in children:
            child_strs.append(build_newick(child, node))
        
        return "(" + ",".join(child_strs) + ")"
    
    result = build_newick(root, -1)
    return f"({result});"


def _nni_refine(newick, D, labels):
    """
    Perform NNI refinement using a simple distance-based cost.
    For each NNI move, compute the change in the sum of distances
    implied by the tree vs. the observed distance matrix.
    """
    nodes, adj, leaves = _parse_newick(newick)
    n = len(nodes)
    
    if n <= 3:
        return newick
    
    label_to_node = {leaf: node for node, leaf in leaves.items()}
    node_to_label = {node: leaf for leaf, node in label_to_node.items()}
    
    # Precompute label indices for fast D lookup
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    def compute_cost(adj):
        """Compute the cost of the tree: sum of squared differences between
        observed and implied distances."""
        # For each pair of leaves, compute the path in the tree and sum branch lengths.
        # But we don't have branch lengths! We need to estimate them.
        # 
        # Alternative: use a topological cost. For each internal edge, compute
        # the "support" of the split it induces.
        # 
        # Let's use a different approach: for each NNI move, compute the change
        # in the number of "consistent" splits.
        # 
        # Actually, let's just use the QP cost with estimated branch lengths.
        # For each edge, estimate its length as the average distance between
        # leaves on opposite sides, minus the average within-side distances.
        
        # Build edge list
        edges = []
        edge_idx = {}
        for u in adj:
            for v in adj[u]:
                if u < v:
                    idx = len(edges)
                    edges.append((u, v))
                    edge_idx[(u, v)] = idx
                    edge_idx[(v, u)] = idx
        
        num_edges = len(edges)
        if num_edges == 0:
            return 0.0
        
        # Estimate branch lengths
        edge_lengths = {}
        for (u, v) in edges:
            # Get leaves on each side
            visited_u = set()
            stack = [u]
            while stack:
                curr = stack.pop()
                if curr in visited_u:
                    continue
                visited_u.add(curr)
                for neighbor in adj[curr]:
                    if neighbor != v and neighbor not in visited_u:
                        stack.append(neighbor)
            
            leaves_u = [node for node in visited_u if node in leaves]
            leaves_v = [node for node in adj if node not in visited_u and node in leaves]
            
            if not leaves_u or not leaves_v:
                edge_lengths[(u, v)] = 0.1
                continue
            
            # Average distance between leaves_u and leaves_v
            cross_dists = []
            for lu in leaves_u:
                for lv in leaves_v:
                    lu_label = leaves[lu]
                    lv_label = leaves[lv]
                    cross_dists.append(D[label_to_idx[lu_label], label_to_idx[lv_label]])
            
            # Average distance within leaves_u
            within_u = []
            for i in range(len(leaves_u)):
                for j in range(i + 1, len(leaves_u)):
                    lu_label = leaves[leaves_u[i]]
                    lv_label = leaves[leaves_u[j]]
                    within_u.append(D[label_to_idx[lu_label], label_to_idx[lv_label]])
            
            # Average distance within leaves_v
            within_v = []
            for i in range(len(leaves_v)):
                for j in range(i + 1, len(leaves_v)):
                    lu_label = leaves[leaves_v[i]]
                    lv_label = leaves[leaves_v[j]]
                    within_v.append(D[label_to_idx[lu_label], label_to_idx[lv_label]])
            
            avg_cross = np.mean(cross_dists) if cross_dists else 0.5
            avg_within_u = np.mean(within_u) if within_u else 0.0
            avg_within_v = np.mean(within_v) if within_v else 0.0
            
            # Branch length estimate
            bl = 0.5 * (avg_cross - avg_within_u - avg_within_v)
            edge_lengths[(u, v)] = max(0.0, bl)
            edge_lengths[(v, u)] = edge_lengths[(u, v)]
        
        # Now compute implied distances and compare to observed
        leaf_nodes = list(leaves.keys())
        num_leaves = len(leaf_nodes)
        
        total_cost = 0.0
        for i in range(num_leaves):
            for j in range(i + 1, num_leaves):
                li = leaf_nodes[i]
                lj = leaf_nodes[j]
                
                # BFS to find path
                visited = {li}
                queue = [(li, 0.0)]
                found_dist = None
                while queue:
                    curr, dist = queue.pop(0)
                    if curr == lj:
                        found_dist = dist
                        break
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            e_key = (min(curr, neighbor), max(curr, neighbor))
                            edge_len = edge_lengths.get(e_key, 0.1)
                            queue.append((neighbor, dist + edge_len))
                
                if found_dist is not None:
                    li_label = leaves[li]
                    lj_label = leaves[lj]
                    observed = D[label_to_idx[li_label], label_to_idx[lj_label]]
                    total_cost += (observed - found_dist) ** 2
        
        return total_cost
    
    initial_cost = compute_cost(adj)
    
    # Try NNI moves iteratively
    improved = True
    max_iterations = 30
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if _time_left() < 5.0:
            break
        
        # Find all NNI moves
        for u in adj:
            if u in leaves:
                continue
            for v in adj[u]:
                if v in leaves:
                    continue
                if u >= v:
                    continue
                
                u_neighbors = [x for x in adj[u] if x != v]
                v_neighbors = [x for x in adj[v] if x != u]
                
                if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                    continue
                
                a, b = u_neighbors
                c, d = v_neighbors
                
                # Try NNI move: swap b and c
                # Save state
                adj_backup = {k: list(v) for k, v in adj.items()}
                
                # Apply: remove u-b, v-c; add u-c, v-b
                if b in adj[u] and c in adj[v]:
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(c)
                    adj[c].remove(v)
                    
                    adj[u].append(c)
                    adj[c].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                    
                    new_cost = compute_cost(adj)
                    
                    if new_cost < initial_cost - 1e-10:
                        initial_cost = new_cost
                        improved = True
                        break
                    else:
                        adj.clear()
                        adj.update(adj_backup)
                
                if improved:
                    break
            
            if improved:
                break
    
    # Rebuild Newick
    if not leaves:
        return newick
    
    root_leaf = next(iter(leaves))
    
    def build_newick(node, parent):
        if node in leaves and len(adj[node]) == 1:
            return leaves[node]
        
        children = [x for x in adj[node] if x != parent]
        if not children:
            return leaves.get(node, f"n{node}")
        
        child_strs = []
        for child in children:
            child_strs.append(build_newick(child, node))
        
        return "(" + ",".join(child_strs) + ")"
    
    result = build_newick(root_leaf, -1)
    return f"({result});"


def reconstruct(sequences):
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run NJ as primary method (proven to work)
    nj_newick = _nj_build(D, labels)
    
    # Try NNI refinement on NJ result
    if _time_left() > 30.0:
        try:
            refined_newick = _nni_refine(nj_newick, D, labels)
            if refined_newick:
                nj_newick = refined_newick
        except:
            pass
    
    return nj_newick
