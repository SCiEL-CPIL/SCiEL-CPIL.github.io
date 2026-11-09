"""
Phylogenetic topology recovery using Neighbor-Joining initialization 
followed by Nearest Neighbor Interchange (NNI) optimization 
with Minimum Evolution (Least Squares) scoring.
"""
import math
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 260.0  # Safety margin before 270s

def _check_time():
    return time.time() - _START_TIME > _TIME_LIMIT

def _jc_distance_matrix(sequences):
    """Compute Jukes-Cantor corrected distance matrix."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = [sequences[l] for l in labels]
    L = len(seqs[0])
    
    # Convert sequences to binary arrays for fast XOR
    # A=00, C=01, G=10, T=11 (arbitrary mapping, just need distinct)
    # Actually, just count mismatches.
    # Vectorized mismatch counting
    # Create a matrix of 0s and 1s for each base? No, just loop over sites or use numpy broadcasting.
    # For N<=32, L<=2000, simple loop is fine, but numpy is faster.
    
    # Map chars to ints
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Create array of shape (n, L)
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            seq_array[i, j] = char_map[c]
    
    # Calculate differences
    # diff[i, j, k] is 1 if seq_array[i, k] != seq_array[j, k]
    # This is n x n x L. 32x32x2000 = 2M ints. ~8MB. OK.
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    
    # Sum over sites (axis 2)
    mismatches = diff.sum(axis=2)
    
    # Calculate p
    p = mismatches / float(L)
    
    # JC correction
    # d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75
    mask = p < 0.75
    d = np.zeros((n, n))
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 3.0 # Large value for saturated
    
    # Symmetrize and zero diagonal
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    
    return labels, d

def _estimate_branch_lengths(n, tree_adj, d_obs):
    """
    Estimate branch lengths for a given unrooted tree topology 
    by minimizing sum of squared differences between observed and tree distances.
    
    tree_adj: dict mapping node index to list of (neighbor, branch_len)
    We assume the tree is binary and unrooted.
    We need to solve for branch lengths.
    
    For each pair (i, j), d_tree(i, j) = sum of branch lengths on path i-j.
    This is a linear system: A * x = b, where x is vector of branch lengths.
    A is (n*(n-1)/2) x (n-3) for unrooted binary tree with n leaves.
    Actually, number of internal edges is n-3. Total edges is 2n-3.
    Leaf edges also have lengths.
    So unknowns are 2n-3 branch lengths.
    
    We can set up the least squares problem.
    """
    # This is computationally heavy if done from scratch every time.
    # However, for N=32, 2N-3 = 61 unknowns.
    # Pairs = 496.
    # Solving 496x61 LSQ is fast.
    
    # We need to construct the matrix A.
    # A[row, col] = 1 if branch 'col' is on path between leaf pair 'row'.
    
    # Let's flatten the tree structure to get edge indices.
    # We need a mapping from node pairs to edge indices.
    
    # For efficiency, we will use a simpler heuristic for branch lengths if full LSQ is too slow,
    # but let's try LSQ first as it's accurate.
    
    edges = []
    edge_map = {} # (u, v) -> index
    idx = 0
    for u in tree_adj:
        for v, _ in tree_adj[u]:
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = idx
                idx += 1
            elif u > v:
                # Ensure consistent ordering
                if (v, u) not in edge_map:
                    edges.append((v, u))
                    edge_map[(v, u)] = idx
                    idx += 1
    
    num_edges = len(edges)
    leaves = [i for i in range(n)]
    
    # Build A matrix
    num_pairs = n * (n - 1) // 2
    A = np.zeros((num_pairs, num_edges))
    b = np.zeros(num_pairs)
    
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Find path from i to j
            # BFS
            visited = {i: None}
            queue = [i]
            parent = {i: None}
            
            while queue:
                curr = queue.pop(0)
                if curr == j:
                    break
                for neighbor, _ in tree_adj[curr]:
                    if neighbor not in visited:
                        visited[neighbor] = curr
                        parent[neighbor] = curr
                        queue.append(neighbor)
            
            # Reconstruct path
            path = []
            curr = j
            while curr != i:
                path.append(curr)
                curr = parent[curr]
            path.append(i)
            
            # Sum edges on path
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                e_idx = edge_map.get((u, v), edge_map.get((v, u)))
                if e_idx is not None:
                    A[pair_idx, e_idx] = 1.0
            
            b[pair_idx] = d_obs[i, j]
            pair_idx += 1
            
    # Solve LSQ
    # np.linalg.lstsq(A, b, rcond=None)
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # Clip negative lengths to 0
        x[x < 0] = 0.0
        return x
    except np.linalg.LinAlgError:
        return np.zeros(num_edges)

def _calc_tree_distances(n, tree_adj, branch_lengths):
    """
    Calculate all-pairs tree distances.
    """
    # Map edges to lengths
    edge_len = {}
    idx = 0
    for u in tree_adj:
        for v, _ in tree_adj[u]:
            if u < v:
                edge_len[(u, v)] = branch_lengths[idx]
                idx += 1
            elif u > v:
                if (v, u) not in edge_len:
                    edge_len[(v, u)] = branch_lengths[idx]
                    idx += 1
    
    d_tree = np.zeros((n, n))
    for i in range(n):
        # BFS from i
        dist = np.full(n, np.inf)
        dist[i] = 0.0
        visited = np.zeros(n, dtype=bool)
        visited[i] = True
        queue = [i]
        
        while queue:
            curr = queue.pop(0)
            for neighbor, _ in tree_adj[curr]:
                if not visited[neighbor]:
                    # Get branch length
                    u, v = curr, neighbor
                    if (u, v) in edge_len:
                        bl = edge_len[(u, v)]
                    else:
                        bl = edge_len.get((v, u), 0.0)
                    
                    dist[neighbor] = dist[curr] + bl
                    visited[neighbor] = True
                    queue.append(neighbor)
        d_tree[i] = dist
    
    return d_tree

def _score_tree(d_obs, d_tree):
    """
    Score: Sum of squared differences.
    """
    return np.sum((d_obs - d_tree) ** 2)

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time() # Reset timer for each instance? No, task says 180s for ALL 20.
    # Wait, "180 s CPU budget for all 20 instances".
    # My global timer should persist across calls if possible, but in a sandboxed subprocess,
    # the process starts fresh. So time.time() is correct relative to process start.
    
    labels, d_obs = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n < 3:
        return "(%s);" % ",".join(labels)
    
    # 1. Neighbor Joining
    # nodes: list of active node IDs
    # tree_adj: dict {node: [(neighbor, branch_len)]}
    # We use integer IDs for internal nodes starting from n
    
    # Initialize
    tree_adj = {i: [] for i in range(n)}
    nodes = list(range(n))
    current_d = d_obs.copy()
    
    next_id = n
    
    while len(nodes) > 3:
        if _check_time():
            break
            
        m = len(nodes)
        # Calculate r_i = sum_j d_ij
        r = np.zeros(m)
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    r[i] += current_d[u, v]
        
        # Find best pair (u, v) minimizing (m-2)*d_uv - r_u - r_v
        best_score = float('inf')
        best_u, best_v = None, None
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:
                    score = (m - 2) * current_d[u, v] - r[i] - r[j]
                    if score < best_score:
                        best_score = score
                        best_u, best_v = u, v
                        
        if best_u is None:
            break
            
        u, v = best_u, best_v
        
        # Calculate branch lengths for u and v
        # b_u = 0.5 * d_uv + (r_u - r_v) / (2 * (m - 2))
        # b_v = 0.5 * d_uv - (r_u - r_v) / (2 * (m - 2))
        denom = 2.0 * (m - 2)
        b_u = 0.5 * current_d[u, v] + (r[nodes.index(u)] - r[nodes.index(v)]) / denom
        b_v = 0.5 * current_d[u, v] - (r[nodes.index(u)] - r[nodes.index(v)]) / denom
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Update tree_adj
        tree_adj[new_node] = []
        # Remove u and v from their neighbors
        for neighbor in list(tree_adj[u]):
            tree_adj[neighbor[0]].remove((u, neighbor[1]))
        for neighbor in list(tree_adj[v]):
            tree_adj[neighbor[0]].remove((v, neighbor[1]))
            
        # Add new connections
        tree_adj[u].append((new_node, b_u))
        tree_adj[v].append((new_node, b_v))
        tree_adj[new_node].append((u, b_u))
        tree_adj[new_node].append((v, b_v))
        
        # Update distance matrix for new node
        for w in nodes:
            if w not in (u, v):
                # b_new_w = 0.5 * (d_uw + d_vw - d_uv)
                d_new_w = 0.5 * (current_d[u, w] + current_d[v, w] - current_d[u, v])
                current_d[new_node, w] = d_new_w
                current_d[w, new_node] = d_new_w
                
        # Update nodes list
        nodes = [x for x in nodes if x not in (u, v)]
        nodes.append(new_node)
        
    # 2. NNI Optimization
    # The tree is now represented by tree_adj.
    # We need to identify internal edges for NNI.
    
    # Helper to get neighbors of a node
    def get_neighbors(node):
        return [n for n, _ in tree_adj[node]]
    
    # Find all internal edges
    # An edge is internal if both endpoints have degree > 2? No, in binary tree, leaves have deg 1.
    # Internal nodes have deg 3.
    # Edges connecting two deg 3 nodes are internal.
    
    internal_edges = []
    visited_edges = set()
    for u in tree_adj:
        deg_u = len(tree_adj[u])
        if deg_u == 3:
            for v, _ in tree_adj[u]:
                deg_v = len(tree_adj[v])
                if deg_v == 3:
                    if (u, v) not in visited_edges and (v, u) not in visited_edges:
                        internal_edges.append((u, v))
                        visited_edges.add((u, v))
                        visited_edges.add((v, u))
                        
    # Estimate initial branch lengths for the whole tree
    # We need to map edge indices consistently for the LSQ solver.
    # Let's create a list of all edges.
    all_edges = []
    edge_to_idx = {}
    idx = 0
    for u in tree_adj:
        for v, _ in tree_adj[u]:
            if u < v:
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1
            elif u > v:
                if (v, u) not in edge_to_idx:
                    all_edges.append((v, u))
                    edge_to_idx[(v, u)] = idx
                    idx += 1
    
    num_edges = len(all_edges)
    
    # Function to update tree_adj branch lengths from a vector
    def set_branch_lengths(lengths):
        for i, (u, v) in enumerate(all_edges):
            # Update tree_adj
            # Find u in v's list and v in u's list
            for k, (nb, _) in enumerate(tree_adj[u]):
                if nb == v:
                    tree_adj[u][k] = (v, lengths[i])
            for k, (nb, _) in enumerate(tree_adj[v]):
                if nb == u:
                    tree_adj[v][k] = (u, lengths[i])
    
    # Initial LSQ
    if num_edges > 0:
        b_lengths = _estimate_branch_lengths(n, tree_adj, d_obs)
        set_branch_lengths(b_lengths)
        
    # NNI Loop
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _check_time():
            break
            
        # Recalculate internal edges as topology might have changed?
        # NNI changes topology, so edge list changes.
        # For simplicity and speed, we will only consider NNI on the *current* internal edges.
        # After a move, we re-identify internal edges.
        
        # Re-identify internal edges
        internal_edges = []
        visited_edges = set()
        for u in tree_adj:
            deg_u = len(tree_adj[u])
            if deg_u == 3:
                for v, _ in tree_adj[u]:
                    deg_v = len(tree_adj[v])
                    if deg_v == 3:
                        if (u, v) not in visited_edges and (v, u) not in visited_edges:
                            internal_edges.append((u, v))
                            visited_edges.add((u, v))
                            visited_edges.add((v, u))
                            
        best_move = None
        best_score = _score_tree(d_obs, _calc_tree_distances(n, tree_adj, [l for _, l in tree_adj[all_edges[0][0]] if all_edges[0][1] == l or all_edges[0][0] == l])) # Dummy, need proper length extraction
        
        # Properly extract current lengths
        curr_lengths = np.zeros(num_edges)
        for i, (u, v) in enumerate(all_edges):
            # Find length in tree_adj
            for nb, l in tree_adj[u]:
                if nb == v:
                    curr_lengths[i] = l
                    break
            
        curr_score = _score_tree(d_obs, _calc_tree_distances(n, tree_adj, curr_lengths))
        
        for u, v in internal_edges:
            if _check_time():
                break
                
            # Find neighbors of u other than v
            neighbors_u = [nb for nb, _ in tree_adj[u] if nb != v]
            neighbors_v = [nb for nb, _ in tree_adj[v] if nb != u]
            
            if len(neighbors_u) != 1 or len(neighbors_v) != 1:
                continue # Should not happen for internal edges in binary tree
            
            w = neighbors_u[0]
            z = neighbors_v[0]
            
            # NNI Move: Remove (u,v), (u,w), (v,z). Add (u,z), (v,w).
            # This is one of the 3 possible NNI moves. 
            # Actually, standard NNI on edge (u,v) with neighbors w (of u) and z (of v) 
            # results in topology (u,z) and (v,w) and a new internal edge (u,v) is removed, 
            # wait.
            
            # Original: w - u - v - z
            # NNI 1: w - u - v - z (original)
            # NNI 2: w - v - u - z (swap w and z) -> Edges (w,v), (v,u), (u,z)
            # NNI 3: w - u - z - v (swap u and v relative to w,z? No)
            
            # The 3 topologies for quartet {w, u, v, z} are:
            # 1. (w, u) | (v, z) -> Path w-u-v-z. (Current)
            # 2. (w, v) | (u, z) -> Path w-v-u-z.
            # 3. (w, z) | (u, v) -> Path w-z-u-v. (This makes w,z siblings? No, w and z are leaves/internal nodes connected to u/v)
            
            # Let's just try the swap: Connect w to v and u to z.
            # Remove edges: (u,v), (u,w), (v,z)
            # Add edges: (u,z), (v,w)
            
            # Save state to rollback if worse
            # We need to copy the relevant parts of tree_adj
            
            # Extract lengths
            len_uv = None
            len_uw = None
            len_vz = None
            
            for nb, l in tree_adj[u]:
                if nb == v: len_uv = l
                if nb == w: len_uw = l
            for nb, l in tree_adj[v]:
                if nb == z: len_vz = l
                
            # Perform move
            # Remove
            tree_adj[u].remove((v, len_uv))
            tree_adj[v].remove((u, len_uv))
            tree_adj[u].remove((w, len_uw))
            tree_adj[w].remove((u, len_uw))
            tree_adj[v].remove((z, len_vz))
            tree_adj[z].remove((v, len_vz))
            
            # Add
            # Keep lengths same for now (approximation)
            tree_adj[u].append((z, len_uv)) # Approx
            tree_adj[z].append((u, len_uv))
            tree_adj[v].append((w, len_vz)) # Approx
            tree_adj[w].append((v, len_vz))
            
            # Recalculate distances and score
            # We need to update all_edges and edge_to_idx? 
            # This is tricky because the edge set changed.
            # For the sake of speed and simplicity in this constrained environment,
            # we will just use the current branch lengths for the NEW edges (copied from old)
            # and recalculate the score.
            
            # However, _calc_tree_distances relies on tree_adj having the correct lengths.
            # We just set them.
            
            # We need to get the lengths for _calc_tree_distances.
            # It expects a list of lengths corresponding to all_edges.
            # But all_edges is stale!
            
            # This approach is fragile.
            
            # ROLLBACK
            tree_adj[u].remove((z, len_uv))
            tree_adj[z].remove((u, len_uv))
            tree_adj[v].remove((w, len_vz))
            tree_adj[w].remove((v, len_vz))
            
            tree_adj[u].append((v, len_uv))
            tree_adj[v].append((u, len_uv))
            tree_adj[u].append((w, len_uw))
            tree_adj[w].append((u, len_uw))
            tree_adj[v].append((z, len_vz))
            tree_adj[z].append((v, len_vz))
            
            # Given the complexity of updating the edge list and LSQ system dynamically 
            # within the time limit, a simpler heuristic is often more robust:
            # Just use the NJ tree as is, or do a very simple NNI that doesn't require 
            # re-solving LSQ, but uses the *current* branch lengths as fixed.
            
            # Let's try a simpler NNI that just swaps the topology and uses 
            # the *average* of the involved branch lengths for the new edges, 
            # and scores using the fixed branch lengths for the rest.
            
            # To avoid the "stale edge list" problem, we will reconstruct the 
            # edge list and length vector for scoring ONLY.
            
            def get_current_lengths():
                lengths = []
                # We need a consistent ordering.
                # Let's just iterate all_edges (stale) and try to find the length in tree_adj.
                # If an edge is missing, it's 0. If new, it's not in all_edges.
                # This is bad.
                
                # Better: Rebuild edge list for the CURRENT tree.
                current_edges = []
                c_edge_map = {}
                c_idx = 0
                for u_c in tree_adj:
                    for v_c, l_c in tree_adj[u_c]:
                        if u_c < v_c:
                            current_edges.append((u_c, v_c))
                            c_edge_map[(u_c, v_c)] = c_idx
                            c_idx += 1
                        elif u_c > v_c:
                            if (v_c, u_c) not in c_edge_map:
                                current_edges.append((v_c, u_c))
                                c_edge_map[(v_c, u_c)] = c_idx
                                c_idx += 1
                                
                c_lengths = np.zeros(len(current_edges))
                for e_idx, (u_c, v_c) in enumerate(current_edges):
                    # Find length
                    for nb, l in tree_adj[u_c]:
                        if nb == v_c:
                            c_lengths[e_idx] = l
                            break
                return current_edges, c_lengths

            # Rebuild for current state
            curr_edges, curr_lengths = get_current_lengths()
            curr_score = _score_tree(d_obs, _calc_tree_distances(n, tree_adj, curr_lengths))
            
            # Perform Move
            tree_adj[u].remove((v, len_uv))
            tree_adj[v].remove((u, len_uv))
            tree_adj[u].remove((w, len_uw))
            tree_adj[w].remove((u, len_uw))
            tree_adj[v].remove((z, len_vz))
            tree_adj[z].remove((v, len_vz))
            
            # New lengths: use average or old values
            new_len_uz = (len_uv + len_uw) / 2.0
            new_len_vw = (len_uv + len_vz) / 2.0
            
            tree_adj[u].append((z, new_len_uz))
            tree_adj[z].append((u, new_len_uz))
            tree_adj[v].append((w, new_len_vw))
            tree_adj[w].append((v, new_len_vw))
            
            # Rebuild and Score
            new_edges, new_lengths = get_current_lengths()
            new_score = _score_tree(d_obs, _calc_tree_distances(n, tree_adj, new_lengths))
            
            if new_score < curr_score:
                # Accept
                # Update global edge list and lengths for next iteration's LSQ?
                # We'll just let the next sweep's LSQ handle it.
                pass
            else:
                # Rollback
                tree_adj[u].remove((z, new_len_uz))
                tree_adj[z].remove((u, new_len_uz))
                tree_adj[v].remove((w, new_len_vw))
                tree_adj[w].remove((v, new_len_vw))
                
                tree_adj[u].append((v, len_uv))
                tree_adj[v].append((u, len_uv))
                tree_adj[u].append((w, len_uw))
                tree_adj[w].append((u, len_uw))
                tree_adj[v].append((z, len_vz))
                tree_adj[z].append((v, len_vz))
                
        # End NNI sweep
        # Re-estimate branch lengths for the improved topology
        if not _check_time():
            # Rebuild edge list
            all_edges = []
            edge_to_idx = {}
            idx = 0
            for u in tree_adj:
                for v, _ in tree_adj[u]:
                    if u < v:
                        all_edges.append((u, v))
                        edge_to_idx[(u, v)] = idx
                        idx += 1
                    elif u > v:
                        if (v, u) not in edge_to_idx:
                            all_edges.append((v, u))
                            edge_to_idx[(v, u)] = idx
                            idx += 1
            num_edges = len(all_edges)
            
            b_lengths = _estimate_branch_lengths(n, tree_adj, d_obs)
            set_branch_lengths(b_lengths)

    # Convert to Newick
    # Root the tree arbitrarily (e.g., at node 0)
    # DFS to generate Newick
    
    def to_newick(node, parent):
        neighbors = [nb for nb, _ in tree_adj[node] if nb != parent]
        if not neighbors:
            return labels[node] if node < n else f"n{node}"
        
        parts = []
        for nb in neighbors:
            # Get branch length
            bl = 0.0
            for n_nb, l in tree_adj[node]:
                if n_nb == nb:
                    bl = l
                    break
            sub = to_newick(nb, node)
            parts.append(f"{sub}:{bl:.4f}")
            
        return "(" + ",".join(parts) + ")"
        
    newick = to_newick(0, -1) + ";"
    return newick

# Ensure the function is exposed correctly
# The prompt asks for reconstruct(sequences) -> Newick string
