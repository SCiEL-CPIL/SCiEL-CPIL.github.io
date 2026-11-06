"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time


def _jc_distance(p):
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    # Count differences between all pairs
    # For efficiency, use numpy if available, else fallback
    try:
        import numpy as np
        # Convert sequences to integer arrays for faster comparison
        # A=0, C=1, G=2, T=3
        int_seqs = []
        for s in seqs:
            arr = np.empty(L, dtype=np.int8)
            for i, c in enumerate(s):
                if c == 'A':
                    arr[i] = 0
                elif c == 'C':
                    arr[i] = 1
                elif c == 'G':
                    arr[i] = 2
                else:
                    arr[i] = 3
            int_seqs.append(arr)
        
        int_matrix = np.array(int_seqs, dtype=np.int8)
        # Calculate pairwise differences
        # diff[i,j] = number of positions where seq_i != seq_j
        # Broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
        # This might be memory heavy for large n*L, but n<=32, L<=2000 -> 32*32*2000*1 byte = 2MB. Fine.
        diff_count = np.sum(int_matrix[:, np.newaxis, :] != int_matrix[np.newaxis, :, :], axis=2)
        p_matrix = diff_count.astype(np.float64) / L
        # Apply JC correction
        # Handle p >= 0.7499
        p_matrix = np.where(p_matrix >= 0.7499, 0.7498, p_matrix)
        dist_matrix = -0.75 * np.log(1.0 - 4.0 * p_matrix / 3.0)
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix
    except ImportError:
        # Fallback to pure Python
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                diffs = sum(1 for x, y in zip(seqs[i], seqs[j]) if x != y)
                p = diffs / L
                d = _jc_distance(p)
                dist[i][j] = d
                dist[j][i] = d
        return dist


def _nj_newick(dist_matrix, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    current_dist = [row[:] for row in dist_matrix]
    
    while len(nodes) > 2:
        m = len(nodes)
        # Compute total distances
        total = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    total[i] += current_dist[nodes[i]][nodes[j]]
        
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(m):
            for j in range(i + 1, m):
                u = nodes[i]
                v = nodes[j]
                d_uv = current_dist[u][v]
                criterion = (m - 2) * d_uv - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
        
        u = nodes[best_i]
        v = nodes[best_j]
        
        # Create new node
        new_idx = n + len(nodes) - 2 # Just a unique ID, doesn't matter much as we use trees dict
        # Actually, let's just use a string key or a new integer index that we manage.
        # Since we are removing nodes, let's just pick a new index from a pool or use a counter.
        # Simpler: use the index of the new node in the list of nodes.
        # But we need to update the distance matrix.
        # Let's use a new integer ID that is not in the current nodes list.
        # Since max n is 32, and we add at most n-2 nodes, total nodes < 2*n.
        new_id = max(nodes) + 1
        
        # Calculate branch lengths
        # r_u = 0.5 * d_uv
        # r_v = 0.5 * d_uv
        # But wait, the standard NJ formula for the new node w:
        # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
        # The branch length from u to w is 0.5 * d(u, v)
        # The branch length from v to w is 0.5 * d(u, v)
        
        d_uv = current_dist[u][v]
        branch_len = 0.5 * d_uv
        
        # Update trees
        trees[u] = f"({trees[u]},{trees[v]})" # This is wrong, we need to store the internal node structure
        # Actually, trees[i] should be the Newick string for the subtree rooted at i.
        # When we join u and v, the new node w represents the internal node.
        # The subtree for w is (subtree_u, subtree_v).
        # But we also need to keep track of the topology.
        # Let's just build the Newick string at the end or update it.
        # Since we are doing NJ, the topology is determined by the joins.
        # Let's store the join structure.
        # trees[new_id] = f"({trees[u]},{trees[v]})"
        
        # We need to update the distance matrix for the new node w
        # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
        
        # Remove u and v from nodes
        new_nodes = [x for x in nodes if x not in (u, v)]
        new_nodes.append(new_id)
        
        # Update distance matrix
        # We need to add a row and column for new_id
        # And remove rows and columns for u and v
        # This is O(n^2) per step, which is O(n^3) total. Acceptable for n<=32.
        
        # Let's just create a new distance matrix for the new set of nodes
        # But we need to map old indices to new ones.
        # Actually, it's easier to just keep the full matrix and a list of active nodes.
        # But updating the full matrix is messy.
        # Let's just use a dictionary for distances between active nodes.
        # Or, since n is small, we can just copy the relevant parts.
        
        # Let's stick to the list of active nodes and a 2D list of distances for those nodes.
        # Re-indexing is annoying.
        
        # Alternative: Keep the original indices.
        # current_dist is n x n.
        # We just need to update the distances involving new_id.
        # d(new_id, x) for x in new_nodes (excluding new_id)
        
        # Let's just create a new 2D list for the active nodes.
        # active_nodes = new_nodes
        # new_dist = [[0.0] * len(active_nodes) for _ in range(len(active_nodes))]
        # for i, ui in enumerate(active_nodes):
        #     for j, vj in enumerate(active_nodes):
        #         if ui == new_id:
        #             # d(new_id, vj)
        #             if vj == new_id:
        #                 new_dist[i][j] = 0.0
        #             else:
        #                 # vj is in original nodes
        #                 new_dist[i][j] = 0.5 * (current_dist[u][vj] + current_dist[v][vj] - d_uv)
        #         else:
        #             if vj == new_id:
        #                 new_dist[i][j] = new_dist[j][i] # symmetry
        #             else:
        #                 new_dist[i][j] = current_dist[ui][vj]
        
        # This is getting complicated. Let's use the dictionary approach or just accept the O(n^3) with list re-indexing.
        # Given n<=32, even a naive O(n^3) is fine.
        
        # Let's restart the loop logic with a simpler data structure.
        # nodes: list of labels (strings or ints)
        # dist: dict (i, j) -> distance
        # This is what the original code did. It was slow but correct.
        # The numpy part was for the initial matrix calculation.
        # The NJ loop itself is the bottleneck if done in pure Python with dicts.
        # Let's optimize the NJ loop using lists and indices.
        
        break # Break out of this flawed implementation and rewrite properly below.
    
    # The above implementation was getting messy. Let's write a clean one.
    pass


def reconstruct(sequences):
    import time
    start_time = time.time()
    deadline = start_time + 170  # Safety margin for 180s budget
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # 1. Compute distance matrix
    dist_matrix = _compute_distance_matrix(sequences, labels)
    
    # Convert to dict for easy updating and access by Node ID
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = dist_matrix[i][j]
                dist[(j, i)] = dist_matrix[i][j]

    active_nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active_nodes) > 2:
        if time.time() > deadline:
            # Fallback: return a star tree or the current best effort
            if n > 2:
                # Pick an arbitrary root, say 0
                root = 0
                others = [x for x in active_nodes if x != root]
                inner = ",".join(trees[x] for x in others)
                return f"({trees[root]},{inner});"
            else:
                return f"({','.join(trees[x] for x in active_nodes)});"

        m = len(active_nodes)
        
        # Compute total distances (Q matrix row sums)
        total = {node: 0.0 for node in active_nodes}
        for u in active_nodes:
            for v in active_nodes:
                if u != v:
                    total[u] += dist.get((u, v), 0.0)
        
        best_criterion = float('inf')
        best_u = -1
        best_v = -1
        
        for i in range(m):
            u = active_nodes[i]
            for j in range(i + 1, m):
                v = active_nodes[j]
                d_uv = dist.get((u, v), 0.0)
                criterion = (m - 2) * d_uv - total[u] - total[v]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_u = u
                    best_v = v
        
        u = best_u
        v = best_v
        d_uv = dist.get((u, v), 0.0)
        
        # Create new node
        w = next_id
        next_id += 1
        
        # Update trees
        trees[w] = f"({trees[u]},{trees[v]})"
        
        # Update distances
        # Remove u and v
        for x in active_nodes:
            if x != u and x != v:
                # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
                d_ux = dist.get((u, x), 0.0)
                d_vx = dist.get((v, x), 0.0)
                d_wx = 0.5 * (d_ux + d_vx - d_uv)
                dist[(w, x)] = d_wx
                dist[(x, w)] = d_wx
                
                # Clean up old distances
                if (u, x) in dist: del dist[(u, x)]
                if (x, u) in dist: del dist[(x, u)]
                if (v, x) in dist: del dist[(v, x)]
                if (x, v) in dist: del dist[(x, v)]
                
        # Clean up u-v distance
        if (u, v) in dist: del dist[(u, v)]
        if (v, u) in dist: del dist[(v, u)]
        
        # Update active nodes
        active_nodes.remove(u)
        active_nodes.remove(v)
        active_nodes.append(w)
        
    # Final 3 nodes
    if len(active_nodes) == 3:
        u, v, w = active_nodes
        d_uv = dist.get((u, v), 0.0)
        d_uw = dist.get((u, w), 0.0)
        d_vw = dist.get((v, w), 0.0)
        
        min_d = min(d_uv, d_uw, d_vw)
        if min_d == d_uv:
            a, b, c = u, v, w
        elif min_d == d_uw:
            a, b, c = u, w, v
        else:
            a, b, c = v, w, u
            
        # Join a and b
        new_id_final = next_id
        trees[new_id_final] = f"({trees[a]},{trees[b]})"
        final_newick = f"({trees[new_id_final]},{trees[c]})"
    else:
        final_newick = "(" + ",".join(trees[x] for x in active_nodes) + ")"
        
    return final_newick + ";"
# EVOLVE-BLOCK-END
