"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def priority(adjacency):
    n = adjacency.shape[0]
    # Work with a boolean adjacency matrix for efficiency
    adj = adjacency.astype(bool)
    
    # 1. Generate an initial independent set using Minimum Degree Greedy
    # This is a strong baseline.
    degrees = adj.sum(axis=1)
    in_set = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    
    # We simulate the greedy process to determine the order of selection
    # and thus the "priority" score.
    # Instead of just returning the set, we assign scores based on the order
    # they would be picked by a greedy decoder.
    
    # However, a pure Min-Degree greedy might be suboptimal.
    # Let's try to find a *better* independent set using local search,
    # then assign high scores to its members.
    
    # Helper to get a maximal independent set via greedy min-degree
    def greedy_min_degree(adj, n):
        deg = adj.sum(axis=1).copy()
        rem = np.ones(n, dtype=bool)
        s = np.zeros(n, dtype=bool)
        while rem.any():
            # Find vertex with min degree among remaining
            # Mask degrees of non-remaining to inf
            d = np.where(rem, deg, np.inf)
            v = np.argmin(d)
            if not rem[v]:
                break
            s[v] = True
            # Remove v and its neighbors
            rem[v] = False
            neighbors = adj[v] & rem
            rem[neighbors] = False
            # Update degrees of remaining vertices
            # deg[neighbors] -= 1 is not quite right because we remove them.
            # We just need to update degrees of vertices that stay.
            # Actually, we can just recalculate or update.
            # For n=200, recalculation is fast.
            if rem.any():
                deg[rem] = adj[rem].sum(axis=1)
        return s

    # Run greedy min-degree to get a baseline set
    s_best = greedy_min_degree(adj, n)
    best_size = s_best.sum()
    
    # Try to improve with local search (swap moves)
    # A simple local search: try to replace a vertex in S with two of its neighbors
    # If S contains v, and N(v) contains u, w which are not in S and not adjacent to each other,
    # and u, w are not adjacent to other vertices in S \ {v}, then we can swap v for u, w.
    # This increases size by 1.
    
    improved = True
    s_current = s_best.copy()
    iterations = 0
    max_iter = 500 # Limit iterations for time safety
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        # Iterate through vertices in S
        s_indices = np.where(s_current)[0]
        for v in s_indices:
            # Check if we can remove v and add 2 neighbors
            # Neighbors of v that are not in S
            neighbors = np.where(adj[v] & ~s_current)[0]
            if len(neighbors) < 2:
                continue
            
            # Try all pairs of neighbors? O(k^2). k is degree.
            # For n=200, max degree 199. 199^2 ~ 40k.
            # Number of vertices in S ~ 100.
            # 100 * 40k = 4M operations. Might be slow in Python if not vectorized.
            # Let's be smarter or limit.
            
            # Check if any pair of neighbors (u, w) can replace v.
            # Conditions:
            # 1. u, w not in S (already filtered)
            # 2. u, w not adjacent to each other
            # 3. u, w not adjacent to any other vertex in S \ {v}
            
            # Let S_others = S \ {v}
            # Mask for vertices in S_others
            s_others_mask = s_current.copy()
            s_others_mask[v] = False
            
            # For each neighbor u, check if it is adjacent to S_others
            # If u is adjacent to S_others, it can't be added.
            # So we only consider neighbors u that are "free" (not adjacent to S_others).
            
            free_neighbors = []
            for u in neighbors:
                # Check if u is adjacent to any vertex in S_others
                # adj[u, s_others_mask].any()
                if not np.any(adj[u, s_others_mask]):
                    free_neighbors.append(u)
            
            if len(free_neighbors) < 2:
                continue
            
            # Now find a pair (u, w) in free_neighbors that are not adjacent to each other
            # And then we swap.
            # We just need to find *one* valid pair to improve.
            found_pair = False
            fl = len(free_neighbors)
            for i in range(fl):
                if found_pair: break
                u = free_neighbors[i]
                for j in range(i+1, fl):
                    w = free_neighbors[j]
                    # Check if u and w are adjacent
                    if not adj[u, w]:
                        # Valid pair!
                        # Remove v, add u and w
                        s_current[v] = False
                        s_current[u] = True
                        s_current[w] = True
                        improved = True
                        found_pair = True
                        break
            
            if found_pair:
                # Restart loop to check other vertices? Or just continue?
                # Continuing might miss other improvements, but it's a heuristic.
                # Breaking and restarting is safer for convergence.
                break

    # If we found a better set, use it.
    if s_current.sum() > best_size:
        s_best = s_current
    
    # Assign scores:
    # Vertices in s_best get high score.
    # To ensure the decoder picks exactly s_best (or a superset),
    # we assign score 1.0 to s_best vertices.
    # For others, we assign 0.0.
    # The decoder will pick all s_best vertices because they are independent.
    # It might also pick other vertices if they are independent of s_best and each other.
    # This is fine, as it only increases the set size.
    
    scores = np.zeros(n, dtype=float)
    scores[s_best] = 1.0
    
    # To break ties among s_best vertices (if any, though they are all 1.0),
    # it doesn't matter as they are independent.
    # To break ties among non-s_best vertices, we can use degree to guide the decoder
    # to pick low-degree vertices first if it extends the set.
    # This helps if the decoder picks vertices outside s_best.
    
    # Let's add a small secondary score for non-selected vertices based on negative degree
    # to encourage the decoder to pick low-degree vertices if it goes beyond s_best.
    non_selected = ~s_best
    scores[non_selected] = -0.001 * adjacency[non_selected].sum(axis=1)
    
    return scores
# EVOLVE-BLOCK-END
