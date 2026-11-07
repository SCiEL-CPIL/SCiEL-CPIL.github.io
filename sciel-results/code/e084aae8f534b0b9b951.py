"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    # Fast validity check using bitmasks
    # adj[i] is a bitmask of neighbors of vertex i
    # deg[i] is degree of vertex i
    
    def is_valid_fast(adj_masks, deg, n):
        """Check validity using bitmasks. Returns True if valid."""
        for u in range(n):
            mask_u = adj_masks[u]
            # For each neighbor v of u (v > u to avoid double counting)
            v_mask = mask_u
            while v_mask:
                v = (v_mask & (-v_mask)).bit_length() - 1
                v_mask &= v_mask - 1
                if v <= u:
                    continue
                # s in N(u) \ {v}, t in N(v) \ {u, s}
                # Check if any s, t have same degree
                su_mask = mask_u & ~(1 << v)
                tv_mask = adj_masks[v] & ~(1 << u)
                # For each s in su_mask, check if there exists t in tv_mask with t != s and deg[s] == deg[t]
                s_mask = su_mask
                while s_mask:
                    s = (s_mask & (-s_mask)).bit_length() - 1
                    s_mask &= s_mask - 1
                    t_mask = tv_mask & ~(1 << s)
                    # Check if any t in t_mask has deg[t] == deg[s]
                    target_deg = deg[s]
                    # Build mask of vertices with target_deg
                    # We can precompute deg_to_mask but for simplicity just iterate
                    tm = t_mask
                    while tm:
                        t = (tm & (-tm)).bit_length() - 1
                        tm &= tm - 1
                        if deg[t] == target_deg:
                            return False
        return True

    def compute_deg(adj_masks, n):
        return [bin(adj_masks[i]).count('1') for i in range(n)]

    def edges_from_masks(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def try_add_edges_greedy(adj_masks, n, deadline):
        """Try to add all missing edges that maintain validity. Returns (adj_masks, edge_count)."""
        deg = compute_deg(adj_masks, n)
        added = True
        while added:
            if time.time() > deadline:
                break
            added = False
            for i in range(n):
                if time.time() > deadline:
                    return adj_masks, len(edges_from_masks(adj_masks, n))
                for j in range(i + 1, n):
                    if not (adj_masks[i] & (1 << j)):
                        # Try adding edge (i, j)
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        deg[i] += 1
                        deg[j] += 1
                        if is_valid_fast(adj_masks, deg, n):
                            added = True
                        else:
                            adj_masks[i] &= ~(1 << j)
                            adj_masks[j] &= ~(1 << i)
                            deg[i] -= 1
                            deg[j] -= 1
        return adj_masks, len(edges_from_masks(adj_masks, n))

    def perturbation_reoptimize(adj_masks, n, deadline, rng):
        """Perturb the graph by removing some edges, then re-optimize."""
        adj2 = list(adj_masks)
        # Remove k random edges
        k = rng.randint(3, 8)
        for _ in range(k):
            # Pick a random existing edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i < j and (adj2[i] & (1 << j)):
                adj2[i] &= ~(1 << j)
                adj2[j] &= ~(1 << i)
        # Re-optimize
        adj2, count = try_add_edges_greedy(adj2, n, deadline)
        return adj2, count

    best_edges = None
    best_count = -1

    # Strategy 1: Start with various bipartite splits
    for split in range(4, 12):
        if time.time() > deadline:
            break
        A_sz = split
        B_sz = n - split
        adj_masks = [0] * n
        for i in range(A_sz):
            for j in range(A_sz, n):
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
        adj_masks, count = try_add_edges_greedy(adj_masks, n, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges_from_masks(adj_masks, n)

    # Strategy 2: Start with random dense graphs and optimize
    rng = random.Random(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        adj_masks = [0] * n
        p = 0.7 + 0.15 * (attempt / 50)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        adj_masks, count = try_add_edges_greedy(adj_masks, n, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges_from_masks(adj_masks, n)

    # Strategy 3: Perturbation + re-optimization from best found
    if best_edges is not None and time.time() < deadline - 5:
        best_adj = [0] * n
        for u, v in best_edges:
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        
        for perturb_attempt in range(30):
            if time.time() > deadline:
                break
            adj2, count = perturbation_reoptimize(best_adj, n, deadline, rng)
            if count > best_count:
                best_count = count
                best_edges = edges_from_masks(adj2, n)
                best_adj = list(adj2)

    # Fallback: star graph (always valid)
    if best_edges is None:
        best_edges = [[0, v] for v in range(1, n)]
    else:
        # Ensure proper types
        best_edges = [[int(u), int(v)] for u, v in best_edges]

    return best_edges
# EVOLVE-BLOCK-END
