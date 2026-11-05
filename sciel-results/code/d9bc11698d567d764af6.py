import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}. Maximize n.
    """
    deadline = time.time() + 140  # safety margin before 150s kill

    # Track the best valid graph found (highest n, then highest min degree)
    best_n = 39
    best_min_deg = 0
    best_edges = []

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        required = n - 39

        # Strategy A: Greedy from scratch + local search (multiple seeds)
        for seed in range(50):
            if time.time() > deadline:
                break
            adj = _greedy_seed(n, seed)
            result = _local_search(adj, n, required, deadline)
            if result is not None:
                n_found, min_deg, edges_found = result
                if n_found > best_n or (n_found == best_n and min_deg > best_min_deg):
                    best_n = n_found
                    best_min_deg = min_deg
                    best_edges = edges_found
                if min_deg >= required:
                    return {'n': n_found, 'edges': edges_found}
            if time.time() > deadline:
                break

    # Fallback
    return {'n': best_n, 'edges': best_edges}


def _greedy_seed(n, seed):
    """
    Greedily build a C4-free graph by adding edges in random order,
    checking C4 constraint via vectorized method.
    """
    random.seed(seed)
    adj = np.zeros((n, n), dtype=bool)
    degrees = np.zeros(n, dtype=int)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)

    for u, v in all_edges:
        if not adj[u, v] and not _has_c4(adj, u, v):
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1

    return adj


def _has_c4(adj, u, v):
    """
    Vectorized check: does adding edge (u,v) create a C4?
    A C4 exists iff there is x != u,v with adj[u,x] and adj[x,y] and adj[y,v] for some y != u,v,x.
    Equivalent: (N(u) \\ {v}) has a vertex x such that N(x) \\ {u,v} intersects N(v).
    Vectorized: for all x in N(u)\\{v}, check if adj[x, N(v)\\{u,v}] has any True.
    """
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]

    # x candidates: neighbors of u, excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    x_candidates = np.where(mask_u)[0]

    if len(x_candidates) == 0:
        return False

    # For each x in x_candidates, check if N(x) intersects N(v) \\ {u, v}
    # adj[x_candidates, :] gives the adjacency rows for all x_candidates
    # We need: for some x, adj[x, y] for some y in N(v) \\ {u, v}
    
    # Create a mask for y in N(v) \\ {u, v}
    mask_v = adj[v].copy()
    mask_v[u] = False
    mask_v[v] = False

    # For each x in x_candidates, check if adj[x, mask_v] has any True
    # adj[x_candidates, :] is a (len(x_candidates), n) boolean array
    # We want: any row has a True in the columns where mask_v is True
    sub = adj[x_candidates, :]  # (len(x_candidates), n)
    # For each row, check if there's a True in the mask_v columns
    result = sub[:, mask_v]  # (len(x_candidates), count of mask_v True)
    return bool(np.any(result))


def _local_search(adj, n, required, deadline):
    """
    Local search to maximize minimum degree while maintaining C4-freeness.
    Uses edge swaps (remove one edge, add another) to escape local minima.
    Returns (n, min_degree, edges) or None if no improvement.
    """
    adj = adj.copy()
    degrees = adj.sum(axis=1).astype(int)
    best_min_deg = int(degrees.min())
    best_adj = adj.copy()
    best_edges_count = int(adj.sum() // 2)

    # Repeatedly try to add an edge or perform a swap that increases the minimum degree
    for iteration in range(30000):
        if time.time() > deadline:
            break

        min_deg = int(degrees.min())
        if min_deg >= required:
            break

        # Find vertices with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        random.shuffle(min_vertices.tolist())

        improved = False

        # Phase 1: Try adding edges from low-degree vertices
        for u in min_vertices:
            if time.time() > deadline:
                break

            # Try adding edges from u to non-neighbors, sorted by target degree
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            if len(non_neighbors) == 0:
                continue

            # Sort by degree of target (ascending) to balance degrees
            order = non_neighbors[np.argsort(degrees[non_neighbors])]

            for v in order:
                if time.time() > deadline:
                    break
                if not _has_c4(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
                    break

            if improved:
                break

        # Phase 2: If no simple addition worked, try edge swaps
        if not improved:
            # Try removing an edge (a,b) where both have high degree,
            # and adding an edge (u,c) where u has low degree
            # This can help escape local minima

            # Find high-degree vertices to remove edges from
            high_deg_vertices = np.where(degrees >= min_deg + 2)[0]
            if len(high_deg_vertices) == 0:
                high_deg_vertices = np.where(degrees >= min_deg + 1)[0]

            if len(high_deg_vertices) > 0:
                # Randomly shuffle for diversity
                random.shuffle(high_deg_vertices.tolist())

                swap_found = False
                for a in high_deg_vertices[:10]:  # Limit to avoid excessive time
                    if time.time() > deadline:
                        break

                    # Find neighbors of a to consider removing
                    neighbors_a = np.where(adj[a])[0]
                    if len(neighbors_a) == 0:
                        continue

                    random.shuffle(neighbors_a.tolist())

                    for b in neighbors_a[:10]:  # Limit to avoid excessive time
                        if time.time() > deadline:
                            break

                        # Temporarily remove edge (a, b)
                        adj[a, b] = False
                        adj[b, a] = False
                        degrees[a] -= 1
                        degrees[b] -= 1

                        # Try adding edges from low-degree vertices
                        for u in min_vertices:
                            if time.time() > deadline:
                                break

                            non_neighbors_u = np.where(~adj[u])[0]
                            non_neighbors_u = non_neighbors_u[non_neighbors_u != u]
                            if len(non_neighbors_u) == 0:
                                continue

                            # Sort by degree of target (ascending)
                            order = non_neighbors_u[np.argsort(degrees[non_neighbors_u])]

                            for c in order:
                                if time.time() > deadline:
                                    break
                                if c == a or c == b:
                                    continue
                                if not _has_c4(adj, u, c):
                                    # Check that this swap improves or maintains min degree
                                    new_degrees = degrees.copy()
                                    new_degrees[u] += 1
                                    new_degrees[c] += 1
                                    if int(new_degrees.min()) >= min_deg:
                                        # Commit the swap
                                        adj[u, c] = True
                                        adj[c, u] = True
                                        degrees[u] += 1
                                        degrees[c] += 1
                                        improved = True
                                        swap_found = True
                                        break
                                # Restore edge (a, b) if we didn't add a new edge
                                if not swap_found:
                                    adj[a, b] = True
                                    adj[b, a] = True
                                    degrees[a] += 1
                                    degrees[b] += 1
                                    break

                            if swap_found:
                                break

                        if swap_found:
                            break

                        # Restore edge (a, b) if no swap was found
                        if not swap_found:
                            adj[a, b] = True
                            adj[b, a] = True
                            degrees[a] += 1
                            degrees[b] += 1

                    if swap_found:
                        break

            # If still no improvement, try random edge additions as a last resort
            if not improved:
                all_missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                if len(all_missing) == 0:
                    break
                random.shuffle(all_missing)
                found = False
                for u, v in all_missing[:50]:
                    if time.time() > deadline:
                        break
                    if not _has_c4(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        degrees[u] += 1
                        degrees[v] += 1
                        found = True
                        break
                if not found:
                    break

        # Update best
        cur_min = int(degrees.min())
        cur_edges = int(adj.sum() // 2)
        if cur_min > best_min_deg or (cur_min == best_min_deg and cur_edges > best_edges_count):
            best_min_deg = cur_min
            best_adj = adj.copy()
            best_edges_count = cur_edges

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])

    return (n, best_min_deg, edges)

# EVOLVE-BLOCK-END
