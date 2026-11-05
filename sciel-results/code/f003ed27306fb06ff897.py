<<<<<<< SEARCH
def _fast_local_search(adj, deg, cur_score, t0, rng, iters_budget):
    """Incremental local search. Uses triangle list; rebuilds only when needed.

    For speed, we recompute the max triangle sum from scratch each step
    but with early-exit and only over existing triangles. We also track
    the set of 'active' high-sum triangles.
    """
    tris = _build_tri_list(adj)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    start_temp = 3.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    # Cache: map from frozenset edge to index for O(1) remove
    edge_set = set(edges)

    iters = 0
    while iters < iters_budget:
        if (iters % 1000 == 0) and (time.time() - t0 > DEADLINE):
            break
        # recheck deadline cheaply every 2000 iters
        if iters % 2000 == 0 and iters > 0 and (time.time() - t0) > DEADLINE:
            break

        if not edges or not nonedges:
            break

        er = edges[rng.randrange(len(edges))]
        ea = nonedges[rng.randrange(len(nonedges))]
        if er == ea:
            iters += 1
            continue
        u, v = er
        x, y = ea

        # apply
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1; deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1; deg[y] += 1

        tris = _build_tri_list(adj)
        new_score = _max_tri_sum(adj, deg, tris)

        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.5
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-d / temp) > rng.random()

        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            # update lists
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
            tris = tris  # keep for next loop
        else:
            # undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1; deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1; deg[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score
=======
def _fast_local_search(adj_np, deg_np, cur_score, t0, rng, iters_budget):
    """Incremental local search using NumPy arrays for fast scoring."""
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj_np[i, j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj_np[i, j]]

    best_score = cur_score
    best_adj = adj_np.copy()
    best_deg = deg_np.copy()

    start_temp = 3.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    while iters < iters_budget:
        if (iters % 500 == 0) and (time.time() - t0 > DEADLINE):
            break

        if not edges or not nonedges:
            break

        er_idx = rng.randrange(len(edges))
        ea_idx = rng.randrange(len(nonedges))
        
        er = edges[er_idx]
        ea = nonedges[ea_idx]
        
        if er == ea:
            iters += 1
            continue
            
        u, v = er
        x, y = ea

        # apply swap
        adj_np[u, v] = 0
        adj_np[v, u] = 0
        deg_np[u] -= 1
        deg_np[v] -= 1
        adj_np[x, y] = 1
        adj_np[y, x] = 1
        deg_np[x] += 1
        deg_np[y] += 1

        new_score = _score_np(adj_np, deg_np)

        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.5
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-d / temp) > rng.random()

        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj_np.copy()
                best_deg = deg_np.copy()
            # update lists
            edges[er_idx] = ea
            nonedges[ea_idx] = er
        else:
            # undo
            adj_np[x, y] = 0
            adj_np[y, x] = 0
            deg_np[x] -= 1
            deg_np[y] -= 1
            adj_np[u, v] = 1
            adj_np[v, u] = 1
            deg_np[u] += 1
            deg_np[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score
>>>>>>> REPLACE

Finally, update `construct` to use NumPy arrays and return the correct format.

