import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under the 150s kill

    N = 16
    # All unordered vertex pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    pair_index = {}
    for i, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = i

    # ---------- Exact 3-colourability check (bitmask, DSATUR ordering) ----------
    def is_3_colorable(adj_mask):
        # Build static ordering: highest degree first (DSATUR-ish heuristic)
        order = list(range(N))
        order.sort(key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        # For each vertex in order, list its neighbours that appear later in order
        later_neighbors = [[] for _ in range(N)]
        for idx in range(N):
            v = order[idx]
            m = adj_mask[v]
            for j in range(idx + 1, N):
                u = order[j]
                if m & (1 << u):
                    later_neighbors[idx].append(j)
        colors = [-1] * N
        # forbidden color bitmask per position (bit c set = color c forbidden)
        forb = [0] * N

        def backtrack(idx):
            if idx == N:
                return True
            # Choose colour
            avail = 7 & ~forb[idx]
            if avail == 0:
                return False
            # Try colours; prefer the one that forbids the most (most constraining)
            c = avail
            while c:
                bit = c & (-c)
                c -= bit
                col = bit.bit_length() - 1
                colors[idx] = col
                # update forb for later neighbours
                touched = []
                for j in later_neighbors[idx]:
                    old = forb[j]
                    new = old | bit
                    forb[j] = new
                    touched.append((j, old))
                if backtrack(idx + 1):
                    return True
                # undo
                for j, old in touched:
                    forb[j] = old
                colors[idx] = -1
            return False

        return backtrack(0)

    # ---------- Clebsch graph seed ----------
    vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]  # 16 even-parity 5-bit
    clebsch_adj = [0] * N
    clebsch_edges = []
    for u_idx in range(N):
        for v_idx in range(u_idx + 1, N):
            if bin(vectors[u_idx] ^ vectors[v_idx]).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)

    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40

    def update_best(adj_mask, edges, count):
        nonlocal best_edges, best_count
        if count > best_count:
            best_count = count
            best_edges = [e[:] for e in edges]

    # ---------- Triangle-free edge-add test ----------
    def can_add(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    # ---------- Local search: add edges (with 4-chromatic guard) ----------
    def local_add(adj_mask, edges, count):
        nonlocal best_edges, best_count
        while time.time() < deadline:
            # collect addable edges
            addable = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj_mask[u] & (1 << v)) and can_add(adj_mask, u, v):
                        addable.append((u, v))
            if not addable:
                break
            random.shuffle(addable)
            progress = False
            for u, v in addable:
                if time.time() > deadline:
                    break
                if adj_mask[u] & (1 << v):
                    continue
                if not can_add(adj_mask, u, v):
                    continue
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                if not is_3_colorable(adj_mask):
                    update_best(adj_mask, edges, count + 1)
                    count += 1
                    progress = True
                    break
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    edges.pop()
            if not progress:
                break
        return adj_mask, edges, count

    # ---------- Local search: 2-opt edge swaps (structural escape) ----------
    def local_swap(adj_mask, edges, count):
        nonlocal best_edges, best_count
        while time.time() < deadline:
            # candidate edges to remove (those whose removal keeps 4-chromatic)
            removable = []
            for e in edges:
                u, v = e
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                if not is_3_colorable(adj_mask):
                    removable.append((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            # candidate edges to add (triangle-free)
            addable = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj_mask[u] & (1 << v)) and can_add(adj_mask, u, v):
                        addable.append((u, v))
            if not removable or not addable:
                break
            random.shuffle(removable)
            random.shuffle(addable)
            progress = False
            for ru, rv in removable:
                if time.time() > deadline:
                    break
                # remove (ru,rv)
                adj_mask[ru] &= ~(1 << rv)
                adj_mask[rv] &= ~(1 << ru)
                for au, av in addable:
                    if time.time() > deadline:
                        break
                    if adj_mask[au] & (1 << av):
                        continue
                    if not can_add(adj_mask, au, av):
                        continue
                    adj_mask[au] |= (1 << av)
                    adj_mask[av] |= (1 << au)
                    if not is_3_colorable(adj_mask):
                        # accept swap: update edge list
                        edges.remove([ru, rv])
                        edges.append([au, av])
                        update_best(adj_mask, edges, count)
                        progress = True
                        break
                    # revert add
                    adj_mask[au] &= ~(1 << av)
                    adj_mask[av] &= ~(1 << au)
                # revert remove
                adj_mask[ru] |= (1 << rv)
                adj_mask[rv] |= (1 << ru)
                if progress:
                    break
            if not progress:
                break
        return adj_mask, edges, count

    # ---------- Structural change: randomized greedy from scratch ----------
    def random_greedy():
        if time.time() > deadline:
            return None
        # random permutation of all edges
        perm = all_pairs[:]
        random.shuffle(perm)
        # bias: interleave with Clebsch edges to keep good structure
        order = perm[:]
        for e in clebsch_edges:
            if random.random() < 0.5:
                order.insert(random.randint(0, len(order) - 1), e)
        adj_mask = [0] * N
        edges = []
        count = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if adj_mask[u] & (1 << v):
                continue
            if not can_add(adj_mask, u, v):
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])
            count += 1
        return adj_mask, edges, count

    # ---------- Main search loop ----------
    # Phase 1: local search from Clebsch (add + swap)
    adj_mask = clebsch_adj[:]
    edges = [e[:] for e in clebsch_edges]
    count = best_count
    adj_mask, edges, count = local_add(adj_mask, edges, count)
    adj_mask, edges, count = local_swap(adj_mask, edges, count)

    # Phase 2: randomized greedy restarts + local refinement
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        res = random_greedy()
        if res is None:
            break
        adj_mask, edges, count = res
        # Only refine if promising (close to or above best)
        if count + 6 < best_count:
            continue
        adj_mask, edges, count = local_add(adj_mask, edges, count)
        if count > best_count - 2:
            adj_mask, edges, count = local_swap(adj_mask, edges, count)
        # Occasionally also refine the Clebsch neighbourhood again
        if iterations % 5 == 0:
            adj2 = clebsch_adj[:]
            edges2 = [e[:] for e in clebsch_edges]
            count2 = len(clebsch_edges)
            adj2, edges2, count2 = local_add(adj2, edges2, count2)
            adj2, edges2, count2 = local_swap(adj2, edges2, count2)

    # Final safety: ensure we return a valid (triangle-free, 4-chromatic) graph
    # best_edges is only updated when is_3_colorable returned False, so it's 4-chromatic.
    # Triangle-free is maintained by construction (can_add check).
    return best_edges
