import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under the 150s kill

    N = 16
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    # ---------- Fast 3-colourability check using bitmasks ----------
    def is_3_colorable(adj_mask):
        # Use DSATUR-like: pick vertex with most coloured neighbours, then highest degree
        # For speed on 16 vertices, we use a fixed ordering by degree descending
        order = list(range(N))
        order.sort(key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        
        # Precompute for each position in order, the later neighbours
        later_neighbours = [[] for _ in range(N)]
        for idx in range(N):
            v = order[idx]
            m = adj_mask[v]
            for j in range(idx + 1, N):
                u = order[j]
                if m & (1 << u):
                    later_neighbours[idx].append(j)
        
        colors = [-1] * N
        forb = [0] * N  # bitmask of forbidden colors per position
        
        def backtrack(idx):
            if idx == N:
                return True
            avail = 7 & ~forb[idx]
            if avail == 0:
                return False
            # Try colors in order of most constraining
            c = avail
            while c:
                bit = c & (-c)
                c -= bit
                col = bit.bit_length() - 1
                colors[idx] = col
                touched = []
                for j in later_neighbours[idx]:
                    old = forb[j]
                    new = old | bit
                    forb[j] = new
                    touched.append((j, old))
                if backtrack(idx + 1):
                    return True
                for j, old in touched:
                    forb[j] = old
                colors[idx] = -1
            return False
        
        return backtrack(0)

    # ---------- Clebsch graph (known optimal for 16 vertices) ----------
    # The Clebsch graph is a 5-regular triangle-free graph on 16 vertices with chromatic number 4
    # It has 40 edges and is the unique (up to isomorphism) maximum triangle-free 4-chromatic graph on 16 vertices
    vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]  # 16 even-parity 5-bit vectors
    clebsch_adj = [0] * N
    clebsch_edges = []
    for u_idx in range(N):
        for v_idx in range(u_idx + 1, N):
            if bin(vectors[u_idx] ^ vectors[v_idx]).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)

    best_edges = [e[:] for e in clebsch_edges]
    best_count = len(clebsch_edges)  # 40

    def update_best(adj_mask, edges, count):
        nonlocal best_edges, best_count
        if count > best_count:
            best_count = count
            best_edges = [e[:] for e in edges]

    # ---------- Triangle-free check ----------
    def can_add(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    # ---------- Exact search: try to add edges one by one, maintaining 4-chromatic ----------
    def local_add(adj_mask, edges, count):
        nonlocal best_edges, best_count
        while time.time() < deadline:
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

    # ---------- Local search: edge swaps ----------
    def local_swap(adj_mask, edges, count):
        nonlocal best_edges, best_count
        while time.time() < deadline:
            removable = []
            for e in edges:
                u, v = e
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                if not is_3_colorable(adj_mask):
                    removable.append((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
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
                        edges.remove([ru, rv])
                        edges.append([au, av])
                        update_best(adj_mask, edges, count)
                        progress = True
                        break
                    adj_mask[au] &= ~(1 << av)
                    adj_mask[av] &= ~(1 << au)
                adj_mask[ru] |= (1 << rv)
                adj_mask[rv] |= (1 << ru)
                if progress:
                    break
            if not progress:
                break
        return adj_mask, edges, count

    # ---------- Main search ----------
    # Start from Clebsch and try to improve
    adj_mask = clebsch_adj[:]
    edges = [e[:] for e in clebsch_edges]
    count = best_count
    adj_mask, edges, count = local_add(adj_mask, edges, count)
    adj_mask, edges, count = local_swap(adj_mask, edges, count)

    # Try randomized restarts
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Randomized greedy construction
        perm = all_pairs[:]
        random.shuffle(perm)
        # Bias toward Clebsch edges
        for e in clebsch_edges:
            if random.random() < 0.7:
                order_idx = random.randint(0, len(perm) - 1)
                perm.insert(order_idx, e)
        
        adj_mask2 = [0] * N
        edges2 = []
        count2 = 0
        for u, v in perm:
            if time.time() > deadline:
                break
            if adj_mask2[u] & (1 << v):
                continue
            if not can_add(adj_mask2, u, v):
                continue
            adj_mask2[u] |= (1 << v)
            adj_mask2[v] |= (1 << u)
            edges2.append([u, v])
            count2 += 1
        
        # Only refine if promising
        if count2 + 5 < best_count:
            continue
        
        adj_mask2, edges2, count2 = local_add(adj_mask2, edges2, count2)
        if count2 > best_count - 2:
            adj_mask2, edges2, count2 = local_swap(adj_mask2, edges2, count2)

    return best_edges
