import time
import random

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        return True
    return False

def _find_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        xc = common & adj[w]
                        lsb2 = xc & -xc
                        x = lsb2.bit_length() - 1
                        return [u, v, w, x]
    return None

def _has_independent_6(adj, n):
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)

    def dfs(verts, depth):
        if depth == 6:
            return True
        if not verts:
            return False
        if bin(verts).count('1') < 6 - depth:
            return False
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            if dfs(new_verts, depth + 1):
                return True
    return dfs(full_mask, 0)

def _find_independent_6(adj, n):
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)

    def dfs(verts, path):
        if len(path) == 6:
            return path[:]
        if not verts:
            return None
        if bin(verts).count('1') < 6 - len(path):
            return None
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            result = dfs(new_verts, path + [u])
            if result:
                return result
        return None

    return dfs(full_mask, [])

def _is_valid_graph(n, edges):
    adj = _build_adj(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _edge_safe_k4(adj, u, v):
    """Check if adding edge (u,v) keeps graph K4-free (assuming not already present)."""
    if (adj[u] >> v) & 1:
        return True  # already present
    common = adj[u] & adj[v]
    m = common
    while m:
        lsb = m & -m
        w = lsb.bit_length() - 1
        m ^= lsb
        if common & adj[w]:
            return False
    return True

def _greedy_k4_free_maximal(n, order):
    """
    Build a maximal K4-free graph by processing vertices in `order`.
    For each vertex u, connect to ALL later vertices in order that are K4-safe.
    This yields a maximal K4-free graph (no edge can be added without creating K4).
    """
    adj = [0] * n
    pos = {v: i for i, v in enumerate(order)}
    for idx, u in enumerate(order):
        for jdx in range(idx + 1, n):
            v = order[jdx]
            if _edge_safe_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _greedy_k4_free_random(n, seed):
    """Random-order greedy maximal K4-free graph."""
    random.seed(seed)
    order = list(range(n))
    random.shuffle(order)
    return _greedy_k4_free_maximal(n, order)

def _repair(adj, n, deadline):
    """
    Conflict-driven repair: while graph has K4 or independent-6, fix one.
    - K4: remove an edge from the K4 (prefer edges that appear in fewest K4s -> approximate by degree).
    - Independent-6: add a K4-safe edge within the set; if none safe, remove a random edge outside
      that is most 'useful' (high co-degree) to reshape structure, then retry.
    Returns (adj, valid_bool).
    """
    while time.time() < deadline:
        k4 = _find_k4(adj, n)
        if k4 is not None:
            u, v, w, x = k4
            pairs = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            # Pick the edge with the smallest sum of degrees (least structural impact)
            best = min(pairs, key=lambda p: bin(adj[p[0]]).count('1') + bin(adj[p[1]]).count('1'))
            i, j = best
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            continue

        ind6 = _find_independent_6(adj, n)
        if ind6 is not None:
            # Try to add a K4-safe edge within the independent set
            added = False
            for i in range(6):
                for j in range(i + 1, 6):
                    a, b = ind6[i], ind6[j]
                    if not (adj[a] >> b) & 1 and _edge_safe_k4(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        added = True
                        break
                if added:
                    break
            if added:
                continue
            # No safe internal edge: do a "kick" - remove a random edge that is NOT in the independent set
            # This perturbs the structure to allow a different safe edge later.
            # Choose an edge with high co-degree to make a meaningful change.
            candidates = []
            for a in range(n):
                if a in ind6:
                    continue
                m = adj[a]
                while m:
                    lsb = m & -m
                    b = lsb.bit_length() - 1
                    m ^= lsb
                    if b > a and b not in ind6:
                        candidates.append((a, b))
            if candidates:
                # Pick one with high combined degree
                a, b = max(candidates, key=lambda p: bin(adj[p[0]]).count('1') + bin(adj[p[1]]).count('1'))
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
            else:
                # Fallback: remove a random edge anywhere
                all_edges = []
                for a in range(n):
                    m = adj[a]
                    while m:
                        lsb = m & -m
                        b = lsb.bit_length() - 1
                        m ^= lsb
                        if b > a:
                            all_edges.append((a, b))
                if all_edges:
                    a, b = random.choice(all_edges)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                else:
                    return adj, False
            continue

        # No K4 and no independent 6 -> valid
        return adj, True

    # Time out: check current state
    if _has_k4(adj, n):
        return adj, False
    if _has_independent_6(adj, n):
        return adj, False
    return adj, True

def _try_extend(adj_base, n_base, n_target, deadline, seed):
    """
    Try to extend a valid K4-free graph on n_base vertices to n_target vertices.
    Add (n_target - n_base) new vertices, connecting them greedily (K4-safe) to old vertices,
    then repair.
    """
    if n_target <= n_base:
        return None
    adj = [0] * n_target
    # Copy old graph
    for u in range(n_base):
        adj[u] = adj_base[u]
    # New vertices start isolated; connect greedily to old + new
    new_verts = list(range(n_base, n_target))
    random.seed(seed)
    order = new_verts[:]
    random.shuffle(order)
    for u in order:
        # Connect to all existing vertices (old + previously added new) that are K4-safe
        for v in range(n_target):
            if v == u:
                continue
            if v < u or v < n_base:
                if _edge_safe_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    # Repair
    adj2, valid = _repair(adj, n_target, deadline)
    if valid:
        return adj2
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def remaining():
        return deadline - time.time()

    best_adj = None
    best_n = 0

    # Phase 1: Build maximal K4-free graphs via greedy for several n, repair to valid.
    # Start from larger n and go down until we find a valid one.
    for n in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 22, 20, 16]:
        if remaining() < 5:
            break
        if n <= best_n:
            continue

        found = False
        # Try multiple random orderings (different maximal K4-free graphs)
        max_attempts = 10
        for attempt in range(max_attempts):
            if remaining() < 2:
                break
            t_per = min(remaining() / max(1, max_attempts - attempt), 12)
            if t_per < 0.3:
                break
            seed = 7777 + n * 13 + attempt
            adj = _greedy_k4_free_random(n, seed)
            adj2, valid = _repair(adj, n, start_time + t_per)
            if valid:
                if n > best_n:
                    best_n = n
                    best_adj = adj2
                found = True
                break

        if found:
            # We have a valid graph of size n; try to extend to n+1, n+2, ... up to 35
            # using the extend routine, to potentially beat the published 35.
            for target in range(n + 1, 36):
                if remaining() < 4:
                    break
                if target <= best_n:
                    continue
                ext = _try_extend(best_adj, best_n, target, min(remaining() - 1, 15),
                                  seed=9999 + target)
                if ext is not None:
                    if target > best_n:
                        best_n = target
                        best_adj = ext
            break  # Stop the outer n-loop; we've used extend to push further

    # Phase 2: If we still don't have a good graph, do a final bounded repair from a
    # high-quality greedy seed at the largest feasible n.
    if best_n < 20 and remaining() > 10:
        for n in [30, 28, 26, 24, 22, 20, 18, 16]:
            if remaining() < 3:
                break
            adj = _greedy_k4_free_random(n, seed=424242 + n)
            adj2, valid = _repair(adj, n, start_time + min(remaining() - 1, 10))
            if valid and n > best_n:
                best_n = n
                best_adj = adj2
                break

    # Fallback: guaranteed small valid graph (Clebsch, n=16)
    if best_adj is None or best_n < 16:
        # Build Clebsch graph
        n = 16
        vertices = [i for i in range(16) if bin(i).count('1') % 2 == 0]
        idx_map = {v: i for i, v in enumerate(vertices)}
        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                diff = bin(vertices[i] ^ vertices[j]).count('1')
                if diff in (1, 3):
                    edges.append([i, j])
        best_adj = _build_adj(n, edges)
        best_n = n

    # Final safety: verify best_adj is valid; if not, fall back to Clebsch.
    if not _is_valid_graph(best_n, _edges_from_adj(best_adj, best_n)):
        n = 16
        vertices = [i for i in range(16) if bin(i).count('1') % 2 == 0]
        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                diff = bin(vertices[i] ^ vertices[j]).count('1')
                if diff in (1, 3):
                    edges.append([i, j])
        best_adj = _build_adj(n, edges)
        best_n = n

    return {'n': best_n, 'edges': _edges_from_adj(best_adj, best_n)}
