import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles.
    Only considers 4-cycles that use the new edge (u,v): u-v-x-y-u."""
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if y not in neigh[x]:
                continue
            # cycle u-v-x-y-u: colors c, adj[(v,x)], adj[(x,y)], adj[(y,u)]
            if c == adj[(v, x)] or c == adj[(x, y)] or c == adj[(y, u)]:
                return False
            if adj[(v, x)] == adj[(x, y)] or adj[(v, x)] == adj[(y, u)]:
                return False
            if adj[(x, y)] == adj[(y, u)]:
                return False
    return True


def _greedy_build(order, color_pref, rng, time_budget):
    """Greedy build.  For each edge in order, try colours in a preference
    order (color_pref) that is shuffled slightly, and add the first colour
    that keeps all 4-cycles rainbow."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None] * N for _ in range(N)]
    edges = []

    for (u, v) in order:
        if time.time() > time_budget:
            break
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        if not available:
            continue
        # Build a preference order: start with the structured colour if it is
        # available, then the rest shuffled.
        pref = color_pref(u, v)
        ordered = []
        if pref in available:
            ordered.append(pref)
            available.remove(pref)
        rng.shuffle(available)
        ordered.extend(available)
        for c in ordered:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    return edges


def _try_add_edges(edges, adj, neigh, color_at, rng, time_budget, color_pref):
    """Try to add more edges to an existing valid graph."""
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    existing = set()
    for u, v, c in edges:
        existing.add((min(u, v), max(u, v)))
    candidates = [(u, v) for (u, v) in all_pairs if (u, v) not in existing]
    rng.shuffle(candidates)

    for (u, v) in candidates:
        if time.time() > time_budget:
            break
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        if not available:
            continue
        pref = color_pref(u, v)
        ordered = []
        if pref in available:
            ordered.append(pref)
            available.remove(pref)
        rng.shuffle(available)
        ordered.extend(available)
        for c in ordered:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    return edges


def _local_search(edges, adj, neigh, color_at, rng, time_budget, color_pref):
    """Local search: repeatedly try to add missing edges; if stuck, remove a
    random edge and try to add two or more back."""
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    for _ in range(5000):
        if time.time() > time_budget:
            break
        existing = set()
        for u, v, c in edges:
            existing.add((min(u, v), max(u, v)))
        candidates = [(u, v) for (u, v) in all_pairs if (u, v) not in existing]
        if not candidates:
            break
        rng.shuffle(candidates)
        added_any = False
        for (u, v) in candidates[:80]:
            if time.time() > time_budget:
                break
            used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
            used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
            available = [c for c in range(C) if c not in used_u and c not in used_v]
            if not available:
                continue
            pref = color_pref(u, v)
            ordered = []
            if pref in available:
                ordered.append(pref)
                available.remove(pref)
            rng.shuffle(available)
            ordered.extend(available)
            for c in ordered:
                if _check_add(adj, neigh, u, v, c):
                    adj[(u, v)] = c
                    adj[(v, u)] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    color_at[u][v] = c
                    color_at[v][u] = c
                    edges.append([u, v, c])
                    added_any = True
                    break
        if not added_any and edges:
            # Remove a random edge and try to refill
            idx = rng.randrange(len(edges))
            a, b, c = edges[idx]
            edges.pop(idx)
            del adj[(a, b)]
            del adj[(b, a)]
            neigh[a].discard(b)
            neigh[b].discard(a)
            color_at[a][b] = None
            color_at[b][a] = None
            # try to refill
            existing2 = set()
            for u, v, cc in edges:
                existing2.add((min(u, v), max(u, v)))
            candidates2 = [(u, v) for (u, v) in all_pairs if (u, v) not in existing2]
            rng.shuffle(candidates2)
            for (u, v) in candidates2[:60]:
                if time.time() > time_budget:
                    break
                used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
                used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
                available2 = [c for c in range(C) if c not in used_u and c not in used_v]
                if not available2:
                    continue
                pref = color_pref(u, v)
                ordered = []
                if pref in available2:
                    ordered.append(pref)
                    available2.remove(pref)
                rng.shuffle(available2)
                ordered.extend(available2)
                for c in ordered:
                    if _check_add(adj, neigh, u, v, c):
                        adj[(u, v)] = c
                        adj[(v, u)] = c
                        neigh[u].add(v)
                        neigh[v].add(u)
                        color_at[u][v] = c
                        color_at[v][u] = c
                        edges.append([u, v, c])
                        break
    return edges


def _build_initial_state(edges):
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None] * N for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)
        color_at[u][v] = c
        color_at[v][u] = c
    return adj, neigh, color_at


def _xor_color(u, v):
    """A structured colour: (u ^ v) mod C."""
    return (u ^ v) % C


def _additive_color(u, v):
    """Another structured colour: (u + v) mod C."""
    return (u + v) % C


def _mixed_color(u, v):
    """Mixed structured colour."""
    return (u * 3 + v * 7) % C


def construct():
    start = time.time()
    best_edges = []
    best_len = 0

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Try multiple color preference functions
    color_prefs = [_xor_color, _additive_color, _mixed_color]

    # --- Phase 1: multiple greedy constructions with different seeds and color prefs ---
    for trial in range(150):
        if time.time() > DEADLINE - 30:
            break
        rng = random.Random(trial * 137 + 42)
        order = all_pairs[:]
        rng.shuffle(order)
        cp = color_prefs[trial % len(color_prefs)]
        edges = _greedy_build(order, cp, rng, DEADLINE - 30)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
        # Early exit if we are already very dense
        if best_len >= 200:
            break

    # --- Phase 2: local search on the best graph ---
    if best_edges and time.time() < DEADLINE - 15:
        adj, neigh, color_at = _build_initial_state(best_edges)
        rng = random.Random(12345)
        budget = DEADLINE - 10
        improved = _local_search(best_edges[:], adj, neigh, color_at, rng, budget, _xor_color)
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved

    # --- Phase 3: final top-up ---
    if best_edges and time.time() < DEADLINE - 3:
        adj, neigh, color_at = _build_initial_state(best_edges)
        rng = random.Random(99999)
        final = _try_add_edges(best_edges[:], adj, neigh, color_at, rng, DEADLINE - 1, _xor_color)
        if len(final) > best_len:
            best_edges = final

    if not best_edges:
        return [[0, v, v] for v in range(1, N)]
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
