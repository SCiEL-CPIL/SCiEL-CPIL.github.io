"""Bootstrap search program: Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools

DEADLINE = 150
SAFETY_MARGIN = 12
START_TIME = None


def find_violation_fast(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None.
    Uses bitmasks; checks size-6 subsets first (most numerous), then larger.
    """
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return combo

    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok:
                return combo
    return None


def find_violation(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None."""
    return find_violation_fast(adj_masks, n)


def random_graph(n, p, rng):
    adj_masks = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
    return adj_masks, edges


def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e


def is_regular_subset(combo, adj_masks):
    """Check if the induced subgraph on combo is regular."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    d0 = (adj_masks[combo[0]] & mask).bit_count()
    for v in combo[1:]:
        if (adj_masks[v] & mask).bit_count() != d0:
            return False
    return True


def local_search_cp(adj_masks, n, rng, deadline):
    """Constraint propagation local search: flip edges that resolve the most conflicts."""
    cur = adj_masks[:]
    
    # Precompute all 6-vertex subsets
    all_subsets = list(itertools.combinations(range(n), 6))
    
    # Initial violation status
    violating = set()
    for i, combo in enumerate(all_subsets):
        if is_regular_subset(combo, cur):
            violating.add(i)
    
    if not violating:
        # Check larger subsets
        for k in range(7, n + 1):
            for combo in itertools.combinations(range(n), k):
                if is_regular_subset(combo, cur):
                    return n, get_edges_from_masks(cur, n), False
        return n, get_edges_from_masks(cur, n), True
    
    # For each edge, precompute which subsets it belongs to
    edge_to_subsets = {}
    for i, combo in enumerate(all_subsets):
        for a in range(len(combo)):
            for b in range(a + 1, len(combo)):
                u, v = combo[a], combo[b]
                if u > v:
                    u, v = v, u
                key = (u, v)
                if key not in edge_to_subsets:
                    edge_to_subsets[key] = []
                edge_to_subsets[key].append(i)
    
    it = 0
    max_iter = 500
    while it < max_iter:
        if time.time() > deadline:
            break
        
        if not violating:
            # Check larger subsets
            found_large = False
            for k in range(7, n + 1):
                for combo in itertools.combinations(range(n), k):
                    if is_regular_subset(combo, cur):
                        found_large = True
                        break
                if found_large:
                    break
            if not found_large:
                return n, get_edges_from_masks(cur, n), True
            # If there's a large violation, pick a random 6-subset from it and continue
            # For now, just break and let the outer loop handle it
            break
        
        # Score each edge by how many violating subsets it resolves
        # Flipping an edge in a violating subset may break regularity
        best_edge = None
        best_score = -1
        
        # Sample a subset of candidate edges for efficiency
        all_edges = list(edge_to_subsets.keys())
        if len(all_edges) > 200:
            candidates = rng.sample(all_edges, 200)
        else:
            candidates = all_edges
        
        for edge in candidates:
            u, v = edge
            sub_idx_list = edge_to_subsets.get(edge, [])
            if not sub_idx_list:
                continue
            # Count how many of these subsets are currently violating
            count_violating = sum(1 for si in sub_idx_list if si in violating)
            if count_violating > best_score:
                best_score = count_violating
                best_edge = edge
        
        if best_edge is None or best_score <= 0:
            # No good edge found; try random
            if all_edges:
                best_edge = rng.choice(all_edges)
            else:
                break
        
        u, v = best_edge
        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        
        # Update violation status for affected subsets
        affected = edge_to_subsets.get((u, v), [])
        for si in affected:
            combo = all_subsets[si]
            if is_regular_subset(combo, cur):
                violating.add(si)
            else:
                violating.discard(si)
        
        it += 1
    
    viol = find_violation(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None


def local_search(adj_masks, n, rng, deadline):
    """Flip edges to eliminate regular induced subgraphs. Prefer flipping edges inside a violating set."""
    cur = adj_masks[:]

    viol = find_violation(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    it = 0
    while it < 800:
        if time.time() > deadline:
            break

        viol = find_violation(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = list(viol)
        internal = []
        internal_nonedges = []
        for i in range(len(vset)):
            for j in range(i + 1, len(vset)):
                u, v = vset[i], vset[j]
                if cur[u] & (1 << v):
                    internal.append((u, v))
                else:
                    internal_nonedges.append((u, v))
        
        if internal:
            u, v = rng.choice(internal)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            u = rng.randrange(n)
            v = rng.randrange(n)
            while u == v:
                v = rng.randrange(n)
            if u > v:
                u, v = v, u

        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1

    viol = find_violation(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        max_attempts = 15 if n >= 19 else 25

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            p = 0.3 + 0.4 * rng.random()

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break

            # Try constraint propagation first
            new_n, new_edges, new_valid = local_search_cp(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

            # Fallback to standard local search
            if time.time() > deadline:
                break
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            continue

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
