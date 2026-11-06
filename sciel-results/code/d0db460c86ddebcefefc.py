import time
import random
import itertools

# ---------------------------------------------------------------------------
# Precompute Q3 edge sets (all 840 distinct edge-sets of Q3 on vertices 0..7)
# ---------------------------------------------------------------------------
def _gen_q3_patterns():
    base = []
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j) in (1, 2, 4):
                base.append((i, j))
    pats = set()
    for perm in itertools.permutations(range(8)):
        pats.add(frozenset(
            (min(perm[i], perm[j]), max(perm[i], perm[j]))
            for (i, j) in base
        ))
    return list(pats)

Q3_PATTERNS = _gen_q3_patterns()

# Edge index in K8 (28 edges): i<j, idx = 7*i - i*(i-1)//2 + (j-i-1)
def _e8(i, j):
    if i > j:
        i, j = j, i
    return 7 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute pattern bitmasks (28-bit)
PATTERN_MASKS = []
for pat in Q3_PATTERNS:
    m = 0
    for (i, j) in pat:
        m |= 1 << _e8(i, j)
    PATTERN_MASKS.append(m)
PATTERN_MASKS = list(set(PATTERN_MASKS))

# For each of the 28 K8 edges, list of pattern-mask indices that contain it
EDGE_IN_PATTERNS = [[] for _ in range(28)]
for pi, pm in enumerate(PATTERN_MASKS):
    for e in range(28):
        if pm & (1 << e):
            EDGE_IN_PATTERNS[e].append(pi)

# ---------------------------------------------------------------------------
# Q3 counter: count Q3 subgraphs in a graph on 16 vertices.
# Uses bitmasks over 8-vertex subsets.
# ---------------------------------------------------------------------------
def _count_q3(adj, n=16):
    """Count Q3 subgraphs. adj[i] is a bitmask of neighbors of i."""
    count = 0
    for verts in itertools.combinations(range(n), 8):
        mask = 0
        for i in range(8):
            u = verts[i]
            for j in range(i + 1, 8):
                v = verts[j]
                if adj[u] & (1 << v):
                    mask |= 1 << _e8(i, j)
        for pm in PATTERN_MASKS:
            if (mask & pm) == pm:
                count += 1
    return count

def _count_q3_with_edge(adj, new_edge, n=16):
    """Count Q3 subgraphs that use new_edge. Only check 8-sets containing both endpoints."""
    u, v = new_edge
    others = [x for x in range(n) if x != u and x != v]
    count = 0
    for subset in itertools.combinations(others, 6):
        verts = list(subset) + [u, v]
        mask = 0
        for i in range(8):
            a = verts[i]
            for j in range(i + 1, 8):
                b = verts[j]
                if adj[a] & (1 << b):
                    mask |= 1 << _e8(i, j)
        for pm in PATTERN_MASKS:
            if (mask & pm) == pm:
                count += 1
    return count

# ---------------------------------------------------------------------------
# construct()
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = start + 118  # safety margin before 150s kill
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Start with K3,13 (39 edges, known Q3-free)
    adj = [0] * n
    edges = set()
    for a in range(3):
        for b in range(3, n):
            edges.add((a, b))
            adj[a] |= 1 << b
            adj[b] |= 1 << a

    best_edges = set(edges)
    best_count = len(edges)

    def _add_edge(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        edges.add((u, v))

    def _remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.discard((u, v))

    def _try_greedy_add(candidates):
        """Greedy: try to add each candidate edge if it doesn't create a Q3."""
        nonlocal best_edges, best_count
        random.shuffle(candidates)
        improved = False
        for e in candidates[:]:
            if time.time() > deadline - 5:
                break
            u, v = e
            if (u, v) in edges:
                continue
            _add_edge(u, v)
            if _count_q3_with_edge(adj, e) == 0:
                improved = True
                best_count = len(edges)
                if best_count > len(best_edges):
                    best_edges = set(edges)
            else:
                _remove_edge(u, v)
            if e in candidates:
                candidates.remove(e)
        return improved

    def _local_swap_search():
        """1-1 swap: remove one edge, try to add another. Accept if Q3-free and net gain or same count."""
        nonlocal best_edges, best_count
        edge_list = list(edges)
        random.shuffle(edge_list)
        improved = False
        for rem in edge_list:
            if time.time() > deadline - 5:
                break
            u1, v1 = rem
            _remove_edge(u1, v1)
            # Try to add random candidate edges
            cands = [e for e in all_edges if e not in edges]
            random.shuffle(cands)
            for add in cands[:20]:
                if time.time() > deadline - 5:
                    break
                u2, v2 = add
                if (u2, v2) == (u1, v1):
                    continue
                _add_edge(u2, v2)
                if _count_q3_with_edge(adj, add) == 0:
                    # Accept: net edge count same (removed 1, added 1)
                    # But maybe we can add MORE after this swap
                    improved = True
                    # Try greedy add after this swap
                    remaining = [e for e in all_edges if e not in edges]
                    random.shuffle(remaining)
                    for e2 in remaining[:15]:
                        if time.time() > deadline - 5:
                            break
                        if e2 in edges:
                            continue
                        _add_edge(*e2)
                        if _count_q3_with_edge(adj, e2) == 0:
                            improved = True
                        else:
                            _remove_edge(*e2)
                    if len(edges) > len(best_edges):
                        best_edges = set(edges)
                        best_count = len(edges)
                    break
                else:
                    _remove_edge(u2, v2)
            # If we didn't add anything, restore removed edge
            if (u1, v1) not in edges:
                _add_edge(u1, v1)
        return improved

    def _simulated_annealing():
        """SA on edge set: moves are add/remove/swap. Accept Q3 violations with temperature."""
        nonlocal best_edges, best_count
        T = 50.0
        T_min = 0.1
        cooling = 0.97
        current_q3 = _count_q3(adj)
        cur_edges = set(edges)

        while T > T_min and time.time() < deadline - 10:
            # Random move
            move = random.random()
            if move < 0.4:
                # Add a random edge
                cands = [e for e in all_edges if e not in edges]
                if not cands:
                    T *= cooling
                    continue
                e = random.choice(cands)
                _add_edge(*e)
                new_q3 = current_q3 + _count_q3_with_edge(adj, e)
                delta = (new_q3 - current_q3)  # positive = worse
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    current_q3 = new_q3
                    if new_q3 == 0 and len(edges) > len(best_edges):
                        best_edges = set(edges)
                        best_count = len(edges)
                else:
                    _remove_edge(*e)
            elif move < 0.7:
                # Remove a random edge
                if not edges:
                    T *= cooling
                    continue
                e = random.choice(list(edges))
                _remove_edge(*e)
                new_q3 = current_q3 - _count_q3_with_edge(adj, e)
                # Removing an edge can only decrease or keep Q3 count
                current_q3 = max(0, new_q3)
                if current_q3 == 0 and len(edges) > len(best_edges):
                    best_edges = set(edges)
                    best_count = len(edges)
            else:
                # Swap: remove one, add another
                if not edges:
                    T *= cooling
                    continue
                rem = random.choice(list(edges))
                cands = [e for e in all_edges if e not in edges and e != rem]
                if not cands:
                    T *= cooling
                    continue
                add = random.choice(cands)
                _remove_edge(*rem)
                _add_edge(*add)
                new_q3 = _count_q3(adj)
                delta = new_q3 - current_q3
                if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                    current_q3 = new_q3
                    if new_q3 == 0 and len(edges) > len(best_edges):
                        best_edges = set(edges)
                        best_count = len(edges)
                else:
                    _remove_edge(*add)
                    _add_edge(*rem)
            T *= cooling

    # --- Main search loop ---
    phase = 0
    while time.time() < deadline - 5:
        phase += 1
        if phase % 1 == 0:
            # Phase: greedy add
            cands = [e for e in all_edges if e not in edges]
            if not cands:
                break
            improved = _try_greedy_add(cands)
            if not improved and phase > 3:
                # Local swap search
                _local_swap_search()
            if not improved and phase > 6:
                # Simulated annealing
                _simulated_annealing()
                # After SA, do a greedy cleanup
                cands = [e for e in all_edges if e not in edges]
                if cands:
                    _try_greedy_add(cands)

    # Final: ensure best_edges is Q3-free (it should be, but verify)
    # Reconstruct adj from best_edges for verification
    adj_check = [0] * n
    for (u, v) in best_edges:
        adj_check[u] |= 1 << v
        adj_check[v] |= 1 << u
    if _count_q3(adj_check) != 0:
        # Fallback: use K3,13
        best_edges = set()
        for a in range(3):
            for b in range(3, n):
                best_edges.add((a, b))

    return [list(e) for e in best_edges]
