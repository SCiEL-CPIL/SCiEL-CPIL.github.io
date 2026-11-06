"""Triangle-free 4-chromatic graphs on 16 vertices via local search with swaps.

Design: local search with edge-add and edge-swap moves. Every accepted move must
preserve BOTH triangle-freeness and non-3-colorability (exact 3-coloring check).
Starts from the Grötzsch core, greedily completes, then refines.
"""
# EVOLVE-BLOCK-START
import random
import time

N = 16

def _build_grotzsch_edges():
    """Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[10, 5 + u] for u in range(5)]
    return edges

def _make_bitmask_adj(edge_list, n=N):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid_triangle_free(adj):
    """Verify that adjacency bitmasks define a simple triangle-free graph."""
    n = len(adj)
    for u in range(n):
        if adj[u] & (1 << u):
            return False
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj, deadline=None):
    """
    Check if the graph defined by adj bitmasks is 3-colorable.
    Returns True if 3-colorable, False otherwise.
    Uses backtracking with vertex ordering by degree (descending) for speed.
    """
    n = len(adj)
    # Order vertices by descending degree to fail fast
    order = list(range(n))
    order.sort(key=lambda v: adj[v].bit_count(), reverse=True)
    # Map: position in order -> vertex
    colors = [-1] * n  # indexed by vertex
    # For quick "available colors" we track per-vertex forbidden mask
    # We'll do recursive backtracking over the order.

    # Precompute neighbor lists
    neigh = []
    for v in range(n):
        lst = []
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            lst.append(u)
        neigh.append(lst)

    # pos_in_order for quick check
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    # We assign colors in 'order'. To check if a color is available for v,
    # we look at neighbors already colored (pos < current index).
    # We'll maintain colors array and check on the fly.

    # To make it fast, use a "forward checking" style: maintain for each
    # uncolored vertex a bitmask of forbidden colors.
    # But simpler: recursive with pruning.

    import sys
    sys.setrecursionlimit(10000)

    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return None  # signal timeout
        if idx == n:
            return True
        v = order[idx]
        # forbidden colors from colored neighbors
        forb = 0
        for u in neigh[v]:
            if colors[u] != -1:
                forb |= (1 << colors[u])
        avail = (~forb) & 0b111
        if avail == 0:
            return False
        # Try colors in a heuristic order
        while avail:
            lb = avail & (-avail)
            c = lb.bit_length() - 1
            avail ^= lb
            colors[v] = c
            res = backtrack(idx + 1)
            if res is not None:
                if res:
                    return True
                else:
                    # timeout
                    colors[v] = -1
                    return None
            colors[v] = -1
        return False

    result = backtrack(0)
    if result is None:
        # Timeout: conservatively assume NOT 3-colorable is wrong;
        # we must be safe. Return True (assume colorable) to avoid
        # accepting a possibly-3-colorable graph as 4-chromatic.
        return True
    return result

def _local_search(seed, deadline):
    n = N
    rng = random.Random(seed)

    # Initialize with Grotzsch core
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    non_edges = [e for e in all_edges if e not in edge_set]
    rng.shuffle(non_edges)

    best_count = len(edge_set)
    best_adj = adj[:]
    best_edge_set = set(edge_set)

    # Track consecutive failed adds for swap triggering
    fail_count = 0

    while time.time() < deadline:
        # 1. Try to add a random non-edge (that is triangle-free AND keeps 4-chromatic)
        added = False
        if non_edges:
            # Try a few random non-edges
            tried = 0
            while non_edges and tried < 8:
                idx = rng.randrange(len(non_edges))
                u, v = non_edges[idx]
                # Triangle-free check
                if adj[u] & adj[v] == 0:
                    # Temporarily add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    if not _is_3_colorable(adj, deadline):
                        # Valid: triangle-free and still 4-chromatic
                        added = True
                        if len(edge_set) > best_count:
                            best_count = len(edge_set)
                            best_adj = adj[:]
                            best_edge_set = set(edge_set)
                        non_edges.pop(idx)
                        fail_count = 0
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove((u, v))
                tried += 1
            if not added:
                fail_count += 1

        # 2. If no edge added (or after some adds), try a swap
        if not added:
            if not edge_set:
                break
            # Pick a random edge to remove
            e_out = rng.choice(tuple(edge_set))
            u_out, v_out = e_out

            # Remove it
            adj[u_out] &= ~(1 << v_out)
            adj[v_out] &= ~(1 << u_out)
            edge_set.remove(e_out)

            # Now try to add a random non-edge (including possibly e_out)
            # Build candidate pool: all non-edges in current state
            # We'll just sample random pairs
            swap_success = False
            attempts = 0
            while not swap_success and attempts < 40 and time.time() < deadline:
                attempts += 1
                # Pick random pair
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v:
                    continue
                e_in = (min(u, v), max(u, v))
                if e_in in edge_set:
                    continue
                # Triangle-free check
                if adj[u] & adj[v] == 0:
                    # Temporarily add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add(e_in)
                    if not _is_3_colorable(adj, deadline):
                        # Valid swap
                        swap_success = True
                        if len(edge_set) > best_count:
                            best_count = len(edge_set)
                            best_adj = adj[:]
                            best_edge_set = set(edge_set)
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edge_set.remove(e_in)

            # If no swap succeeded, revert the removal of e_out
            if not swap_success:
                adj[u_out] |= (1 << v_out)
                adj[v_out] |= (1 << u_out)
                edge_set.add(e_out)
                # Update non_edges list: e_out is back as an edge, so remove from non_edges if present
                if e_out in non_edges:
                    non_edges.remove(e_out)
            else:
                # e_out is now a non-edge
                if e_out not in non_edges:
                    non_edges.append(e_out)
                # e_in was removed from consideration; if it was in non_edges, remove it
                if e_in in non_edges:
                    non_edges.remove(e_in)

        # Occasionally reshuffle non_edges to escape local ordering
        if fail_count > 30:
            rng.shuffle(non_edges)
            fail_count = 0

    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])

    return result_edges, len(result_edges)

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges. Uses local search with swaps, where every
    move preserves triangle-freeness and non-3-colorability.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit

    best_edges = []
    best_count = 0

    # First, get a good starting point via greedy completion of Grotzsch core
    # (greedy only adds triangle-free edges; we then verify 4-chromaticity)
    grotzsch = _build_grotzsch_edges()
    # Greedy complete with a few random orderings, keeping only 4-chromatic results
    for gseed in range(6):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(1000 + gseed)
        adj = _make_bitmask_adj(grotzsch, N)
        edge_set = set((min(u, v), max(u, v)) for u, v in grotzsch)
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        remaining = [e for e in all_edges if e not in edge_set]
        rng.shuffle(remaining)
        for u, v in remaining:
            if time.time() > deadline - 5.0:
                break
            if adj[u] & adj[v] == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        # Check 4-chromaticity
        if not _is_3_colorable(adj, deadline):
            edges = []
            for u in range(N):
                for v in range(u + 1, N):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Now run local search from multiple seeds, each starting from the best found
    seed = 0
    while time.time() < deadline - 10.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        if best_count >= 55:
            break

    # Final validation
    if best_edges:
        if not _is_valid_triangle_free(_make_bitmask_adj(best_edges)):
            best_edges = list(grotzsch)
        # Also verify 4-chromaticity of the final answer
        elif _is_3_colorable(_make_bitmask_adj(best_edges)):
            # Fallback to Grotzsch (known 4-chromatic)
            best_edges = list(grotzsch)
    else:
        best_edges = list(grotzsch)

    return best_edges
# EVOLVE-BLOCK-END
