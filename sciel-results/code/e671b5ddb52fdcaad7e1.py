"""K2,3-free graph on 20 vertices - spectral init + local search."""
import time
import random
import numpy as np

N = 20
DEADLINE = 130.0  # safety margin under 150s wall-clock

def _count_common(a, b, adj):
    """Number of common neighbours of a and b."""
    return int(np.sum(np.asarray(adj[a]) & np.asarray(adj[b])))

def _is_k23_free(adj):
    """Check no two vertices share 3+ common neighbours."""
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            c = int(np.sum(np.asarray(adj[i]) & np.asarray(adj[j])))
            if c >= 3:
                return False
    return True

def _edge_count(adj):
    return int(np.sum(adj)) // 2

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _spectral_init(rng):
    """Build a random symmetric matrix, take top eigenvector, threshold."""
    # Random symmetric matrix
    A = rng.standard_normal((N, N))
    A = (A + A.T) / 2.0
    # Zero diagonal
    np.fill_diagonal(A, 0.0)
    # Top eigenvector
    w, V = np.linalg.eigh(A)
    v = V[:, -1]
    # Threshold at mean
    x = (v > v.mean()).astype(np.int64)
    # Build adjacency from x (symmetric, no self-loops)
    adj = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            if x[i] == 1 and x[j] == 1:
                adj[i, j] = adj[j, i] = 1
    return adj

def _random_init(rng, density=0.6):
    """Random graph with given edge density."""
    adj = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < density:
                adj[i, j] = adj[j, i] = 1
    return adj

def _local_search(adj, rng, time_limit):
    """Iterative improvement: add edges, remove violations, swap edges."""
    n = N
    start = time.time()
    best_edges = _edge_count(adj)
    no_improve = 0
    max_no_improve = 2000

    while time.time() - start < time_limit:
        improved = False

        # Phase 1: try to add edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    continue
                # Check if adding (i,j) creates a K2,3
                ok = True
                # For every common neighbour k of i and j, adding (i,j)
                # means i and j now share k as common neighbour.
                # We need: for every pair (a,b) in the new graph, common(a,b) <= 2
                # Adding edge (i,j): 
                #   - i and j now share all common neighbours of i and j
                #   - For any vertex k adjacent to both i and j: common(i,j) increases
                #   - For any vertex k adjacent to i: common(j,k) may increase if j~k... 
                # Actually: adding edge (i,j) affects common-neighbour counts for:
                #   - pair (i,j): gains common neighbours = N(i) ∩ N(j)
                #   - pair (i,k) for k in N(j): gains j as common neighbour (if i~j now)
                #   - pair (j,k) for k in N(i): gains i as common neighbour
                #   - pair (i,j): the new pair
                # Let's just check all pairs after tentative addition
                adj[i, j] = adj[j, i] = 1
                valid = True
                for a in range(n):
                    for b in range(a + 1, n):
                        if _count_common(a, b, adj) >= 3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    improved = True
                    best_edges = _edge_count(adj)
                else:
                    adj[i, j] = adj[j, i] = 0
                if time.time() - start > time_limit:
                    return adj

        # Phase 2: try swaps (remove one edge, add another)
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    continue
                # Try removing (i,j) and adding (k,l)
                adj[i, j] = adj[j, i] = 0
                for k in range(n):
                    for l in range(k + 1, n):
                        if adj[k, l] or (k == i and l == j) or (k == j and l == i):
                            continue
                        # Check validity
                        adj[k, l] = adj[l, k] = 1
                        valid = True
                        for a in range(n):
                            for b in range(a + 1, n):
                                if _count_common(a, b, adj) >= 3:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            improved = True
                            best_edges = _edge_count(adj)
                        else:
                            adj[k, l] = adj[l, k] = 0
                        if time.time() - start > time_limit:
                            adj[i, j] = adj[j, i] = 1
                            return adj
                adj[i, j] = adj[j, i] = 1
                if time.time() - start > time_limit:
                    return adj

        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0

    return adj

def _simulated_annealing(adj, rng, time_limit):
    """SA with add/remove/swap moves."""
    n = N
    start = time.time()
    temp = 1.0
    cooling = 0.999
    best_adj = adj.copy()
    best_edges = _edge_count(adj)

    while time.time() - start < time_limit and temp > 0.01:
        # Pick a random move
        move_type = rng.random()
        if move_type < 0.4:
            # Add an edge
            i, j = rng.sample(range(n), 2)
            if adj[i, j]:
                continue
            adj[i, j] = adj[j, i] = 1
            delta = 1
            valid = _is_k23_free(adj)
            if not valid:
                adj[i, j] = adj[j, i] = 0
                if rng.random() > np.exp(-1.0 / temp):
                    continue
                # Accept invalid? No, we must maintain validity.
                # Only accept if valid.
                continue
        elif move_type < 0.7:
            # Remove an edge
            i, j = rng.sample(range(n), 2)
            if not adj[i, j]:
                continue
            adj[i, j] = adj[j, i] = 0
            delta = -1
            # Always valid after removal
        else:
            # Swap
            i1, j1 = rng.sample(range(n), 2)
            i2, j2 = rng.sample(range(n), 2)
            if not adj[i1, j1] or adj[i2, j2]:
                continue
            adj[i1, j1] = adj[j1, i1] = 0
            adj[i2, j2] = adj[j2, i2] = 1
            delta = 0
            if not _is_k23_free(adj):
                adj[i1, j1] = adj[j1, i1] = 1
                adj[i2, j2] = adj[j2, i2] = 0
                continue

        new_edges = _edge_count(adj)
        if new_edges > best_edges:
            best_adj = adj.copy()
            best_edges = new_edges

        # Accept/reject
        if delta >= 0:
            pass  # always accept
        else:
            if rng.random() > np.exp(delta / temp):
                # revert
                if move_type < 0.4:
                    adj[i, j] = adj[j, i] = 0
                elif move_type < 0.7:
                    adj[i, j] = adj[j, i] = 1
                else:
                    adj[i1, j1] = adj[j1, i1] = 1
                    adj[i2, j2] = adj[j2, i2] = 0

        temp *= cooling

    return best_adj

def construct():
    global DEADLINE
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    best_adj = None
    best_edges = 0

    # Try multiple initializations
    inits = []
    # Spectral init
    inits.append(_spectral_init(np_rng))
    # Random inits at various densities
    for d in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        inits.append(_random_init(rng, d))
    # Dense random
    inits.append(_random_init(rng, 0.75))

    # Also try a structured init: complement of a sparse graph
    # Start with complete graph and remove edges to satisfy K2,3-free
    comp_adj = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(comp_adj, 0)
    inits.append(comp_adj)

    for init in inits:
        if time.time() > deadline - 5:
            break
        # Fix up: remove edges to make it K2,3-free
        adj = init.copy()
        # Greedy fix: remove edges that cause violations
        changed = True
        while changed:
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    if adj[i, j] and _count_common(i, j, adj) >= 3:
                        adj[i, j] = adj[j, i] = 0
                        changed = True
        if not _is_k23_free(adj):
            # More aggressive fix
            adj = np.zeros((N, N), dtype=np.int64)
            for i in range(N):
                for j in range(i + 1, N):
                    if init[i, j]:
                        adj[i, j] = adj[j, i] = 1
                        if not _is_k23_free(adj):
                            adj[i, j] = adj[j, i] = 0

        e = _edge_count(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj.copy()

    # Local search from best
    if best_adj is not None and time.time() < deadline - 10:
        adj = _local_search(best_adj, rng, time_limit=min(30, deadline - time.time() - 2))
        e = _edge_count(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj

    # SA from best
    if best_adj is not None and time.time() < deadline - 10:
        adj = _simulated_annealing(best_adj, rng, time_limit=min(40, deadline - time.time() - 2))
        e = _edge_count(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj

    # More restarts with local search if time remains
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        d = 0.5 + 0.3 * rng.random()
        adj = _random_init(rng, d)
        # Greedy fix
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j]:
                    adj[i, j] = adj[j, i] = 1
                    if not _is_k23_free(adj):
                        adj[i, j] = adj[j, i] = 0
        # Local search
        remaining = deadline - time.time() - 5
        if remaining > 3:
            adj = _local_search(adj, rng, time_limit=min(5, remaining))
        e = _edge_count(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj.copy()

    if best_adj is None:
        # Fallback: 5 disjoint K4s
        edges = []
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.append([4 * k + a, 4 * k + b])
        return edges

    return _adj_to_edges(best_adj)
