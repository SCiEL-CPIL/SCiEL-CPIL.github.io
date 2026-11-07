# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    start = time.time()
    deadline = start + 50.0  # 50s safety margin under 60s limit

    n = graph['n']
    edges = graph['edges']
    E = len(edges)
    m = n // 2

    # Identify edge types: cycle edges (within ring) vs matching edges (i <-> i+m)
    # Cycle edges: both endpoints in [0, m) or both in [m, 2m)
    # Matching edges: one endpoint in [0, m), other in [m, 2m)
    edge_types = []  # 0 = cycle, 1 = matching
    for (u, v) in edges:
        if (u < m and v < m) or (u >= m and v >= m):
            edge_types.append(0)
        else:
            edge_types.append(1)

    # Build adjacency: for each vertex, list of (neighbor, edge_index)
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def spectral_radius(signs):
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        try:
            evals = np.linalg.eigvalsh(A)
        except np.linalg.LinAlgError:
            return 1e10
        return float(np.max(np.abs(evals)))

    def is_connected(signs):
        # Check lift connectivity via BFS on the 2n-vertex lift
        # Vertex (v, side): edge (u,v) with sign s connects (u, side) to (v, side) if s=+1,
        # or (v, 1-side) if s=-1
        visited = np.zeros((n, 2), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            v, side = stack.pop()
            for (nb, eidx) in adj[v]:
                s = signs[eidx]
                new_side = side if s == 1 else 1 - side
                if not visited[nb, new_side]:
                    visited[nb, new_side] = True
                    stack.append((nb, new_side))
        return bool(np.all(visited))

    def evaluate(signs):
        """Return spectral radius if valid, else inf."""
        if not is_connected(signs):
            return float('inf')
        return spectral_radius(signs)

    # --- Step 1: Analytic seed ---
    # For each ring, compute cycle spectral radius
    # Ring 0: vertices 0..m-1, Ring 1: vertices m..2m-1
    # Cycle edges in ring 0: edges where both endpoints in [0, m)
    # We need to find the cycle structure. For a prism, each ring is an m-cycle.
    
    # Build cycle adjacency per ring
    def get_ring_cycle_edges(ring_offset):
        """Return list of (u, v, edge_idx) for cycle edges in the ring."""
        result = []
        for idx, (u, v) in enumerate(edges):
            if edge_types[idx] == 0:
                # Both in same ring
                if u >= ring_offset and v >= ring_offset:
                    result.append((u, v, idx))
        return result

    # For a cycle graph, the spectral radius of the signed cycle is |sum of signs * exp(i*k*2pi/m)|
    # Max over k of |prod of signs along the cycle path|... actually for a cycle:
    # The eigenvalues of a signed cycle are 2*|prod(signs)|^(1/m) * cos(2*pi*k/m + phase)
    # Simpler: just compute numerically for the cycle matrix

    def cycle_spectral_radius(cycle_edges, signs):
        """Compute spectral radius of the cycle subgraph."""
        # Get unique vertices
        verts = set()
        for (u, v, _) in cycle_edges:
            verts.add(u)
            verts.add(v)
        verts = sorted(verts)
        if not verts:
            return 0.0
        vmap = {v: i for i, v in enumerate(verts)}
        nc = len(verts)
        C = np.zeros((nc, nc), dtype=float)
        for (u, v, eidx) in cycle_edges:
            s = signs[eidx]
            C[vmap[u], vmap[v]] = s
            C[vmap[v], vmap[u]] = s
        try:
            evals = np.linalg.eigvalsh(C)
        except np.linalg.LinAlgError:
            return 1e10
        return float(np.max(np.abs(evals)))

    # Find matching edges
    matching_edges = [(idx, edges[idx]) for idx in range(E) if edge_types[idx] == 1]
    cycle_edges_ring0 = get_ring_cycle_edges(0)
    cycle_edges_ring1 = get_ring_cycle_edges(m)

    # Initial signs: all +1
    best_signs = np.ones(E, dtype=int)
    best_val = evaluate(best_signs)

    # Analytic: set matching signs to cancel the ring with larger spectral radius
    # For each matching edge i <-> i+m, sign affects diagonal shift
    # Strategy: try both all-matching +1 and all-matching -1, pick better
    for match_sign in [1, -1]:
        cand = np.ones(E, dtype=int)
        for idx, (u, v) in matching_edges:
            cand[idx] = match_sign
        val = evaluate(cand)
        if val < best_val:
            best_val = val
            best_signs = cand.copy()

    # --- Step 2: Hill climbing ---
    def hill_climb(signs, max_iters=200000):
        cur = signs.copy()
        cur_val = evaluate(cur)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            for i in range(E):
                if time.time() > deadline - 5:
                    break
                trial = cur.copy()
                trial[i] = -trial[i]
                trial_val = evaluate(trial)
                if trial_val < cur_val - 1e-12:
                    cur = trial
                    cur_val = trial_val
                    improved = True
        return cur, cur_val

    best_signs, best_val = hill_climb(best_signs)

    # --- Step 3: Exhaustive matching sign search ---
    # Enumerate all 2^m matching sign combinations
    # For each, optimize cycle signs via hill climbing
    num_matching = len(matching_edges)
    if num_matching <= 12:
        for mask in range(1 << num_matching):
            if time.time() > deadline - 3:
                break
            cand = np.ones(E, dtype=int)
            for j, (idx, (u, v)) in enumerate(matching_edges):
                cand[idx] = 1 if (mask >> j) & 1 else -1
            # Hill climb from this seed
            hc_signs, hc_val = hill_climb(cand, max_iters=50000)
            if hc_val < best_val - 1e-12:
                best_val = hc_val
                best_signs = hc_signs.copy()

    # --- Step 4: Random restarts with hill climbing ---
    rng = np.random.default_rng(42)
    for _ in range(2000):
        if time.time() > deadline - 3:
            break
        cand = rng.choice([-1, 1], size=E).astype(int)
        hc_signs, hc_val = hill_climb(cand, max_iters=20000)
        if hc_val < best_val - 1e-12:
            best_val = hc_val
            best_signs = hc_signs.copy()

    # Final check
    if best_val == float('inf'):
        # Fallback: all +1
        best_signs = np.ones(E, dtype=int)

    return best_signs
# EVOLVE-BLOCK-END
