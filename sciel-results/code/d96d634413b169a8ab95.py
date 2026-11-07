import numpy as np
import time

def _build_signed_adj(n, m, edges, tree_indices, chord_indices, chord_mask):
    """Build 2m x 2m signed adjacency matrix. Tree edges +1, chords per mask."""
    A = np.zeros((n, n), dtype=float)
    # We need edge->(u,v) mapping handled outside; here we fill from edges list
    return A

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = graph['n']
    m = n // 2
    edges = graph['edges']

    # Map edge tuple to index
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i

    # Spanning tree: top path, bottom path, one matching edge
    tree_edges = []
    for i in range(m - 1):
        tree_edges.append((i, i + 1))
        tree_edges.append((m + i, m + i + 1))
    tree_edges.append((0, m))

    # Chords: top closure, bottom closure, remaining matching
    chord_edges = []
    chord_edges.append((m - 1, 0))
    chord_edges.append((2 * m - 1, m))
    for i in range(1, m):
        chord_edges.append((i, m + i))

    all_logical_edges = tree_edges + chord_edges
    assert len(all_logical_edges) == len(edges), f"Edge count mismatch: {len(all_logical_edges)} vs {len(edges)}"

    tree_indices = [edge_map[e] for e in tree_edges]
    chord_indices = [edge_map[e] for e in chord_edges]
    num_chords = len(chord_indices)

    # Precompute the base unsigned adjacency structure as a list of (u, v) per edge index
    # and a list of which edges are tree vs chord.
    # Build a template: for each edge index, (u, v).
    edge_uv = edges  # list of (u,v)

    # Precompute: for each edge, its (u,v). We'll fill A symmetrically.
    # Tree edges always +1. Chord edges vary.

    # Precompute the "fixed" part of A (tree edges = +1)
    A_base = np.zeros((n, n), dtype=float)
    for idx in tree_indices:
        u, v = edge_uv[idx]
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0

    # For chords, we'll add/subtract based on mask.
    # Precompute chord (u,v) pairs.
    chord_uv = [edge_uv[idx] for idx in chord_indices]

    # Fast connectivity check on the 2-lift.
    # Build adjacency of the lift once per candidate? That's expensive.
    # Instead: the lift has 2n vertices. Vertex (v, s) connects to (w, s * sign(u,v)) for each edge (u,v) with u=v... 
    # Actually: for edge (u,v) with sign sigma, in the lift: (u,0)-(v,0) if sigma=+1 else (u,0)-(v,1); (u,1)-(v,1) if sigma=+1 else (u,1)-(v,0).
    # So the lift adjacency: from (u, s) to (v, s) if sigma=+1, else (v, 1-s).
    #
    # Connectivity: BFS from (0,0).

    def lift_connected(signs):
        # signs: array of length len(edges), each +-1
        # Build adjacency on the fly with BFS
        n_lift = 2 * n
        visited = np.zeros(n_lift, dtype=bool)
        from collections import deque
        q = deque()
        start = 0  # vertex (0, 0)
        q.append(start)
        visited[start] = True
        count = 1
        while q:
            node = q.popleft()
            v = node // 2
            s = node % 2
            for idx, (u, w) in enumerate(edge_uv):
                if u == v:
                    other = w
                elif w == v:
                    other = u
                else:
                    continue
                sigma = signs[idx]
                if sigma == 1:
                    ns = s
                else:
                    ns = 1 - s
                nnode = other * 2 + ns
                if not visited[nnode]:
                    visited[nnode] = True
                    count += 1
                    q.append(nnode)
        return count == n_lift

    best_radius = float('inf')
    best_signs = None

    # Enumerate all 2^(m+1) switching classes.
    # For speed, precompute the full signs array and only vary chord signs.
    base_signs = np.ones(len(edges), dtype=np.int8)
    # tree edges already +1

    total_masks = 1 << num_chords
    for mask in range(total_masks):
        if time.time() > deadline:
            break

        signs = base_signs.copy()
        for i, idx in enumerate(chord_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
            # else +1 (already)

        # Fast validity: connectivity
        if not lift_connected(signs):
            continue

        # Build signed adjacency matrix
        A = A_base.copy()
        for i, idx in enumerate(chord_indices):
            u, v = chord_uv[i]
            sgn = -1.0 if ((mask >> i) & 1) else 1.0
            A[u, v] = sgn
            A[v, u] = sgn

        # Spectral radius
        try:
            ev = np.linalg.eigvalsh(A)
            radius = float(np.max(np.abs(ev)))
        except Exception:
            continue

        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    if best_signs is None:
        # Fallback
        rng = np.random.default_rng(3)
        best_signs = rng.choice([-1, 1], size=len(edges))

    return best_signs
