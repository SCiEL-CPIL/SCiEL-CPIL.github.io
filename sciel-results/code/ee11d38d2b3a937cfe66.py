"""Spectral graph lifts: sign prism-graph edges to minimize mean spectral radius."""
import time
import numpy as np

# ---- deadline guard: evaluator kills after 90s wall-clock ----
_T_START = time.time()
_DEADLINE = 78.0  # safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T_START)


def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = list(graph['edges'])
    E = len(edges)
    n = 2 * m

    # Build base signed-adjacency structure.
    # For a two-lift, the lift adjacency matrix is block-structured:
    #   A_lift = [[A,  B], [B, A]]  where A = sum_{s=+1} e, B = sum_{s=-1} e
    # Eigenvalues of the lift are eigenvalues of (A+B) and (A-B).
    # A+B is the (unsigned) base adjacency (independent of signs) -> constant eigenvalues.
    # A-B is the signed base adjacency -> depends on signs.
    # Spectral radius of lift = max( rho(A+B), rho(A-B) ).
    # So we minimize rho(A-B) = max |eig| of signed adjacency.

    # Build unsigned adjacency (constant part) and per-edge signed adjacency contribution.
    # Represent each edge as (u,v,sign) -> contributes sign to A-B at (u,v),(v,u).
    # A-B = sum_e s_e * (E_uv + E_vu).

    # Precompute edge index arrays.
    u_idx = np.array([e[0] for e in edges], dtype=int)
    v_idx = np.array([e[1] for e in edges], dtype=int)

    # Unsigned adjacency (constant) -> its spectral radius is fixed.
    A_unsigned = np.zeros((n, n), dtype=float)
    A_unsigned[u_idx, v_idx] = 1.0
    A_unsigned[v_idx, u_idx] = 1.0
    # spectral radius of unsigned part (fixed lower bound)
    # For 3-regular graph, rho(A_unsigned) = 3 (regular graph: top eigenvalue = degree).
    # So the lift spectral radius = max(3, rho(signed)). We minimize rho(signed).

    def signed_radius(signs):
        # signs: array of +1/-1 length E
        S = np.zeros((n, n), dtype=float)
        S[u_idx, v_idx] = signs
        S[v_idx, u_idx] = signs
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(signs):
        # The lift is connected iff the signed base graph is "balanced" in the sense
        # that the two-lift (Kronecker with sign) is connected. For a two-lift,
        # connectivity holds iff there exists at least one cycle with product of signs = -1
        # (i.e., the graph is not balanced). Actually: the two-lift of a connected graph is
        # connected iff the graph is NOT balanced (i.e., there is an odd-sign cycle).
        # Balanced means every cycle has product +1, equivalent to: there exists a vertex
        # labeling x_v in {+1,-1} such that for every edge (u,v), s_e = x_u * x_v.
        # So lift connected <=> NOT balanced.
        # Check balance: BFS assigning x_v, x_u * s_e must be consistent.
        # If consistent (balanced) -> lift is TWO components -> invalid.
        # If inconsistent -> connected -> valid.
        # Fast O(n+E) BFS:
        x = np.full(n, 0, dtype=int)  # 0 unassigned, else +1/-1
        from collections import deque
        # Build adjacency list of (neighbor, edge_sign)
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            s = int(signs[i])
            adj[u].append((v, s))
            adj[v].append((u, s))
        for start in range(n):
            if x[start] == 0:
                x[start] = 1
                q = deque([start])
                while q:
                    u = q.popleft()
                    for (v, s) in adj[u]:
                        expected = x[u] * s
                        if x[v] == 0:
                            x[v] = expected
                            q.append(v)
                        elif x[v] != expected:
                            return True  # inconsistent -> not balanced -> connected
        # Fully balanced -> lift disconnected -> invalid
        return False

    def valid(signs):
        return is_connected(signs)

    # Local search: flip one edge, accept if improves signed_radius (and stays valid).
    def local_search(signs, time_limit_frac):
        signs = signs.copy()
        if not valid(signs):
            return None
        cur = signed_radius(signs)
        deadline = _remaining() * time_limit_frac
        improved = True
        while improved and _remaining() > 0.5:
            improved = False
            for i in range(E):
                if _remaining() < 0.5:
                    break
                old = signs[i]
                signs[i] = -old
                if valid(signs):
                    r = signed_radius(signs)
                    if r < cur - 1e-12:
                        cur = r
                        improved = True
                    else:
                        signs[i] = old
                else:
                    signs[i] = old
        return signs, cur

    best_signs = None
    best_val = float('inf')

    def consider(signs):
        nonlocal best_signs, best_val
        if valid(signs):
            r = signed_radius(signs)
            if r < best_val - 1e-12:
                best_val = r
                best_signs = signs.copy()
                return True
        return False

    # Multi-start: random inits + local search
    rng = np.random.default_rng(3)
    # First do the seed-like random sampling to get a valid baseline
    for _ in range(2000):
        if _remaining() < 1.0:
            break
        signs = rng.choice([-1, 1], size=E)
        if consider(signs):
            pass

    # Now local search from several random starts
    starts = 40
    for s in range(starts):
        if _remaining() < 3.0:
            break
        signs = rng.choice([-1, 1], size=E)
        # ensure validity by flipping edges until valid (or just use random valid)
        # Just run local search which checks validity internally
        res = local_search(signs, time_limit_frac=1.0)
        if res is not None:
            ss, cv = res
            if cv < best_val - 1e-12:
                best_val = cv
                best_signs = ss.copy()

    # Polish: from best, do a finer local search
    if best_signs is not None:
        res = local_search(best_signs, time_limit_frac=1.0)
        if res is not None:
            ss, cv = res
            if cv < best_val - 1e-12:
                best_val = cv
                best_signs = ss.copy()

    if best_signs is None:
        # fallback: all +1 (balanced -> invalid) so try random valid
        for _ in range(20000):
            signs = rng.choice([-1, 1], size=E)
            if valid(signs):
                best_signs = signs
                break
        if best_signs is None:
            # desperate: force a -1 to break balance
            best_signs = np.ones(E, dtype=int)
            best_signs[0] = -1

    return best_signs
