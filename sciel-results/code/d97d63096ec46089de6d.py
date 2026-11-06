"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 130.0  # safety margin before 150s kill
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_ok():
    return _elapsed() < _DEADLINE


def _generate_graph(n, p, seed):
    """Generate graph matching the verifier's random_graph(n, p, seed)."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _bron_kerbosch(adj):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    cliques = []
    n = len(adj)

    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return

        # Choose pivot to minimize branching
        pivot = None
        max_count = -1
        for v in P:
            c = len(P & adj[v])
            if c > max_count:
                max_count = c
                pivot = v
        if pivot is None:
            for v in X:
                c = len(P & adj[v])
                if c > max_count:
                    max_count = c
                    pivot = v

        if pivot is not None:
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)

        for v in candidates:
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)

    bk(frozenset(), set(range(n)), frozenset())
    return cliques


def _greedy_transversal(cliques):
    """Greedy set cover: pick vertex covering most uncovered cliques."""
    if not cliques:
        return []

    # Map each vertex to the set of cliques it covers
    vertex_to_cliques = {}
    for i, c in enumerate(cliques):
        for v in c:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = set()
            vertex_to_cliques[v].add(i)

    uncovered = set(range(len(cliques)))
    selected = []

    while uncovered and _time_ok():
        best_v = None
        best_count = -1
        for v, covered in vertex_to_cliques.items():
            cnt = len(covered & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v is None or best_count <= 0:
            break
        selected.append(best_v)
        uncovered -= vertex_to_cliques[best_v]

    return sorted(selected)


def _local_search_improve(cliques, selected):
    """Try to reduce the transversal by removing vertices and re-covering."""
    selected_set = set(selected)

    # Build clique coverage info
    def is_valid(sel):
        sel = set(sel)
        for c in cliques:
            if not (c & sel):
                return False
        return True

    improved = True
    while improved and _time_ok():
        improved = False
        # Try removing each vertex
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if is_valid(candidate):
                selected = candidate
                improved = True
                break

    # Try swapping: remove one, add another
    if _time_ok():
        all_vertices = set()
        for c in cliques:
            all_vertices.update(c)

        improved = True
        while improved and _time_ok():
            improved = False
            for i in range(len(selected)):
                for v in all_vertices:
                    if v in selected_set:
                        continue
                    candidate = selected[:i] + selected[i+1:] + [v]
                    if len(candidate) < len(selected):
                        pass  # same size, no improvement
                    # Only try if same size but different (for potential future improvements)
                    # Actually, we want to keep size the same or smaller
                    if is_valid(candidate) and len(candidate) <= len(selected):
                        # Check if it's actually different
                        if set(candidate) != selected_set:
                            selected = sorted(candidate)
                            selected_set = set(selected)
                            improved = True
                            break
                if improved:
                    break

    return sorted(selected)


def _solve_instance(n, p, seed):
    """Solve one instance: find minimum vertex transversal of all maximal cliques >= 2."""
    adj = _generate_graph(n, p, seed)
    cliques = _bron_kerbosch(adj)

    if not cliques:
        return []

    # Greedy initial solution
    solution = _greedy_transversal(cliques)

    # Local search improvement
    solution = _local_search_improve(cliques, solution)

    return solution


def construct():
    global _start_time
    _start_time = time.time()

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if not _time_ok():
            # Fallback: return all non-isolated vertices (valid but large)
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            out.append(sorted(touched))
        else:
            out.append(_solve_instance(n, p, seed))

    return out
# EVOLVE-BLOCK-END
