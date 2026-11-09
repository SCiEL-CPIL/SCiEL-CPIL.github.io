"""Rewritten program: local-search over 16-vertex graphs with 65 edges.

Start from K_{8,8} plus one internal edge, then try edge swaps that keep the
edge count at 65 and reduce the maximum triangle degree-sum. A deadline guard
returns the best result found before the wall-clock limit.
"""
import time
import random
import sys

_N = 16
_EDGE_TARGET = 65
# Safety margin: kill at 150s wall-clock, so stop at 130s to be safe.
_DEADLINE = 130.0


def _build_adjacency(edges):
    """Return (adj list of sets, degree list, edge set)."""
    adj = [set() for _ in range(_N)]
    es = set()
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
        es.add((min(u, v), max(u, v)))
    deg = [len(adj[i]) for i in range(_N)]
    return adj, deg, es


def _triangles(adj):
    """Yield (u, v, w) for each triangle (u<v<w)."""
    for u in range(_N):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    yield (u, v, w)


def _score(adj, deg):
    """Largest degree sum over triangles; -inf (use 0) if no triangles."""
    best = -1
    for (u, v, w) in _triangles(adj):
        s = deg[u] + deg[v] + deg[w]
        if s > best:
            best = s
    return best if best >= 0 else 0


def _seed_edges():
    # K_{8,8} plus one edge inside the first part.
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
    edges.append([0, 1])
    return edges


def _edge_swap(adj, deg, es, remove, add):
    """Apply one edge swap (remove edge, add edge) to copies; return new state
    or None if the swap is invalid (would create a duplicate or self-loop)."""
    ru, rv = remove
    au, av = add
    # Validate add.
    if au == av:
        return None
    if (min(au, av), max(au, av)) in es:
        return None
    # Validate remove exists.
    if (min(ru, rv), max(ru, rv)) not in es:
        return None
    # Build new adjacency/degree/edge-set by copying.
    new_adj = [set(s) for s in adj]
    new_es = set(es)
    new_deg = list(deg)
    # Remove.
    new_adj[ru].discard(rv)
    new_adj[rv].discard(ru)
    new_deg[ru] -= 1
    new_deg[rv] -= 1
    new_es.discard((min(ru, rv), max(ru, rv)))
    # Add.
    new_adj[au].add(av)
    new_adj[av].add(au)
    new_deg[au] += 1
    new_deg[av] += 1
    new_es.add((min(au, av), max(au, av)))
    return new_adj, new_deg, new_es


def _edges_from_es(es):
    return [[u, v] for (u, v) in sorted(es)]


def construct():
    start = time.time()
    rng = random.Random(12345)

    edges = _seed_edges()
    adj, deg, es = _build_adjacency(edges)
    best_score = _score(adj, deg)
    best_edges = list(edges)
    best_adj, best_deg, best_es = adj, deg, es

    # Precompute candidate add edges (non-edges) and all edges for swaps.
    def candidate_adds(eset):
        adds = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if (u, v) not in eset:
                    adds.append((u, v))
        return adds

    # Local search: repeatedly try to find a swap that improves the score.
    # We do a limited number of full passes to stay within the deadline.
    max_passes = 200
    for p in range(max_passes):
        if time.time() - start > _DEADLINE - 5.0:
            break
        adds = candidate_adds(es)
        rng.shuffle(adds)
        # Shuffle order of remove edges too.
        rem_list = list(es)
        rng.shuffle(rem_list)
        improved = False
        for add in adds:
            if time.time() - start > _DEADLINE - 5.0:
                break
            for remove in rem_list:
                # Skip trivially: removing an edge that shares no structure may
                # still be worth trying, but to save time we only try removes
                # that are "related" (share a vertex) with the add, plus a few
                # random global removes. This keeps the candidate set small.
                shares = (add[0] in remove) or (add[1] in remove)
                if not shares and rng.random() > 0.05:
                    continue
                new_state = _edge_swap(adj, deg, es, remove, add)
                if new_state is None:
                    continue
                nadj, ndeg, nes = new_state
                ns = _score(nadj, ndeg)
                if ns < best_score:
                    best_score = ns
                    best_edges = _edges_from_es(nes)
                    best_adj, best_deg, best_es = nadj, ndeg, nes
                    adj, deg, es = nadj, ndeg, nes
                    improved = True
                    break
            if improved:
                # Restart the outer add loop with the new state.
                break
        if not improved:
            # Try a few random global swaps (remove any edge, add any non-edge)
            # to escape local minima.
            adds2 = candidate_adds(es)
            rem2 = list(es)
            for _ in range(400):
                if time.time() - start > _DEADLINE - 5.0:
                    break
                a = adds2[rng.randrange(len(adds2))]
                r = rem2[rng.randrange(len(rem2))]
                new_state = _edge_swap(adj, deg, es, r, a)
                if new_state is None:
                    continue
                nadj, ndeg, nes = new_state
                ns = _score(nadj, ndeg)
                if ns < best_score:
                    best_score = ns
                    best_edges = _edges_from_es(nes)
                    best_adj, best_deg, best_es = nadj, ndeg, nes
                    adj, deg, es = nadj, ndeg, nes
                    improved = True
            if not improved:
                # No improvement found in this pass; try a small random restart
                # perturbation to escape the local minimum.
                # Perturb: do 3 random valid swaps (accept even if score worsens
                # temporarily) to move away from the local optimum.
                for _ in range(3):
                    adds3 = candidate_adds(es)
                    rem3 = list(es)
                    if not adds3 or not rem3:
                        break
                    a = adds3[rng.randrange(len(adds3))]
                    r = rem3[rng.randrange(len(rem3))]
                    new_state = _edge_swap(adj, deg, es, r, a)
                    if new_state is None:
                        continue
                    adj, deg, es = new_state
                # Re-run one improvement pass.
                adds4 = candidate_adds(es)
                rng.shuffle(adds4)
                rem4 = list(es)
                rng.shuffle(rem4)
                for add in adds4:
                    if time.time() - start > _DEADLINE - 5.0:
                        break
                    for remove in rem4:
                        shares = (add[0] in remove) or (add[1] in remove)
                        if not shares and rng.random() > 0.1:
                            continue
                        new_state = _edge_swap(adj, deg, es, remove, add)
                        if new_state is None:
                            continue
                        nadj, ndeg, nes = new_state
                        ns = _score(nadj, ndeg)
                        if ns < best_score:
                            best_score = ns
                            best_edges = _edges_from_es(nes)
                            best_adj, best_deg, best_es = nadj, ndeg, nes
                            adj, deg, es = nadj, ndeg, nes
                            break

    # Final validation: ensure exactly 65 edges and simple graph.
    final_adj, final_deg, final_es = _build_adjacency(best_edges)
    if len(final_es) != _EDGE_TARGET:
        # Fallback to seed if something went wrong.
        best_edges = _seed_edges()
    return {'n': _N, 'edges': best_edges}


if __name__ == '__main__':
    res = construct()
    adj, deg, es = _build_adjacency(res['edges'])
    print('edges:', len(es), 'score:', _score(adj, deg))
