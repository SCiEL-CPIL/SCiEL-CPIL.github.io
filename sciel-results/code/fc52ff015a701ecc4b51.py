"""Optimized graph construction for min triangle degree-sum."""
import time
import random
import numpy as np

N = 16
TARGET_EDGES = 65  # floor(16^2/4) + 1


def _score(adj):
    """Return (max_degree_sum, n_triangles)."""
    n = len(adj)
    # Precompute common-neighbour bitmasks
    common = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                common[i | (j << 4)] = adj[i] & adj[j]
    deg = adj.sum(axis=1)
    best = 0
    ntri = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                cm = common[i | (j << 4)]
                # triangles where i<j<k
                cm &= ~((1 << (j + 1)) - 1)
                while cm:
                    k = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    ntri += 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best, ntri


def _try_swap(adj, i, j, k, l):
    """Try removing edge (i,j) and adding edge (k,l). Return new adj if valid."""
    if adj[i, j] != 1 or adj[k, l] != 0:
        return None
    if i == k or i == l or j == k or j == l:
        return None
    new_adj = adj.copy()
    new_adj[i, j] = 0
    new_adj[j, i] = 0
    new_adj[k, l] = 1
    new_adj[l, k] = 1
    return new_adj


def _score_improvement(adj, i, j, k, l):
    """Estimate change in max degree sum from removing (i,j) and adding (k,l)."""
    deg = adj.sum(axis=1)
    # Removing (i,j): deg[i]-=1, deg[j]-=1
    # Adding (k,l): deg[k]+=1, deg[l]+=1
    # This is a rough heuristic; full recompute is done in accept step.
    return 0


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s

    # Initial: K_{8,8} + one intra-part edge
    adj = np.zeros((N, N), dtype=np.uint8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1

    best_adj = adj.copy()
    best_score, best_ntri = _score(adj)

    rng = random.Random(42)

    # Precompute all potential edge pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # Local search: random edge swaps
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Pick a random existing edge to remove and a random non-edge to add
        edges_present = [(i, j) for i, j in all_pairs if adj[i, j]]
        edges_absent = [(i, j) for i, j in all_pairs if not adj[i, j]]

        if not edges_present or not edges_absent:
            break

        # Try multiple swaps per iteration
        for _ in range(200):
            if time.time() > deadline:
                break
            i, j = rng.choice(edges_present)
            k, l = rng.choice(edges_absent)
            if i == k or i == l or j == k or j == l:
                continue
            new_adj = _try_swap(adj, i, j, k, l)
            if new_adj is None:
                continue
            s, nt = _score(new_adj)
            if s < best_score:
                best_score = s
                best_adj = new_adj.copy()
                adj = new_adj
                # Recompute edge lists
                edges_present = [(i, j) for i, j in all_pairs if adj[i, j]]
                edges_absent = [(i, j) for i, j in all_pairs if not adj[i, j]]
            elif s == best_score and nt < best_ntri:
                # Same max but fewer triangles with that max (tiebreak)
                adj = new_adj
                best_ntri = nt
                edges_present = [(i, j) for i, j in all_pairs if adj[i, j]]
                edges_absent = [(i, j) for i, j in all_pairs if not adj[i, j]]

    # Final: ensure exactly 65 edges
    edge_count = int(adj.sum()) // 2
    edges_list = [[i, j] for i in range(N) for j in range(i + 1, N) if adj[i, j]]

    return {'n': N, 'edges': edges_list}


if __name__ == '__main__':
    result = construct()
    print(f"Edges: {len(result['edges'])}")
    adj = np.zeros((N, N), dtype=np.uint8)
    for u, v in result['edges']:
        adj[u, v] = 1
        adj[v, u] = 1
    s, nt = _score(adj)
    print(f"Score: {s}, Triangles: {nt}")
