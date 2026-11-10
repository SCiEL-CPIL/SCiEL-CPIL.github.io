"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    triples = list(combinations(range(n), 3))
    # Precompute triple intersection masks
    triple_masks = []
    for (u, v, w) in triples:
        triple_masks.append((u, v, w))

    def popcount(x):
        return bin(x).count('1')

    def is_k33_free(adj_mask):
        for (u, v, w) in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) >= 3:
                return False
        return True

    def edge_count(adj_mask):
        return sum(popcount(m) for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    def greedy_add(adj_mask, order, rng):
        """Add edges in given order if K3,3-free."""
        for (u, v) in order:
            if time.time() > deadline:
                break
            if adj_mask[u] & (1 << v):
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if not is_k33_free(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        return adj_mask

    def local_search(adj_mask, max_iters=200):
        """Try to improve by removing edges to fix violations, then re-adding."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Find a violating triple
            violating = None
            for (u, v, w) in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) >= 3:
                    violating = (u, v, w, common)
                    break
            if violating is None:
                break
            u, v, w, common = violating
            # Try removing one edge incident to one of the common neighbors
            improved = False
            for c in [x for x in range(n) if common & (1 << x)]:
                for (a, b) in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & (1 << b):
                        # Remove edge a-b
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                        if is_k33_free(adj_mask):
                            improved = True
                            break
                        else:
                            # Restore
                            adj_mask[a] |= (1 << b)
                            adj_mask[b] |= (1 << a)
                if improved:
                    break
            if not improved:
                break
        # After fixing, try to re-add missing edges
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
        random.shuffle(missing)
        for (u, v) in missing:
            if time.time() > deadline:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if not is_k33_free(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        return adj_mask

    best_adj = None
    best_edges = 0

    # Strategy 1: Start from K_{2,18}
    adj = [0] * n
    for a in range(2):
        for b in range(2, n):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)

    missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
    random.shuffle(missing)
    adj = greedy_add(adj, missing, random)
    adj = local_search(adj)
    cnt = edge_count(adj)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj)

    # Strategy 2: Random initializations with varying densities
    rng = random.Random(42)
    densities = [0.3, 0.4, 0.5, 0.6, 0.35, 0.45, 0.55, 0.65]
    trial = 0
    for density in densities:
        for _ in range(5):
            trial += 1
            if time.time() > deadline:
                break
            adj = [0] * n
            for (i, j) in all_pairs:
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

            if not is_k33_free(adj):
                # Try to fix by removing edges
                adj = local_search(adj, max_iters=50)
                if not is_k33_free(adj):
                    continue

            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
            rng.shuffle(missing)
            adj = greedy_add(adj, missing, rng)
            adj = local_search(adj)
            cnt = edge_count(adj)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = list(adj)

        if time.time() > deadline:
            break

    # Build result from best
    if best_adj is None:
        # Fallback: K_{2,18}
        adj = [0] * n
        for a in range(2):
            for b in range(2, n):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj
        best_edges = 36

    edges = mask_to_edges(best_adj)
    return edges
# EVOLVE-BLOCK-END
