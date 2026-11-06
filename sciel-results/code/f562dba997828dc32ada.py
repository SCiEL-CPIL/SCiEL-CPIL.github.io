import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_k33_free(adj):
        """True iff no triple has 3 common neighbors (i.e. no K3,3)."""
        for a in range(n):
            for b in range(a + 1, n):
                ab = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (ab & adj[c]).bit_count() > 2:
                        return False
        return True

    def can_add(adj, u, v):
        """Would adding edge (u,v) create a K3,3?"""
        if adj[u] & (1 << v):
            return False
        # New common neighbors: a vertex x is in N(u)∩N(v)∩N(w) for some w.
        # Such a triple {u,v,w} would then have common neighbors including x.
        # Check all triples that include u and v: {u, v, w}
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] & adj[v] & adj[w]).bit_count() > 2:
                return False
        # Also triples {u, a, b} where u-v edge adds v to N(u)∩N(a)∩N(b):
        # v is a new common neighbor of {u,a,b} iff v in N(a) and N(b).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if (adj[b] & (1 << v)):
                    if (adj[u] & adj[a] & adj[b]).bit_count() > 2:
                        return False
        # Symmetric: triples {v, a, b} where u-v adds u to N(v)∩N(a)∩N(b)
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if (adj[b] & (1 << u)):
                    if (adj[v] & adj[a] & adj[b]).bit_count() > 2:
                        return False
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy(seed):
        random.seed(seed)
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj):
        cur = adj[:]
        # Pass 1: add any possible edge
        changed = True
        while changed:
            if time.time() > deadline:
                return cur
            changed = False
            for u, v in all_pairs:
                if cur[u] & (1 << v):
                    continue
                if can_add(cur, u, v):
                    cur[u] |= (1 << v)
                    cur[v] |= (1 << u)
                    changed = True
        # Pass 2: edge swaps (remove one edge to enable adding two others)
        improved = True
        while improved:
            if time.time() > deadline:
                return cur
            improved = False
            # Missing edges
            missing = [(u, v) for u, v in all_pairs if not (cur[u] & (1 << v))]
            random.shuffle(missing)
            present = [(u, v) for u, v in all_pairs if (cur[u] & (1 << v))]
            random.shuffle(present)
            for e_add in missing:
                if time.time() > deadline:
                    return cur
                u, v = e_add
                # Try removing each present edge to enable adding e_add
                for e_rm in present:
                    a, b = e_rm
                    if a == u or a == v or b == u or b == v:
                        continue
                    # Remove e_rm
                    cur[a] &= ~(1 << b)
                    cur[b] &= ~(1 << a)
                    if can_add(cur, u, v):
                        cur[u] |= (1 << v)
                        cur[v] |= (1 << u)
                        improved = True
                        break
                    # Restore
                    cur[a] |= (1 << b)
                    cur[b] |= (1 << a)
                if improved:
                    break
        return cur

    best_adj = [0] * n
    best_edges = 0

    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        adj = greedy(i)
        adj = local_search(adj)
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj[:]

    # Final polish
    if time.time() < deadline - 2:
        adj = local_search(best_adj)
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
