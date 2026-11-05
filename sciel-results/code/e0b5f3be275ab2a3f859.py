import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        common_list = [j for j in range(n) if (common >> j) & 1]
        for idx, w in enumerate(common_list):
            for x in common_list[idx + 1:]:
                if (adj[w] >> x) & 1:
                    return False
        return True

    def has_triangle_in_set(adj, mask):
        """Check if the set of vertices (bitmask) contains a triangle."""
        verts = [i for i in range(n) if (mask >> i) & 1]
        for i in range(len(verts)):
            vi = verts[i]
            for j in range(i + 1, len(verts)):
                vj = verts[j]
                if (adj[vi] >> vj) & 1:
                    # Check if any common neighbor is also in the set
                    common = adj[vi] & adj[vj] & mask
                    if common:
                        return True
        return False

    def upper_bound_check(adj, s):
        """Return True if no (s+1)-subset is triangle-free (score <= s)."""
        # Check all subsets of size s+1
        from itertools import combinations
        for combo in combinations(range(n), s + 1):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if not has_triangle_in_set(adj, mask):
                return False
        return True

    def largest_tf(adj, time_limit=None):
        """Exact B&B for largest triangle-free induced subgraph."""
        best = [0]
        full_mask = (1 << n) - 1
        # Vertex ordering by degree descending for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])

        def bnb(S_mask, cands_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            # Upper bound: sz + popcount(cands_mask)
            if sz + bin(cands_mask).count('1') <= best[0]:
                return

            # Find addable vertices
            addable = []
            for v in order:
                if not ((cands_mask >> v) & 1):
                    continue
                NSv = adj[v] & S_mask
                # Check if NSv is independent
                ok = True
                NSv_list = [x for x in range(n) if (NSv >> x) & 1]
                for ai in range(len(NSv_list)):
                    a = NSv_list[ai]
                    for b in NSv_list[ai + 1:]:
                        if (adj[a] >> b) & 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    addable.append(v)

            if not addable:
                if sz > best[0]:
                    best[0] = sz
                return

            for v in addable:
                bnb(S_mask | (1 << v), cands_mask & ~(1 << v))
                if time_limit and time.time() > time_limit:
                    return

        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    random.seed(42)

    # Phase 1: Build a dense K4-free graph
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Quick score estimate
    init_score = largest_tf(adj)
    best_adj = adj[:]
    best_score = init_score
    print(f"Initial: score={best_score}, edges={sum(bin(a).count('1') for a in adj)//2}")

    # Phase 2: Local search with fast upper-bound check
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        if iteration % 200 == 0:
            print(f"Iter {iteration}, best={best_score}, t={time.time()-start_time:.1f}s")

        # Random perturbation: start from best or random
        if random.random() < 0.3 or best_score <= 8:
            adj = best_adj[:]
        else:
            adj = [0] * n
            random.shuffle(all_edges)
            for u, v in all_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        # Greedy edge addition
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        # Fast upper bound check: if score <= best_score, do exact
        # First try a quick check with s = best_score
        if upper_bound_check(adj, best_score - 1):
            # Score is at most best_score - 1, compute exact
            new_score = largest_tf(adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                print(f"Improved to {best_score} at iter {iteration}")
        else:
            # Score > best_score - 1, could still be equal
            # Do exact only if we have time
            if time.time() < deadline - 10.0:
                new_score = largest_tf(adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    print(f"Improved to {best_score} at iter {iteration}")

        # Local improvement: try single edge changes
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Try removing edges
                for u in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    nbrs = [v for v in range(n) if v > u and (adj[u] >> v) & 1]
                    random.shuffle(nbrs)
                    for v in nbrs:
                        # Remove edge
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        if upper_bound_check(adj, best_score - 1):
                            new_score = largest_tf(adj)
                            if new_score < best_score:
                                best_score = new_score
                                best_adj = adj[:]
                                improved = True
                                print(f"Edge removal improved to {best_score}")
                                break
                            # Revert
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            # Revert
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        if improved:
                            break

    # Final: return best
    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}")
    return edges
