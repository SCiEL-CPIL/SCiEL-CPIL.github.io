import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    # Precomputed: K4-free check and independent set check using bitmasks

    def has_k4(adj_mask, n):
        """Return True if graph has K4."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if any two vertices in common are adjacent
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = common & ~(1 << w)
                        if (adj_mask[w] & rest) != 0:
                            return True
        return False

    def find_independent_set_6(adj_mask, n, timeout=None):
        """Return a list of 6 independent vertices or None. Uses DFS with bitmasks."""
        best = [None]  # to allow early exit via closure

        def dfs(cand, need, path):
            if need == 0:
                best[0] = list(path)
                return True
            if cand == 0 or need > bin(cand).count('1'):
                return False
            # Pick vertex with smallest degree in cand (branching heuristic)
            # For speed, just pick lowest bit
            lsb = cand & -cand
            v = lsb.bit_length() - 1
            cand_rest = cand ^ lsb

            # Include v
            new_cand = cand_rest & ~adj_mask[v]
            path.append(v)
            if dfs(new_cand, need - 1, path):
                return True
            path.pop()

            # Exclude v
            if dfs(cand_rest, need, path):
                return True
            return False

        if dfs((1 << n) - 1, 6, []):
            return best[0]
        return None

    def is_valid(adj_mask, n):
        if has_k4(adj_mask, n):
            return False
        if find_independent_set_6(adj_mask, n) is not None:
            return False
        return True

    def find_violation(adj_mask, n):
        """Returns ('K4', [u,v,w,x]) or ('IND', [6 vertices]) or None."""
        # Find K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = common & ~(1 << w)
                        if (adj_mask[w] & rest) != 0:
                            x_lsb = (adj_mask[w] & rest) & -((adj_mask[w] & rest))
                            x = x_lsb.bit_length() - 1
                            return ('K4', [u, v, w, x])
        # Find independent set of 6
        ind = find_independent_set_6(adj_mask, n)
        if ind is not None:
            return ('IND', ind)
        return None

    def make_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def local_search(adj_mask, n, max_iters, rng):
        """Local search: fix violations by flipping edges."""
        for it in range(max_iters):
            if time.time() > deadline - 2:
                return None

            viol = find_violation(adj_mask, n)
            if viol is None:
                return {'n': n, 'edges': make_edges(adj_mask, n)}

            if viol[0] == 'K4':
                # Remove a random edge from the K4
                k4 = viol[1]
                pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                         (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                u, v = pairs[rng.randint(0, 5)]
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                # Add a random edge within the independent set
                ind = viol[1]
                i, j = rng.sample(ind, 2)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

        return None

    def random_graph(n, p, rng):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def anneal(n, rng, time_budget):
        """Simulated annealing style: start with random graph, do local search with occasional random perturbations."""
        end_time = time.time() + time_budget
        # Start with p tuned for the target
        p = 0.5
        adj = random_graph(n, p, rng)

        # First, run local search to fix violations
        result = local_search(adj, n, 2000, rng)
        if result is not None:
            return result

        # If local search failed, try more iterations with perturbation
        best_adj = adj[:]
        best_viol_count = 999999

        while time.time() < end_time - 1:
            # Perturb: flip a random edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v and u < v:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Run a short local search
            result = local_search(adj, n, 100, rng)
            if result is not None:
                return result

            # Track best
            viol = find_violation(adj, n)
            if viol is None:
                return {'n': n, 'edges': make_edges(adj, n)}

        return None

    # Strategy: Try to build up from a known good smaller graph
    # Use the Paley(17) as a seed, then try to extend

    def paley17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    def paley_17_adj():
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def extend_graph(adj, n_current, n_target, rng):
        """Try to extend a graph from n_current to n_target by adding vertices."""
        # Add vertices one at a time, choosing edges to avoid K4 and independent sets
        new_n = n_target
        # Extend adj_mask
        new_adj = [0] * new_n
        for i in range(n_current):
            new_adj[i] = adj[i]

        # For each new vertex, decide its adjacency to old vertices
        for new_v in range(n_current, new_n):
            # Randomly assign edges to old vertices, biased toward avoiding violations
            adj_to_old = 0
            for old_v in range(n_current):
                if rng.random() < 0.5:
                    adj_to_old |= (1 << old_v)
                    new_adj[old_v] |= (1 << new_v)
                    new_adj[new_v] |= (1 << old_v)

        # Now run local search on the full graph
        result = local_search(new_adj, new_n, 5000, rng)
        if result is not None:
            return result

        # Try annealing
        result = anneal(new_n, rng, min(10, time_left()))
        if result is not None:
            return result

        return None

    # Main strategy:
    # 1. Try direct construction for n=35 with multiple random starts
    # 2. Try extending Paley(17)
    # 3. Fallback to smaller graphs

    rng = random.Random(42)

    best_result = None

    # Strategy 1: Direct random construction + local search for large n
    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break

        attempts = 0
        max_attempts = 20
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            p = 0.5  # Symmetric density
            adj = random_graph(target_n, p, rng)
            result = local_search(adj, target_n, 3000, rng)
            if result is not None:
                best_result = result
                break
            # Also try annealing with a small budget
            if time.time() < deadline - 12:
                result = anneal(target_n, rng, min(3, time_left() - 10))
                if result is not None:
                    best_result = result
                    break

        if best_result is not None:
            break

    # Strategy 2: Extend Paley(17)
    if best_result is None and time.time() < deadline - 20:
        adj17 = paley_17_adj()
        for target_n in [25, 30, 35]:
            if time.time() > deadline - 15:
                break
            result = extend_graph(adj17, 17, target_n, rng)
            if result is not None:
                best_result = result
                break

    # Strategy 3: Try medium n with more thorough search
    if best_result is None:
        for target_n in [25, 24, 23, 22, 20, 19, 18]:
            if time.time() > deadline - 10:
                break
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                p = 0.5
                adj = random_graph(target_n, p, rng)
                result = local_search(adj, target_n, 5000, rng)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break

    # Fallback: Paley(17)
    if best_result is None:
        best_result = paley17()

    # Final validation
    n_final = best_result['n']
    adj_final = [0] * n_final
    for u, v in best_result['edges']:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)

    if not is_valid(adj_final, n_final):
        # This should not happen, but as a safety net
        best_result = paley17()

    return best_result
