import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin

    # --- Core utilities ---

    def is_triangle_free(adj, n):
        """Check triangle-freeness using bitmasks: for every edge (i,j), adj[i] & adj[j] must be 0."""
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                if i < j:
                    if adj[i] & adj[j]:
                        return False
                mask ^= lsb
        return True

    def alpha_bound(adj, n, limit=10):
        """
        Exact branch-and-bound for the independence number.
        Returns min(alpha, limit).  If it ever proves alpha >= limit, it returns limit early.
        Uses the classic Tomita-style ordering: pick the vertex of minimum degree in the
        remaining candidate set (induced subgraph) to maximise pruning.
        """
        best = 0

        # Precompute full adjacency masks
        # We'll operate on Python ints as bitmasks.

        def dfs(candidates_mask, current_size):
            nonlocal best
            # Bound
            if current_size + _popcount(candidates_mask) <= best:
                return
            if candidates_mask == 0:
                if current_size > best:
                    best = current_size
                return
            if best >= limit:
                return

            # Pick vertex with minimum degree in the induced subgraph on candidates
            # For speed, just pick the first candidate; for n<=40 this is often fine
            # But let's use a simple heuristic: pick the vertex with the fewest neighbours in candidates
            # Actually, for the independence number, we want to branch on a vertex that
            # removes many candidates when included.  The standard trick is to pick the
            # vertex of maximum degree in the candidate subgraph for the *exclude* branch
            # to be large.  Let's just pick the first one for simplicity and rely on
            # the bound.  For n=40 and limit=10, this should be fast enough.
            
            # Simple: pick lowest set bit
            lsb = candidates_mask & (-candidates_mask)
            v = lsb.bit_length() - 1
            rest = candidates_mask & ~lsb

            # Branch 1: Include v
            # New candidates = rest minus neighbours of v
            new_cands = rest & ~adj[v]
            dfs(new_cands, current_size + 1)

            if best >= limit:
                return

            # Branch 2: Exclude v
            dfs(rest, current_size)

        dfs((1 << n) - 1, 0)
        return best

    def _popcount(x):
        return bin(x).count('1')

    def generate_greedy_tf(n, seed):
        """
        Greedy triangle-free graph generation:
        Shuffle all edges, add each if it doesn't create a triangle.
        This guarantees triangle-freeness by construction.
        """
        random.seed(seed)
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search_reduce_alpha(adj, n, time_limit):
        """
        Local search to reduce the independence number.
        Moves: add an edge (u,v) where u,v are non-adjacent, then remove edges
        to restore triangle-freeness.  Accept if alpha decreases.
        """
        best_adj = adj[:]
        best_alpha = alpha_bound(adj, n, 10)
        if best_alpha < 10:
            return best_adj, best_alpha

        alpha = best_alpha
        temperature = 1.0
        cooling = 0.998

        while time.time() < time_limit:
            # Find an independent set of size close to alpha
            large_is = _find_independent_set(adj, n, min(alpha, 10))
            if not large_is or len(large_is) < 2:
                break

            # Pick two non-adjacent vertices from the independent set
            u, v = random.sample(large_is, 2)
            if (adj[u] >> v) & 1:
                continue

            # Try adding edge (u, v)
            old_adj = adj[:]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

            # Remove edges to break triangles
            conflicts = adj[u] & adj[v]
            while conflicts:
                lsb = conflicts & (-conflicts)
                w = lsb.bit_length() - 1
                # Remove either (u,w) or (v,w)
                if random.random() < 0.5:
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                else:
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
                conflicts ^= lsb

            new_alpha = alpha_bound(adj, n, 10)
            delta = new_alpha - alpha

            if delta < 0:
                alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = adj[:]
                    if new_alpha < 10:
                        return best_adj, best_alpha
            else:
                # Metropolis acceptance
                if random.random() < (2.718281828 ** (-delta / max(temperature, 0.01))):
                    alpha = new_alpha
                else:
                    adj = old_adj[:]

            temperature *= cooling

        return best_adj, best_alpha

    def _find_independent_set(adj, n, target):
        """Find an independent set of size >= target, or as large as possible."""
        found = []

        def dfs(candidates_mask, current_set):
            nonlocal found
            if len(current_set) >= target:
                found = current_set[:]
                return True
            if candidates_mask == 0:
                return False

            lsb = candidates_mask & (-candidates_mask)
            v = lsb.bit_length() - 1
            rest = candidates_mask & ~lsb

            # Include v
            if dfs(rest & ~adj[v], current_set + [v]):
                return True

            # Exclude v
            if dfs(rest, current_set):
                return True

            return False

        dfs((1 << n) - 1, [])
        return found if found else None

    # --- Main search ---

    # Try n=40 first, then 39, then 38, etc.
    for n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 5:
            break

        found = False
        max_restarts = 3000
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break

            seed = r * 7919 + n * 104729
            adj = generate_greedy_tf(n, seed)
            alpha = alpha_bound(adj, n, 10)

            if alpha < 10:
                found = True
                break

            # If alpha is close to 10, try local search
            if alpha <= 11 and time.time() < deadline - 15:
                adj, alpha = local_search_reduce_alpha(adj, n, deadline - 10)
                if alpha < 10:
                    found = True
                    break

        if found:
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= lsb
            return {'n': n, 'edges': edges}

    # Fallback: try a few more n values with more restarts
    for n in [29, 28, 27, 26]:
        if time.time() > deadline - 5:
            break
        for r in range(5000):
            if time.time() > deadline - 5:
                break
            seed = r * 31337 + n
            adj = generate_greedy_tf(n, seed)
            if alpha_bound(adj, n, 10) < 10:
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & (-mask)
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}

    # Absolute fallback
    return {'n': 1, 'edges': []}
