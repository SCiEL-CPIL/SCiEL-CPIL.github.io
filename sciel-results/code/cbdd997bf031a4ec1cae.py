import time
import random

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a 135s deadline to be safe.
    deadline = start_time + 135.0

    # Fallback: Andrásfai graph And(9) on 26 vertices
    # It is triangle-free and has independence number 9.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_independent_set_10(adj, n):
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False
            
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            new_mask_include = non_neighbors[best_v] & mask
            if dfs(new_mask_include, size + 1):
                return True
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                return True
            return False
            
        return dfs(all_mask, 0)

    def try_cayley(n, max_attempts):
        """
        Attempts to construct a valid triangle-free Cayley graph on Z_n.
        A Cayley graph is defined by a connection set S.
        For it to be triangle-free, S must be sum-free: no a, b in S such that a+b in S.
        Note: Since the graph is undirected, S is symmetric (x in S iff -x in S).
        """
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # We want to build a symmetric sum-free set S in Z_n.
            # Start with empty set.
            S = set()
            # Candidates are pairs {x, n-x} for x in 1..n//2
            # We cannot include 0.
            candidates = list(range(1, n // 2 + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if x == n - x: # x = n/2
                    pair = {x}
                else:
                    pair = {x, n - x}
                
                # Check if adding pair keeps S sum-free
                # New S would be S union pair.
                # We need to ensure no a, b in S_union_pair such that a+b in S_union_pair.
                # This is expensive to check fully each time.
                # Approximation: Check against existing S and within pair.
                
                valid = True
                # Check internal consistency of pair
                if len(pair) == 2:
                    p1, p2 = list(pair)
                    if (p1 + p2) % n in pair:
                        valid = False
                    if (p1 + p1) % n in pair:
                        valid = False
                    if (p2 + p2) % n in pair:
                        valid = False
                
                if valid:
                    # Check interaction with existing S
                    for s in S:
                        for p in pair:
                            sum_val = (s + p) % n
                            if sum_val in S or sum_val in pair:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    S.update(pair)
            
            if not S:
                continue
                
            # Build adjacency masks
            adj = [0] * n
            for u in range(n):
                for v in S:
                    neighbor = (u + v) % n
                    adj[u] |= (1 << neighbor)
            
            # Verify triangle-free (should be by construction, but double check is cheap)
            # Actually, the sum-free condition guarantees triangle-free for Cayley graphs on Z_n?
            # Triangle u, u+a, u+b. Edges: a, b, a+b (or b-a).
            # If a, b in S, and a+b in S, triangle.
            # If a, b in S, and b-a in S, triangle?
            # u -> u+a (edge a), u+a -> u+b (edge b-a), u -> u+b (edge b).
            # So we also need S to be difference-free?
            # A set is "sum-free" usually means x+y!=z.
            # For triangle-free Cayley, S must not contain x, y, z such that x+y=z OR x-y=z.
            # Since S is symmetric, x-y=z implies x = y+z. So if S is sum-free, it covers both?
            # If x, y in S, and x-y in S. Let z = x-y. Then x = y+z.
            # If S is sum-free, y+z != x. So x-y not in S.
            # So yes, sum-free is sufficient for triangle-free in undirected Cayley graphs on abelian groups.
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    def try_random_greedy(n, max_attempts):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    # Main search loop
    # We try to find a graph with n vertices, starting from the maximum possible (40)
    # down to 27.
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Strategy 1: Cayley Graphs
        # These are regular and often have better independence properties.
        # We allocate a fixed budget of attempts per n, or dynamic based on time.
        # Let's try 100 Cayley attempts first.
        adj = try_cayley(n, 100)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
        # Strategy 2: Random Greedy
        # If Cayley failed, try random greedy.
        # Allocate remaining time for this n.
        # We don't know how much time is left, so we just run until deadline or success.
        # But we must leave time for the next n if we fail?
        # Actually, if we fail for n=40, we move to n=39.
        # If we spend all time on n=40, we return seed.
        # So we should cap the time per n.
        
        # Let's use a simple heuristic: if we have > 10s left, do more greedy attempts.
        # For now, just do 50 attempts.
        adj = try_random_greedy(n, 50)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
    return best_result
