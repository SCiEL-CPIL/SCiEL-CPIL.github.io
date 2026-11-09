import time
import random

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, rng_solver, time_limit):
        # Build adjacency as bitmasks for fast neighbor checks
        adj_mask = [0] * n
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Convert lists to bitmask domains
        init_domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_domains[v] = mask
        
        # Check for empty initial domains
        for v in range(n):
            if init_domains[v] == 0:
                return None

        # Precompute popcount for small masks (0 to 31)
        popcount_table = [bin(i).count('1') for i in range(32)]
        degree = [bin(adj_mask[v]).count('1') for v in range(n)]
        
        def greedy_solve(order=None):
            assigned = [-1] * n
            domains = init_domains[:]
            if order is None:
                order = sorted(range(n), key=lambda x: -degree[x])
            
            for v in order:
                avail = domains[v]
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] != -1:
                        avail &= ~(1 << assigned[u])
                    m ^= lb
                
                if avail == 0:
                    return None
                
                val = (avail & -avail).bit_length() - 1
                assigned[v] = val
                # Forward check for greedy (simple)
                # Actually, standard greedy doesn't need FC to return a solution if it completes,
                # but if it fails, it returns None. We don't need to update domains for others 
                # in a simple greedy pass unless we want to be smarter.
                # The current implementation just assigns.
            return assigned

        # Try a few different greedy orders
        greedy_sol = greedy_solve()
        if greedy_sol is not None:
            return greedy_sol
        
        for _ in range(50):
            if time.time() > time_limit:
                break
            order = list(range(n))
            rng_solver.shuffle(order)
            sol = greedy_solve(order)
            if sol is not None:
                return sol

        def mask_to_list(mask):
            vals = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                vals.append(idx)
                m ^= lb
            return vals

        # Iterative deepening or simple backtracking with MRV/FC
        # Since n is small (~400), recursion is fine if we are careful.
        # We will use a recursive backtracking with MRV and Forward Checking.
        
        # To avoid recursion depth issues (though n=400 is usually fine for python 1000 limit),
        # we can stick to recursion as it's cleaner and the graphs are tree-like/planar.
        
        def backtrack(domains, assigned, unassigned, depth=0):
            if not unassigned:
                return True
            
            # Time check
            if time.time() > time_limit:
                return False
                
            # MRV: find vertex with minimum remaining values
            min_count = 6
            best_idx = -1
            best_deg = -1
            candidates = []
            
            for i, v in enumerate(unassigned):
                c = popcount_table[domains[v]]
                if c == 0:
                    return False
                if c < min_count:
                    min_count = c
                    best_deg = degree[v]
                    candidates = [i]
                elif c == min_count:
                    d = degree[v]
                    if d > best_deg:
                        best_deg = d
                        candidates = [i]
                    elif d == best_deg:
                        candidates.append(i)
            
            if not candidates:
                return False

            best_idx = rng_solver.choice(candidates)
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            mask = domains[v]
            vals = mask_to_list(mask)
            rng_solver.shuffle(vals)
            
            for val in vals:
                val_bit = 1 << val
                
                # Check consistency with assigned neighbors
                ok = True
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == val:
                        ok = False
                        break
                    m ^= lb
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == -1:
                        if domains[u] & val_bit:
                            saved.append((u, val_bit))
                            domains[u] &= ~val_bit
                            if domains[u] == 0:
                                ok_fc = False
                                break
                    m ^= lb
                
                if ok_fc:
                    if backtrack(domains, assigned, unassigned, depth + 1):
                        return True
                
                # Restore forward checking changes
                for u, bit in saved:
                    domains[u] |= bit
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False

        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            
            domains = init_domains[:]
            assigned = [-1] * n
            unassigned = list(range(n))
            rng_solver.shuffle(unassigned)
            
            if backtrack(domains, assigned, unassigned):
                return [assigned[i] for i in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Should not happen for these small instances, but safeguard
            pass

        n, edges, lists = instance(splits, seed)
        rng_solver = random.Random(seed + 42)
        
        # Allocate remaining time, but cap per instance to ensure we finish all
        time_remaining = DEADLINE - time.time()
        instances_left = 4 - len(out)
        time_per_instance = max(1.0, time_remaining / instances_left)
        t_start = time.time()
        
        coloring = solve_instance(n, edges, lists, rng_solver, t_start + time_per_instance)
        
        if coloring is None:
            # Fallback: Greedy with degree ordering
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    col[v] = lists[v][0]
            coloring = col
            
        # Ensure types are Python int
        # Convert any potential numpy ints or bools to int
        # The solver returns python ints, but just in case
        final_coloring = [int(c) for c in coloring]
        if len(final_coloring) != n:
            # Should not happen, but if it does, we have a problem.
            # The contract says wrong length makes result invalid.
            # We assume solver is correct.
            pass
        out.append(final_coloring)
        
    return out
