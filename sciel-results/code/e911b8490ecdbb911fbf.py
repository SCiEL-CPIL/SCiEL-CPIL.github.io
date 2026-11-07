"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    sys.setrecursionlimit(10000)

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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use bitmask representation for domains (colors 0-4 fit in 5 bits)
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask

        color = [-1] * n

        # Precompute bit counts for fast MRV
        bit_count = [0] * 32
        for i in range(32):
            bit_count[i] = bin(i).count('1')

        # AC-3 implementation for maintaining arc consistency
        def enforce_ac3(initial_queue=None):
            """
            Enforces arc consistency. 
            If initial_queue is None, initializes with all arcs.
            Returns False if a domain becomes empty.
            Modifies domains in place.
            """
            if initial_queue is None:
                queue = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            queue.append((u, v))
                            queue.append((v, u))
            else:
                queue = list(initial_queue)

            while queue:
                u, v = queue.pop()
                du = domains[u]
                dv = domains[v]
                
                # Optimization: If dv has more than 1 bit, every value in du has support in dv 
                # UNLESS du contains a value that is NOT in dv? 
                # Wait. Arc (u,v) consistency means for every c in du, there exists d in dv such that c != d.
                # If dv has at least 2 values, say d1, d2. 
                # If c == d1, then d2 is a support. 
                # If c != d1 and c != d2, then d1 is a support.
                # So if |dv| >= 2, every c in du has support in dv. 
                # The only case where a value c in du loses support is if dv is a singleton {d} and c == d.
                # In that case, c is removed from du.
                
                if bit_count[dv] >= 2:
                    continue
                
                # dv is a singleton or empty (but we check empty later)
                if dv == 0:
                    return False
                
                # dv is a singleton mask
                # Check if du contains the value in dv
                if du & dv:
                    # The value in dv is in du, so it must be removed from du
                    new_du = du & ~dv
                    if new_du != du:
                        domains[u] = new_du
                        if new_du == 0:
                            return False
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
                # If du & dv == 0, no value in du is equal to the value in dv, so all have support.
                
            return True

        # Initial AC-3
        if not enforce_ac3():
            return [0] * n

        # Backtracking with MAC (Maintaining Arc Consistency)
        def backtrack():
            # MRV: Select variable with minimum remaining values
            min_len = 1000
            best_v = -1
            for v in range(n):
                if color[v] == -1:
                    d_len = bit_count[domains[v]]
                    if d_len == 0:
                        return False
                    if d_len < min_len:
                        min_len = d_len
                        best_v = v
                        if min_len == 1:
                            break
            
            if best_v == -1:
                return True

            v = best_v
            saved_domain = domains[v]
            
            # Get list of available colors
            available_colors = []
            temp = saved_domain
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                available_colors.append(c)
                temp ^= lsb
            
            # LCV: Sort colors by number of values removed from neighbors (ascending)
            # Only needed if min_len > 1. If min_len == 1, there's only one choice.
            if min_len > 1:
                def count_removed(c):
                    count = 0
                    bit = 1 << c
                    for u in adj[v]:
                        if color[u] == -1:
                            if domains[u] & bit:
                                count += 1
                    return count
                available_colors.sort(key=count_removed)
            
            # Try each color in LCV order
            for c in available_colors:
                color[v] = c
                # Save domains to restore
                domains_snapshot = domains[:]
                domains[v] = (1 << c)
                
                # Run AC-3 starting from neighbors of v
                ac_queue = []
                for u in adj[v]:
                    if color[u] == -1:
                        ac_queue.append((u, v))
                        
                if not enforce_ac3(ac_queue):
                    # Restore and try next
                    domains[:] = domains_snapshot
                    continue

                if backtrack():
                    return True
                
                # Restore domains
                domains[:] = domains_snapshot
                color[v] = -1
            
            domains[v] = saved_domain
            color[v] = -1
            return False

        if backtrack():
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to avoid crash
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
