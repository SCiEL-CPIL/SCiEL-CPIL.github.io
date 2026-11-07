"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    
    def instance(splits, seed, palette=5):
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

    def solve_csp(n, edges, lists, time_limit=10.0):
        """
        Solves the list coloring problem using Backtracking with MRV and Forward Checking.
        Returns a list of colors if a solution is found, otherwise None.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Convert lists to bitmasks for fast set operations
        # Palette is 0..4, so we need 5 bits.
        # domain[v] is an integer where bit c is 1 if color c is available.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
            
        color = [-1] * n
        
        start_time = time.time()
        timeout = False
        solution_found = False
        
        # Recursive backtracking function
        def backtrack():
            nonlocal timeout, solution_found
            
            if timeout:
                return False
            
            if time.time() - start_time > time_limit:
                timeout = True
                return False
                
            # 1. Variable Selection: MRV (Minimum Remaining Values)
            # Find uncolored vertex with the smallest domain size
            best_v = -1
            min_count = n + 1
            
            for v in range(n):
                if color[v] != -1:
                    continue
                count = bin(domains[v]).count('1')
                if count == 0:
                    return False # Dead end
                if count < min_count:
                    min_count = count
                    best_v = v
                    if count == 1:
                        break # Can't get better than 1
                
            if best_v == -1:
                # All vertices colored
                solution_found = True
                return True
                
            # 2. Value Ordering: Try colors in the domain of best_v
            # We iterate through the bits set in domains[best_v]
            v = best_v
            mask = domains[v]
            
            while mask:
                if timeout:
                    return False
                # Get the lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb # Remove this bit from local mask to try next color later
                
                # Assign color c to v
                color[v] = c
                domains[v] = 0 # Remove all options for v
                
                # 3. Forward Checking: Update neighbors' domains
                undo = []
                valid = True
                for u in adj[v]:
                    if color[u] != -1:
                        # If neighbor is colored, check conflict
                        if color[u] == c:
                            valid = False
                            break
                        continue
                    # Neighbor is uncolored, remove c from its domain
                    if domains[u] & (1 << c):
                        old_domain = domains[u]
                        new_domain = old_domain & ~(1 << c)
                        domains[u] = new_domain
                        undo.append((u, old_domain))
                        if new_domain == 0:
                            valid = False
                            break
                    
                if valid:
                    if backtrack():
                        return True
                else:
                    # Conflict or dead end, undo changes
                    for u, old_dom in undo:
                        domains[u] = old_dom
                        if color[u] == -1:
                             # We didn't color u, so we just restore domain
                             pass
                    # Note: We didn't color u in this branch, so color[u] remains -1.
                    # We only need to restore domains.
                
                # Undo assignment of v
                color[v] = -1
                # Restore domain of v (it was 0, now we need to restore it to what it was before assignment)
                # Actually, we can just recompute or store it. 
                # Simpler: We passed 'mask' which is local. The global domains[v] was set to 0.
                # We need to restore it to the value it had at the start of the loop iteration for v.
                # Let's refactor to store original domain before assignment.
                
            return False
        
        # The above implementation has a subtle bug in undo logic for domains[v].
        # Let's rewrite the core loop more carefully to ensure correctness and speed.
        
        def backtrack_safe():
            nonlocal timeout, solution_found
            
            if timeout:
                return False
            
            if time.time() - start_time > time_limit:
                timeout = True
                return False
                
            # MRV
            best_v = -1
            min_count = n + 1
            for v in range(n):
                if color[v] != -1:
                    continue
                count = bin(domains[v]).count('1')
                if count == 0:
                    return False
                if count < min_count:
                    min_count = count
                    best_v = v
                    if count == 1:
                        break
                        
            if best_v == -1:
                solution_found = True
                return True
                
            v = best_v
            original_domain_v = domains[v]
            color[v] = -1 # Ensure it's uncolored before we start trying
            
            # Try each color in original_domain_v
            mask = original_domain_v
            while mask:
                if timeout:
                    return False
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                # Assign
                color[v] = c
                domains[v] = 0
                
                # Forward Check
                undo_list = []
                valid = True
                for u in adj[v]:
                    if color[u] != -1:
                        if color[u] == c:
                            valid = False
                            break
                        continue
                    if domains[u] & (1 << c):
                        old_u = domains[u]
                        new_u = old_u & ~(1 << c)
                        domains[u] = new_u
                        undo_list.append((u, old_u))
                        if new_u == 0:
                            valid = False
                            break
                            
                if valid:
                    if backtrack_safe():
                        return True
                # Undo Forward Checks
                for u, old_u in undo_list:
                    domains[u] = old_u
                    
                # Undo Assignment
                color[v] = -1
                domains[v] = original_domain_v
                
            return False

        if backtrack_safe() and solution_found:
            return color
        return None

    def greedy_fallback(n, edges, lists):
        """Fast greedy solver for fallback."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [-1] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] != -1}
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0] # Fallback to first in list if stuck
        return colour

    out = []
    global_start = time.time()
    deadline = 145.0 # Safety margin before 150s
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we are running out of time, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            out.append(greedy_fallback(n, edges, lists))
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Try CSP solver first
        res = solve_csp(n, edges, lists, time_limit=5.0)
        
        if res is None:
            # Fallback to greedy if CSP fails or times out
            res = greedy_fallback(n, edges, lists)
            
        # Ensure types are correct (Python int)
        out.append([int(c) for c in res])
        
    return out
# EVOLVE-BLOCK-END
