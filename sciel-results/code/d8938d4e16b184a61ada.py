import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        # Bit c is set if color c is available
        init_domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            init_domains[i] = mask

        # Propagation function
        # Returns True if consistent, False if contradiction (empty domain)
        # Modifies domains in place
        def propagate(domains, changed):
            # changed is a set of vertices
            # We use a stack/list for efficiency
            stack = list(changed)
            while stack:
                v = stack.pop()
                dv = domains[v]
                if dv == 0:
                    return False
                if dv & (dv - 1) == 0: # Power of 2, i.e., singleton
                    # Find the single color
                    # k = index of the set bit
                    # Since palette is 0..4, we can just iterate or use bit trick
                    k = -1
                    temp = dv
                    idx = 0
                    while temp:
                        if temp & 1:
                            k = idx
                            break
                        temp >>= 1
                        idx += 1
                    # Propagate to neighbors
                    for u in adj[v]:
                        du = domains[u]
                        if du & (1 << k):
                            domains[u] = du & ~(1 << k)
                            if domains[u] == 0:
                                return False
                            if domains[u] & (domains[u] - 1) == 0:
                                stack.append(u)
            return True

        # Initial propagation
        # Start with all vertices as "changed" to ensure consistency? 
        # Actually, only if a domain is singleton do we propagate.
        # But initial domains are size 3, so no propagation needed initially unless some list is size 1? 
        # Lists are size 3. So initial propagation does nothing.
        
        # Backtracking search
        def backtrack(domains, assigned_count):
            if time.time() > deadline:
                return None
            
            # Find MRV variable
            min_size = 5
            var = -1
            min_deg = -1
            for v in range(n):
                # Skip assigned? We don't have a separate assigned array.
                # A variable is "assigned" if its domain is a singleton AND it has been processed?
                # No, in this formulation, if a domain is singleton, it's effectively assigned.
                # But we need to ensure we don't pick a variable that is already "fixed" by propagation?
                # Actually, if a domain is singleton, it's fixed. We should pick the next unassigned one.
                # How to know if it's assigned? 
                # In standard backtracking, we pick an unassigned variable.
                # Here, "assigned" means domain is singleton?
                # If we pick a singleton variable, we just recurse with it.
                # It's better to pick a variable with domain size > 1.
                # If all domains are size 1, we are done.
                
                size = bin(domains[v]).count('1')
                if size > 1:
                    if size < min_size or (size == min_size and len(adj[v]) > min_deg):
                        min_size = size
                        var = v
                        min_deg = len(adj[v])
            
            if var == -1:
                # All domains are singletons
                # Construct solution
                sol = [0] * n
                for v in range(n):
                    # Extract color
                    temp = domains[v]
                    idx = 0
                    while temp:
                        if temp & 1:
                            sol[v] = idx
                            break
                        temp >>= 1
                        idx += 1
                return sol
            
            # Try each color in domains[var]
            dv = domains[var]
            # Iterate through bits
            temp = dv
            idx = 0
            while temp:
                if temp & 1:
                    # Try color idx
                    new_domains = domains[:] # Copy
                    new_domains[var] = 1 << idx # Set to singleton
                    
                    # Propagate
                    if propagate(new_domains, {var}):
                        # Recurse
                        result = backtrack(new_domains, assigned_count + 1)
                        if result is not None:
                            return result
                temp >>= 1
                idx += 1
            
            return None

        # Start search
        # We need to handle the case where initial propagation might fail?
        # Initial domains are size 3, so no propagation.
        # But just in case:
        if not propagate(init_domains, set(range(n))):
            return None
        
        return backtrack(init_domains, 0)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            while len(out) < 4:
                out.append([0] * 8)
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        
        if colour is None:
            # Fallback: Randomized Greedy with Restarts
            # This is unlikely to be needed
            colour = None
            for attempt in range(100):
                if time.time() > deadline:
                    break
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                colour_try = [None] * n
                order = list(range(n))
                Random(seed + attempt).shuffle(order)
                
                success = True
                for v in order:
                    taken = {colour_try[u] for u in adj[v] if colour_try[u] is not None}
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        # Pick random free color
                        colour_try[v] = Random(seed + attempt + v).choice(free)
                    else:
                        success = False
                        break
                
                if success:
                    # Verify
                    valid = True
                    for v in range(n):
                        if colour_try[v] not in lists[v]:
                            valid = False
                            break
                        for u in adj[v]:
                            if u < v and colour_try[u] == colour_try[v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        colour = [int(c) for c in colour_try]
                        break
            
            if colour is None:
                colour = [0] * n
        
        out.append(colour)
    
    while len(out) < 4:
        out.append([0] * 8)
    
    return out
