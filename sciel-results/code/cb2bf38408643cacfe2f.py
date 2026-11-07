    def solve_instance(n, edges, lists, deadline):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Popcount lookup for 5-bit integers (0-31)
        popcount = [0] * 32
        for i in range(32):
            popcount[i] = bin(i).count('1')
            
        # Initial domains as bitmasks
        # Colors are 0..4, so 1<<c
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        assignment = [-1] * n
        unassigned = set(range(n))
        
        def mrv_select(unassigned_set, domains, popcount):
            best = -1
            best_size = 4 # Max domain size is 3, so 4 is infinity
            best_deg = -1
            for v in unassigned_set:
                sz = popcount[domains[v]]
                if sz == 0:
                    return v # Fail fast, though forward check should catch this
                deg = len(adj[v])
                if sz < best_size or (sz == best_size and deg > best_deg):
                    best_size = sz
                    best_deg = deg
                    best = v
            return best
        
        def backtrack(unassigned_set, domains, assignment, depth, deadline):
            if time.time() > deadline:
                return None
            
            if not unassigned_set:
                return assignment[:]
            
            v = mrv_select(unassigned_set, domains, popcount)
            if v == -1:
                return None
            
            unassigned_set.remove(v)
            
            # Get colors in domain
            mask = domains[v]
            # Iterate through bits set in mask
            # Colors 0..4
            for c in range(5):
                if mask & (1 << c):
                    assignment[v] = c
                    
                    # Forward check
                    failed = False
                    removed = [] # List of (neighbor, old_domain)
                    for u in adj[v]:
                        if u in unassigned_set:
                            old_dom = domains[u]
                            new_dom = old_dom & ~(1 << c)
                            if new_dom == 0:
                                failed = True
                                break
                            domains[u] = new_dom
                            removed.append((u, old_dom))
                            
                    if not failed:
                        result = backtrack(unassigned_set, domains, assignment, depth + 1, deadline)
                        if result is not None:
                            return result
                    
                    # Undo
                    for u, old_dom in removed:
                        domains[u] = old_dom
                        
            unassigned_set.add(v)
            assignment[v] = -1
            return None
        
        result = backtrack(set(range(n)), domains, assignment, 0, deadline)
        return result
