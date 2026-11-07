import time

def construct():
    from random import Random
    import sys

    # Increase recursion limit just in case, though depth is small
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces
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

    def solve(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        # Using lists for domains might be faster for small domains (size 3) than sets
        # But we need to remove colors. Sets are better for removal.
        # However, since domain size is always 3 initially, and shrinks, 
        # we can use bitmasks for extremely fast operations.
        # Palette is 0..4, so 5 bits needed.
        
        # Bitmask representation of domains
        # bit i is set if color i is available
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
            
        color = [-1] * n
        
        # Helper to count bits
        def popcount(x):
            return bin(x).count('1')
            
        # Helper to get list of available colors from mask
        def get_colors(mask):
            cols = []
            for c in range(5):
                if mask & (1 << c):
                    cols.append(c)
            return cols

        # MRV + Degree tie-breaker
        # To speed up MRV, we can just iterate all unassigned vertices.
        # n is at most 408, so O(n) per step is fine.
        
        unassigned = set(range(n))
        
        def backtrack():
            if not unassigned:
                return True
            
            # Check time limit
            if time.time() > deadline:
                return False

            # MRV: Find vertex with smallest domain size
            # Tie-breaker: Highest degree
            best_v = -1
            min_size = 5
            max_deg = -1
            
            for v in unassigned:
                size = popcount(domains[v])
                if size < min_size:
                    min_size = size
                    best_v = v
                    max_deg = len(adj[v])
                    if size == 0:
                        return False
                elif size == min_size:
                    if len(adj[v]) > max_deg:
                        max_deg = len(adj[v])
                        best_v = v
                        
            if best_v == -1:
                return False
                
            # Try colors for best_v
            v = best_v
            cols = get_colors(domains[v])
            
            for c in cols:
                color[v] = c
                unassigned.remove(v)
                
                # Forward Checking
                removed_info = []
                failed = False
                for u in adj[v]:
                    if u in unassigned:
                        # Check if c is in domain of u
                        if domains[u] & (1 << c):
                            # Remove c from u
                            domains[u] &= ~(1 << c)
                            removed_info.append(u)
                            if domains[u] == 0:
                                failed = True
                                break
                
                if not failed:
                    if backtrack():
                        return True
                else:
                    # Undo failed propagation before trying next color? 
                    # Actually, we need to undo regardless if we didn't return True.
                    pass
                
                # Undo
                color[v] = -1
                unassigned.add(v)
                for u in removed_info:
                    # Restore c to u
                    domains[u] |= (1 << c)
                    
            return False

        if backtrack():
            # Ensure all are assigned
            if all(c != -1 for c in color):
                return color
        return None

    out = []
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s
    deadline = start_time + 140
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid?
            # The contract says "Any broken rule makes the whole result invalid".
            # So we should try to solve as many as possible.
            # If we break early, we might return fewer than 4 lists, which is invalid.
            # However, the current code just breaks. This would result in a list of length < 4.
            # The prompt says: "Return a list of four lists... A wrong length... makes the whole result invalid."
            # So we MUST return 4 lists.
            # If we can't solve one, what do we do?
            # The problem states "all are colourable".
            # If we time out, we fail. But with n~400, it shouldn't time out.
            # Let's just solve them.
            pass
            
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists, deadline)
        
        if sol is None:
            # Fallback: This should not happen. 
            # If it does, we can't return a valid coloring.
            # But we must return a list of 4 lists.
            # We'll return a dummy list to keep structure, but it will score 0 for that instance.
            # Actually, if we return wrong length, whole thing is invalid.
            # So we MUST return a list of length n.
            # Let's just return [0]*n as a placeholder, knowing it's likely invalid.
            sol = [0] * n
            
        # Ensure types are Python int
        out.append([int(c) for c in sol])
        
    return out
