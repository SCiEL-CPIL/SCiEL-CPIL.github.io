import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

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
        lists = [set(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for faster operations
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        colors = [-1] * n
        
        # Use built-in bit_count for speed (Python 3.10+)
        # Fallback if necessary, but assuming modern Python
        def popcount(x):
            return x.bit_count()

        # AC-3 implementation for initial propagation
        def ac3():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop()
                du = domains[u]
                if du == 0:
                    return False
                
                dv = domains[v]
                if dv == 0:
                    return False

                # If dv is a single color, remove that color from du
                if dv & (dv - 1) == 0:
                    c = dv.bit_length() - 1
                    mask_c = 1 << c
                    if du & mask_c:
                        domains[u] = du ^ mask_c
                        if domains[u] == 0:
                            return False
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
            
            return True

        # Backtracking with MRV, Forward Checking, and Unit Propagation
        # We use a stack of (node, color_mask) to handle unit propagation recursively/iteratively
        # To keep it simple and fast, we will do a recursive step that handles unit propagation.
        
        def assign_and_propagate(v, c_mask):
            """
            Assigns color c (from c_mask) to v and propagates.
            Returns a list of (node, old_domain) to undo, or None if conflict.
            """
            undo = []
            # Assign v
            colors[v] = c_mask.bit_length() - 1
            old_v_domain = domains[v]
            domains[v] = c_mask
            undo.append((v, old_v_domain))
            
            # Propagate to neighbors
            # We need to check if any neighbor is forced to a single value
            # If so, we must assign it immediately.
            # We use a stack for this propagation to avoid deep recursion limits if chains are long
            stack = [(v, c_mask)] # (source, source_color_mask)
            
            while stack:
                src, src_mask = stack.pop()
                for u in adj[src]:
                    if colors[u] != -1:
                        # Already assigned, check for conflict
                        if domains[u] == src_mask: # Actually domains[u] is the assigned color mask
                            # If they are the same, conflict
                            # Note: domains[u] holds the assigned color mask if colors[u] != -1
                            # But we should check if src_mask is in u's domain? No, u is assigned.
                            # If u is assigned to color X, and src is assigned to color Y.
                            # If X == Y, conflict.
                            # domains[u] is the mask of the assigned color.
                            # domains[src] is the mask of the assigned color.
                            if domains[u] == src_mask:
                                # Undo everything
                                for node, old_dom in undo:
                                    colors[node] = -1
                                    domains[node] = old_dom
                                return None
                        continue
                    
                    old_u_domain = domains[u]
                    new_u_domain = old_u_domain & ~src_mask
                    if new_u_domain == 0:
                        # Conflict
                        for node, old_dom in undo:
                            colors[node] = -1
                            domains[node] = old_dom
                        return None
                    
                    domains[u] = new_u_domain
                    undo.append((u, old_u_domain))
                    
                    # Check if u is now a singleton (Unit Propagation)
                    if new_u_domain & (new_u_domain - 1) == 0:
                        # u is forced to a single color
                        stack.append((u, new_u_domain))
            
            return undo

        def backtrack():
            if time.time() > deadline:
                return False
            
            # MRV Heuristic
            best_v = -1
            min_size = n + 1
            best_degree = -1
            
            for v in range(n):
                if colors[v] == -1:
                    size = popcount(domains[v])
                    if size == 0:
                        return False
                    deg = len(adj[v])
                    if size < min_size or (size == min_size and deg > best_degree):
                        min_size = size
                        best_degree = deg
                        best_v = v
                        if size == 1:
                            # If size is 1, we can break early as it's the minimum possible
                            # But we should check if there are other size 1s? 
                            # MRV picks the first one found with min size. 
                            # Breaking on size 1 is safe and fast.
                            break
            
            if best_v == -1:
                return True
            
            # Try each color in the domain of best_v
            d = domains[best_v]
            while d:
                c_mask = d & (-d)
                d ^= c_mask
                
                undo = assign_and_propagate(best_v, c_mask)
                if undo is not None:
                    if backtrack():
                        return True
                    else:
                        # Undo the assignment and propagation
                        for node, old_dom in undo:
                            colors[node] = -1
                            domains[node] = old_dom
                # If undo is None, conflict occurred, undo already done inside assign_and_propagate
            
            return False

        if not ac3():
            return [0] * n

        if backtrack():
            return [int(c) for c in colors]
        else:
            return [0] * n

    out = []
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Emergency exit: return dummy values to avoid timeout, though score will be 0
            # We need n to match. Generate instance to get n.
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue

        n, edges, lists = instance(splits, seed)
        
        # If we are very close to deadline, use a very fast heuristic (greedy)
        if time.time() > deadline - 5.0:
            # Simple greedy
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colors = [-1] * n
            for v in range(n):
                taken = {colors[u] for u in adj[v] if colors[u] != -1}
                for c in sorted(lists[v]):
                    if c not in taken:
                        colors[v] = c
                        break
            out.append([int(c) for c in colors])
            continue

        res = solve(n, edges, lists)
        
        # Ensure types are strictly Python int
        res = [int(x) for x in res]
        
        # Validate length
        if len(res) != n:
            res = [0] * n
            
        out.append(res)

    return out
