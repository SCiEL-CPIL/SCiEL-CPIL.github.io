"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import sys as _sys
_sys.setrecursionlimit(10000)

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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Use a more efficient CSP solver with trail-based undo
        # Represent available colors as bitmasks for speed
        # Colors are 0..4, so we need 5 bits
        # available[v] is a bitmask of available colors
        
        # Initialize available as bitmasks
        available = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            available[v] = mask
        
        color = [-1] * n
        unassigned = [True] * n
        num_unassigned = n
        
        # Trail for undo: list of (vertex, old_mask)
        trail = []
        
        def assign(v, c):
            """Assign color c to vertex v, return list of changes for undo"""
            nonlocal num_unassigned
            changes = []
            # Remove c from all unassigned neighbors
            for u in adj[v]:
                if unassigned[u]:
                    if available[u] & (1 << c):
                        changes.append((u, available[u]))
                        available[u] &= ~(1 << c)
                        if available[u] == 0:
                            return None  # Conflict
            # Mark v as assigned
            changes.append((v, available[v]))
            available[v] = 0
            unassigned[v] = False
            num_unassigned -= 1
            return changes
        
        def undo(changes):
            """Undo changes from a failed assignment"""
            nonlocal num_unassigned
            for v, old_mask in reversed(changes):
                available[v] = old_mask
                if old_mask != 0 and not unassigned[v]:
                    # This was an assignment, so it becomes unassigned again
                    unassigned[v] = True
                    num_unassigned += 1
        
        def backtrack():
            nonlocal num_unassigned
            if num_unassigned == 0:
                return True
            
            # Check time
            if time.time() > deadline:
                return False
            
            # Find vertex with minimum remaining values (MRV)
            min_v = -1
            min_size = 5  # Max possible is 3, but use 5 as inf
            for v in range(n):
                if unassigned[v]:
                    sz = bin(available[v]).count('1')
                    if sz < min_size:
                        min_size = sz
                        min_v = v
                        if sz <= 1:
                            break
            
            if min_v == -1 or min_size == 0:
                return False
            
            # Try each color in min_v's available set
            mask = available[min_v]
            while mask:
                # Get lowest set bit
                c = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1  # Clear lowest set bit
                
                changes = assign(min_v, c)
                if changes is not None:
                    color[min_v] = c
                    if backtrack():
                        return True
                    undo(changes)
                    color[min_v] = -1
                # If changes is None, it was a conflict, nothing to undo
                # But we need to check if it was a conflict due to neighbor becoming empty
                # In that case, we should undo the changes that were made before the conflict
                # Actually, assign() returns None on conflict but may have made partial changes
                # Let me fix this...
            
            return False
        
        # Fix the assign function to properly handle partial changes on conflict
        def assign_safe(v, c):
            """Assign color c to vertex v, return list of ALL changes made (for full undo)"""
            nonlocal num_unassigned
            changes = []
            # Remove c from all unassigned neighbors
            for u in adj[v]:
                if unassigned[u]:
                    if available[u] & (1 << c):
                        changes.append((u, available[u]))
                        available[u] &= ~(1 << c)
                        if available[u] == 0:
                            # Conflict: need to undo all changes made so far
                            for uu, old_mask in reversed(changes):
                                available[uu] = old_mask
                            return None
            # Mark v as assigned
            changes.append((v, available[v]))
            available[v] = 0
            unassigned[v] = False
            num_unassigned -= 1
            return changes
        
        def backtrack2():
            nonlocal num_unassigned
            if num_unassigned == 0:
                return True
            
            # Check time
            if time.time() > deadline:
                return False
            
            # Find vertex with minimum remaining values (MRV)
            min_v = -1
            min_size = 5  # Max possible is 3, but use 5 as inf
            for v in range(n):
                if unassigned[v]:
                    sz = bin(available[v]).count('1')
                    if sz < min_size:
                        min_size = sz
                        min_v = v
                        if sz <= 1:
                            break
            
            if min_v == -1 or min_size == 0:
                return False
            
            # Try each color in min_v's available set
            mask = available[min_v]
            while mask:
                # Get lowest set bit
                c = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1  # Clear lowest set bit
                
                changes = assign_safe(min_v, c)
                if changes is not None:
                    color[min_v] = c
                    if backtrack2():
                        return True
                    undo(changes)
                    color[min_v] = -1
            
            return False
        
        # Use backtrack2 which handles conflicts properly
        if backtrack2():
            return color
        else:
            # Fallback to greedy if backtracking fails or times out
            deg = [len(adj[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: -deg[x])
            greedy_color = [-1] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if greedy_color[u] != -1:
                        taken.add(greedy_color[u])
                for c in lists[v]:
                    if c not in taken:
                        greedy_color[v] = c
                        break
                if greedy_color[v] == -1:
                    greedy_color[v] = lists[v][0]  # Fallback, might be invalid
            return greedy_color

    out = []
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check if we have enough time for this instance
        remaining = deadline - time.time()
        if remaining < 5.0:
            # Not enough time, use fast greedy for remaining instances
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            deg = [len(adj[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: -deg[x])
            greedy_color = [-1] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if greedy_color[u] != -1:
                        taken.add(greedy_color[u])
                for c in lists[v]:
                    if c not in taken:
                        greedy_color[v] = c
                        break
                if greedy_color[v] == -1:
                    greedy_color[v] = lists[v][0]
            final_res = [int(c) for c in greedy_color]
            out.append(final_res)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Solve
        res = solve_instance(n, edges, lists, deadline)
        
        # Convert to list of Python ints
        final_res = []
        for c in res:
            if c is None or c == -1:
                final_res.append(0)  # Arbitrary, likely invalid
            else:
                final_res.append(int(c))
                
        out.append(final_res)
    
    return out

# EVOLVE-BLOCK-END
