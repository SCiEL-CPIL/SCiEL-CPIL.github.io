"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)
    
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
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # color[v] = chosen color, -1 if unassigned
        color = [-1] * n
        # avail[v] = set of available colors for v
        avail = [set(lst) for lst in lists]
        
        # We process vertices in 'order'
        # To make backtracking efficient, we can just iterate through the order list
        # and maintain the state.
        
        # A simple recursive backtracking with forward checking
        # Since n is small (~400), recursion depth might be an issue if not careful, 
        # but we can use an iterative approach or just ensure recursion limit is high.
        # Given the structure, a simple DFS on the ordered vertices should work.
        
        # Let's use a standard backtracking approach.
        # We assign colors to vertices in 'order'.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(order):
                return True
            
            v = order[idx]
            
            # Try each available color for v
            # We can try colors in the order they appear in the list
            for c in sorted(avail[v]):
                # Check if c is valid (it is, by avail)
                # Assign c to v
                color[v] = c
                avail[v].discard(c) # Remove chosen color from avail? 
                                    # Actually, avail[v] is the set of choices. 
                                    # Let's keep avail[v] as the current domain.
                                    # When we assign v=c, we remove c from neighbors' domains.
                
                # Record changes to neighbors for backtracking
                changes = []
                valid = True
                for u in adj[v]:
                    if color[u] != -1:
                        # If neighbor is already assigned, check conflict
                        if color[u] == c:
                            valid = False
                            break
                    else:
                        # If neighbor is unassigned, remove c from its domain
                        if c in avail[u]:
                            avail[u].discard(c)
                            changes.append((u, c))
                            if not avail[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                # Undo changes
                for u, c_undone in changes:
                    avail[u].add(c_undone)
                color[v] = -1
                avail[v].add(c) # Restore v's domain to include c again? 
                                # Wait, avail[v] was the domain before assignment.
                                # We didn't remove c from avail[v] in the assignment step 
                                # except logically. But in the loop `for c in sorted(avail[v])`,
                                # avail[v] is iterated. If we modify avail[v] during iteration, 
                                # it's tricky.
                                
                # Let's restructure: 
                # avail[v] should represent the current domain of v.
                # When we assign v=c, we should not remove c from avail[v] immediately 
                # in a way that breaks the loop, but we must ensure we don't pick it again 
                # for this step. The loop handles that.
                # However, the neighbor updates are the critical part.
                
            return False

        # The above backtrack function has a subtle issue with avail[v] modification 
        # if we just discard c. Let's be more precise.
        # avail[v] is the set of colors v can take.
        # When we try color c for v:
        # 1. We temporarily set color[v] = c.
        # 2. We remove c from all unassigned neighbors' avail sets.
        # 3. If success, return True.
        # 4. If fail, restore neighbors' avail sets and color[v] = -1.
        
        # The loop `for c in sorted(avail[v])` iterates over the *current* avail[v].
        # If we don't modify avail[v] itself during the trial (only neighbors), 
        # then the loop is safe.
        
        # Let's rewrite backtrack to be cleaner.
        
        def backtrack_clean(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(order):
                return True
            
            v = order[idx]
            # Get a copy of the list to iterate over, in case avail[v] changes?
            # No, avail[v] doesn't change during the trial of v's colors, 
            # only neighbors' avail sets change.
            # So we can iterate over sorted(avail[v]).
            
            for c in sorted(avail[v]):
                # Check if c is still in avail[v] (it should be)
                if c not in avail[v]:
                    continue
                    
                color[v] = c
                # We don't remove c from avail[v] here, because if this branch fails,
                # we want to try other colors in avail[v].
                # However, logically v is assigned c.
                
                # Propagate to neighbors
                changes = []
                valid = True
                for u in adj[v]:
                    if color[u] != -1:
                        if color[u] == c:
                            valid = False
                            break
                    else:
                        if c in avail[u]:
                            avail[u].discard(c)
                            changes.append(u)
                            if not avail[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack_clean(idx + 1):
                        return True
                
                # Undo
                for u in changes:
                    avail[u].add(c)
                color[v] = -1
                
            return False

        if backtrack_clean(0):
            return [int(c) for c in color]
        else:
            # Should not happen if problem is solvable and time is sufficient
            # But if deadline is hit, we might return partial or invalid.
            # The task says "An instance whose colouring breaks a list or an edge scores 0".
            # So we should try to return *something* if we fail, or just return None?
            # The interface expects a list of ints.
            # If we fail, we can return the first list's colors? No, that's invalid.
            # Given the guarantee, we assume success.
            return [0] * n

    start_time = time.time()
    # 150s limit, use 140s for safety
    deadline = start_time + 140.0
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists, deadline)
        out.append(coloring)
        # Update deadline for next instance? 
        # The deadline is absolute.
        
    return out
# EVOLVE-BLOCK-END
