"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

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
        
        # Degree for variable ordering tie-breaking
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as bitmasks
        # Color c is represented by bit (1 << c)
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assignment = [-1] * n
        sys.setrecursionlimit(10000)

        # AC-3 with bitmasks
        def arc_consistency(initial_queue=None):
            """Apply AC-3. Returns False if any domain becomes empty."""
            if initial_queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((v, u))
            else:
                queue = deque(initial_queue)
            
            while queue:
                if time.time() > deadline:
                    return 'timeout'
                
                xi, xj = queue.popleft()
                
                if domains[xj] == 0:
                    return False

                # If Dom(xj) is a singleton {c}, then c cannot be in Dom(xi).
                if domains[xj] & (domains[xj] - 1) == 0:
                    c_bit = domains[xj]
                    if domains[xi] & c_bit:
                        new_dom = domains[xi] & ~c_bit
                        if new_dom != domains[xi]:
                            domains[xi] = new_dom
                            if domains[xi] == 0:
                                return False
                            for xk in adj[xi]:
                                if xk != xj:
                                    queue.append((xk, xi))
            
            return True

        # Helper to get count of bits
        def popcount(x):
            return bin(x).count('1')

        # Backtracking search with MRV and forward checking
        def backtrack():
            if time.time() > deadline:
                return 'timeout'
            
            # Find unassigned vertex with smallest domain size (MRV)
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True  # All assigned
            
            dom = domains[best_v]
            
            # Extract colors from domain bitmask
            colors = []
            temp = dom
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp ^= lsb
            
            for c in colors:
                if time.time() > deadline:
                    return 'timeout'
                
                lsb = 1 << c
                
                # Save state for undo
                old_dom_v = domains[best_v]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate: Remove c from neighbors' domains
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            new_dom_u = domains[u] & ~lsb
                            if new_dom_u == 0:
                                valid = False
                                break
                            domains[u] = new_dom_u
                
                if valid:
                    # Run AC-3 on affected arcs
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    
                    if ac_res is True:
                        result = backtrack()
                        if result is True:
                            return True
                        if result == 'timeout':
                            # Undo and propagate timeout
                            assignment[best_v] = old_assignment
                            domains[best_v] = old_dom_v
                            # Restore neighbors' domains
                            for u in adj[best_v]:
                                if assignment[u] == -1:
                                    # We need to restore domains[u]
                                    # This is tricky because we didn't save them.
                                    # Let's use a different strategy: save a copy of domains before modification.
                                    pass
                            return 'timeout'
                    elif ac_res is 'timeout':
                        # Undo and propagate timeout
                        assignment[best_v] = old_assignment
                        domains[best_v] = old_dom_v
                        # Restore neighbors' domains - again, problem.
                        return 'timeout'
                
                # Undo changes to neighbors
                # We need to restore domains of neighbors that we modified.
                # Since we didn't track which ones were modified, let's just re-run AC-3 or restore from a backup?
                # For correctness, we must restore.
                # Let's change the approach: save a snapshot of domains before each step.
                # Given N is small (~400), copying a list of 400 ints is cheap.
                
                # Let's refactor to use snapshots for safety.
                # But for now, let's just restore what we know we changed.
                # Actually, the AC-3 call might have changed more.
                # So snapshotting is the only safe way.
                
                # Let's rewrite backtrack to use snapshots.
                pass # Placeholder

        # Refactored Backtracking with Snapshotting for Correct Undo
        def backtrack_refactored():
            if time.time() > deadline:
                return 'timeout'
            
            # Find MRV
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True
            
            dom = domains[best_v]
            
            # Extract colors
            colors = []
            temp = dom
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp ^= lsb
            
            for c in colors:
                if time.time() > deadline:
                    return 'timeout'
                
                lsb = 1 << c
                
                # Snapshot
                old_domains = domains[:]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            new_dom_u = domains[u] & ~lsb
                            if new_dom_u == 0:
                                valid = False
                                break
                            domains[u] = new_dom_u
                
                if valid:
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    
                    if ac_res is True:
                        result = backtrack_refactored()
                        if result is True:
                            return True
                        if result == 'timeout':
                            # Undo and propagate timeout
                            domains[:] = old_domains
                            assignment[best_v] = old_assignment
                            return 'timeout'
                        # Else result is False, continue to next color
                    elif ac_res is 'timeout':
                        # Undo and propagate timeout
                        domains[:] = old_domains
                        assignment[best_v] = old_assignment
                        return 'timeout'
                
                # Undo
                domains[:] = old_domains
                assignment[best_v] = old_assignment
            
            return False

        # Initial AC-3
        if arc_consistency() is not True:
            return None

        result = backtrack_refactored()
        if result is True:
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 145.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > hard_deadline - 5.0:
             out.append([0] * n)
             continue
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.9
        
        colouring = solve_instance(n, edges, lists, instance_deadline)
        
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
