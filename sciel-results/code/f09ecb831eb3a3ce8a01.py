"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
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

    def solve_local_search(n, edges, lists, time_limit=25.0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build list membership check
        list_set = [set(lists[v]) for v in range(n)]
        
        start_time = time.time()
        
        def timed_out():
            return time.time() - start_time > time_limit
        
        # Initialize: assign each vertex a random color from its list
        rng = Random(42)
        
        def random_init():
            colour = [0] * n
            for v in range(n):
                colour[v] = rng.choice(lists[v])
            return colour
        
        def count_violations(colour):
            """Count edge violations (adjacent same color) + list violations."""
            violations = 0
            for v in range(n):
                if colour[v] not in list_set[v]:
                    violations += 1000  # heavy penalty for list violation
            for u, v in edges:
                if colour[u] == colour[v]:
                    violations += 1
            return violations
        
        def get_violated_neighbors(colour, v):
            """Return set of neighbors that have same color as v."""
            result = set()
            for u in adj[v]:
                if colour[u] == colour[v]:
                    result.add(u)
            return result
        
        def local_search_attempt():
            if timed_out():
                return None
            
            colour = random_init()
            
            # Repair phase: fix list violations first
            for v in range(n):
                if colour[v] not in list_set[v]:
                    colour[v] = rng.choice(lists[v])
            
            best_colour = colour[:]
            best_violations = count_violations(best_colour)
            if best_violations == 0:
                return best_colour
            
            max_iterations = 50000
            for it in range(max_iterations):
                if timed_out():
                    return None
                
                violations = count_violations(colour)
                if violations == 0:
                    return colour[:]
                
                if violations < best_violations:
                    best_violations = violations
                    best_colour = colour[:]
                
                # Pick a violated vertex
                # Find vertices with violations
                violated_vertices = []
                for v in range(n):
                    if colour[v] not in list_set[v]:
                        violated_vertices.append(v)
                        continue
                    for u in adj[v]:
                        if u > v and colour[u] == colour[v]:
                            violated_vertices.append(v)
                            break
                
                if not violated_vertices:
                    return colour[:]
                
                v = rng.choice(violated_vertices)
                
                # Try all colors in the list for v
                # Prefer colors that reduce violations
                best_c = colour[v]
                best_delta = 0
                for c in lists[v]:
                    if c == colour[v]:
                        continue
                    # Calculate delta in violations if we change v to c
                    delta = 0
                    # Check if new color creates violations with neighbors
                    for u in adj[v]:
                        if colour[u] == c:
                            delta += 1
                    # Old color's violations with neighbors
                    for u in adj[v]:
                        if colour[u] == colour[v]:
                            delta -= 1
                    # List violation change
                    if colour[v] not in list_set[v]:
                        delta -= 1000
                    if c not in list_set[v]:
                        delta += 1000
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                
                colour[v] = best_c
            
            # Return best found
            if best_violations == 0:
                return best_colour
            return None
        
        # Multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if timed_out():
                break
            result = local_search_attempt()
            if result is not None:
                # Verify
                for v in range(n):
                    if result[v] not in lists[v]:
                        return None
                    for u in adj[v]:
                        if u > v and result[u] == result[v]:
                            return None
                return [int(c) for c in result]
        
        return None

    def greedy_color(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colour = [-1] * n
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = int(free[0])
            else:
                colour[v] = int(lists[v][0])
        
        return colour

    def solve_backtrack(n, edges, lists, time_limit=20.0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        adj_set = [set(adj[i]) for i in range(n)]
        colour = [-1] * n
        domains = [set(lists[v]) for v in range(n)]
        
        start_time = time.time()
        
        def is_timed_out():
            return time.time() - start_time > time_limit
        
        def backtrack():
            if is_timed_out():
                return 'timeout'
            
            best_v = -1
            best_count = n + 1
            for v in range(n):
                if colour[v] != -1:
                    continue
                count = len(domains[v])
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_v = v
                    if count == 1:
                        break
            
            if best_v == -1:
                return True
            
            for c in sorted(domains[best_v]):
                colour[best_v] = c
                domains[best_v].discard(c)
                
                undo = []
                valid = True
                for u in adj_set[best_v]:
                    if colour[u] != -1:
                        if colour[u] == c:
                            valid = False
                            break
                        continue
                    if c in domains[u]:
                        domains[u].discard(c)
                        undo.append((u, c))
                        if len(domains[u]) == 0:
                            valid = False
                            break
                
                if valid:
                    result = backtrack()
                    if result is True:
                        return True
                    if result == 'timeout':
                        for (u, cc) in undo:
                            domains[u].add(cc)
                        colour[best_v] = -1
                        domains[best_v].add(c)
                        return 'timeout'
                
                for (u, cc) in undo:
                    domains[u].add(cc)
                colour[best_v] = -1
                domains[best_v].add(c)
            
            return False
        
        result = backtrack()
        if result is True:
            for v in range(n):
                if colour[v] not in lists[v]:
                    return None
                for u in adj_set[v]:
                    if u > v and colour[u] == colour[v]:
                        return None
            return [int(colour[v]) for v in range(n)]
        return None

    out = []
    start_time = time.time()
    deadline = 140.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            n, edges, lists = instance(splits, seed)
            colour = greedy_color(n, edges, lists)
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        
        remaining = deadline - time.time() - 3.0
        time_limit = max(2.0, min(25.0, remaining / 2.0))
        
        # Try local search first (different paradigm, often faster)
        colour = solve_local_search(n, edges, lists, time_limit=time_limit)
        
        if colour is None:
            # Try backtracking
            remaining2 = deadline - time.time() - 2.0
            bt_limit = max(1.0, min(20.0, remaining2))
            colour = solve_backtrack(n, edges, lists, time_limit=bt_limit)
        
        if colour is None:
            colour = greedy_color(n, edges, lists)
        
        colour = [int(c) for c in colour]
        out.append(colour)
    
    result = []
    for colour in out:
        result.append([int(c) for c in colour])
    
    return result
# EVOLVE-BLOCK-END
