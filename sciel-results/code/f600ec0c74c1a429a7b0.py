"""Model-and-optimize: parameterise the colouring (one int per vertex) and fit
each instance directly against the objective (list-respecting, edge-different
colouring) using a backtracking constraint solver with full Arc Consistency (MAC)."""
# EVOLVE-BLOCK-START
import time
import sys
import collections
from random import Random

sys.setrecursionlimit(10000)

def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 10.0          # safety margin before the cap (return early)

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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

    def solve_coloring(n, adj, lists):
        # Initial domains
        domains = [set(l) for l in lists]
        
        # AC-3 implementation
        def run_ac3(domains, adj, initial_queue=None):
            """
            Runs AC-3 on the domains.
            Returns True if consistent, False if a domain becomes empty.
            Modifies domains in-place.
            """
            if initial_queue is None:
                queue = collections.deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            else:
                queue = collections.deque(initial_queue)
            
            while queue:
                u, v = queue.popleft()
                # Check if any value in domains[v] is unsupported by domains[u]
                # Constraint: color[u] != color[v]
                # Value c in domains[v] is supported if there exists d in domains[u] such that d != c
                # This fails only if domains[u] is a subset of {c}, i.e., domains[u] == {c}
                
                # Optimization: If domains[u] has more than 1 element, it supports all c in domains[v] 
                # unless domains[v] contains elements that are the ONLY element in domains[u].
                # Actually, if len(domains[u]) > 1, then for any c, there is some d != c in domains[u].
                # So c is supported.
                # If len(domains[u]) == 1, let u_val = next(iter(domains[u])).
                # Then c is supported iff c != u_val.
                # So if len(domains[u]) == 1 and u_val in domains[v], we must remove u_val from domains[v].
                
                if len(domains[u]) == 1:
                    u_val = next(iter(domains[u]))
                    if u_val in domains[v]:
                        domains[v].discard(u_val)
                        if not domains[v]:
                            return False
                        # If domains[v] changed, add all neighbors of v to queue
                        for w in adj[v]:
                            if w != u:
                                queue.append((w, v))
                # If len(domains[u]) > 1, no values are removed from domains[v]
                # because for any c in domains[v], there is some d in domains[u] with d != c.
                
            return True

        # Check initial consistency
        if not run_ac3(domains, adj):
            return None
            
        # Check for empty domains
        for i in range(n):
            if not domains[i]:
                return None

        color = [-1] * n
        
        def backtrack():
            # Find unassigned vertex with MRV
            # Tie-breaker: highest degree
            min_size = n + 1
            best_v = -1
            best_deg = -1
            
            for i in range(n):
                if color[i] == -1:
                    sz = len(domains[i])
                    if sz == 0:
                        return False
                    if sz < min_size or (sz == min_size and len(adj[i]) > best_deg):
                        min_size = sz
                        best_v = i
                        best_deg = len(adj[i])
                        if sz == 1:
                            break # Can't get better than 1

            if best_v == -1:
                return True # All assigned

            v = best_v
            
            # LCV: Order values by how many values they remove from neighbors
            values = list(domains[v])
            if len(values) > 1:
                def lcv_score(val):
                    count = 0
                    for nb in adj[v]:
                        if color[nb] == -1 and val in domains[nb]:
                            count += 1
                    return count
                values.sort(key=lcV_score)

            for val in values:
                # Save state for undo (copy domains)
                # Since N is small, copying is fast and safe
                saved_domains = [set(d) for d in domains]
                
                # Assign
                color[v] = val
                domains[v] = {val}
                
                # Propagate AC-3
                # Start queue with neighbors of v
                init_queue = [(nb, v) for nb in adj[v] if color[nb] == -1]
                if run_ac3(domains, adj, init_queue):
                    if backtrack():
                        return True
                else:
                    # Propagation failed, restore and try next value
                    pass
                
                # Undo
                color[v] = -1
                domains = saved_domains

            return False

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        col = solve_coloring(n, adj, lists)
        
        if col is None:
            # Fallback: Should not happen, but prevent crash
            # Try a simple greedy with random restarts if MAC fails (unlikely)
            # Or just return dummy to satisfy structure, scoring 0 for this instance
            col = [lists[i][0] for i in range(n)]
            
        # Ensure types are Python int
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
