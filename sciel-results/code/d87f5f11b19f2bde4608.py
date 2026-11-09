"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    import time as _time

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

    def solve_list_coloring(n, edges, lists, time_limit=25.0):
        """Solve list coloring via backtracking with forward checking and AC-3."""
        start = _time.time()
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(l) for l in lists]
        
        # AC-3: enforce arc consistency
        def ac3(domains, adj):
            from collections import deque
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            while queue:
                u, v = queue.popleft()
                # Check if dom(u) \ dom(v) has any value with no support in dom(v)
                # Actually for coloring: for each c in dom(u), if c in dom(v), 
                # then u and v can't both be c. So we need: for c in dom(u),
                # there must exist some neighbor w != v such that c is not forced...
                # Actually for simple list coloring: arc (u,v) is consistent if
                # for every c in dom(u), there is no constraint that c must equal dom(v).
                # In coloring, the constraint is c_u != c_v. So arc (u,v) is consistent
                # if dom(u) is not a subset of {c : c in dom(v) and c is the only option for v}
                # Actually it's simpler: arc (u,v) is consistent if for every c in dom(u),
                # it's not the case that c is in dom(v) AND v has only c as option... 
                # No wait. For coloring: u and v must have different colors.
                # Arc (u,v) is consistent if: for every c in dom(u), there exists d in dom(v) 
                # with d != c. This fails only if dom(v) = {c} for some c in dom(u).
                # So: if dom(v) is a singleton {c} and c in dom(u), remove c from dom(u).
                if len(domains[v]) == 1:
                    c = next(iter(domains[v]))
                    if c in domains[u]:
                        domains[u].discard(c)
                        if not domains[u]:
                            return False
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
                elif len(domains[u]) == 1:
                    c = next(iter(domains[u]))
                    if c in domains[v]:
                        domains[v].discard(c)
                        if not domains[v]:
                            return False
                        for w in adj[v]:
                            if w != u:
                                queue.append((w, v))
            return True
        
        if not ac3(domains, adj):
            return None
        
        # Check if all domains are singletons
        if all(len(d) == 1 for d in domains):
            return [next(iter(d)) for d in domains]
        
        # Variable ordering: most constrained (smallest domain), then highest degree
        def var_order(domains, adj):
            order = []
            for v in range(n):
                if len(domains[v]) > 1:
                    order.append(v)
            order.sort(key=lambda v: (len(domains[v]), -len(adj[v])))
            return order
        
        # Backtracking with forward checking
        def backtrack(domains, adj, assignment):
            # Find unassigned variable with smallest domain
            best = -1
            best_size = 10
            for v in range(n):
                if assignment[v] is None:
                    sz = len(domains[v])
                    if sz == 0:
                        return False
                    if sz < best_size or (sz == best_size and best != -1 and len(adj[v]) > len(adj[best])):
                        best = v
                        best_size = sz
            if best == -1:
                return True
            
            # Try each color
            for c in sorted(domains[best]):
                # Assign
                assignment[best] = c
                domains[best] = {c}
                
                # Forward check: propagate
                new_domains = [set(d) for d in domains]
                new_domains[best] = {c}
                
                # Propagate to neighbors
                from collections import deque
                changed = True
                while changed:
                    changed = False
                    for u in range(n):
                        if assignment[u] is not None:
                            continue
                        for v in adj[u]:
                            if assignment[v] is not None:
                                if assignment[v] in new_domains[u]:
                                    new_domains[u].discard(assignment[v])
                                    if not new_domains[u]:
                                        return False
                                    changed = True
                            elif len(new_domains[v]) == 1:
                                cv = next(iter(new_domains[v]))
                                if cv in new_domains[u]:
                                    new_domains[u].discard(cv)
                                    if not new_domains[u]:
                                        return False
                                    changed = True
                
                if backtrack(new_domains, adj, assignment):
                    return True
                
                # Undo
                assignment[best] = None
                domains[best] = {c}  # will be overwritten
            return False
        
        # Use iterative deepening or just straight backtracking
        assignment = [None] * n
        # Pre-assign singletons
        for v in range(n):
            if len(domains[v]) == 1:
                assignment[v] = next(iter(domains[v]))
        
        # Try backtracking
        if backtrack(domains, adj, assignment):
            # Verify
            for v in range(n):
                if assignment[v] is None or assignment[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if u > v and assignment[u] == assignment[v]:
                        return None
            return assignment
        return None

    def solve_with_restart(n, edges, lists, time_limit=25.0):
        """Try multiple restarts with different random orderings."""
        from random import Random
        start = _time.time()
        
        # First try deterministic (degree-based)
        result = solve_list_coloring(n, edges, lists, time_limit)
        if result is not None:
            return result
        
        elapsed = _time.time() - start
        if elapsed > time_limit * 0.6:
            return None
        
        # Try with randomized tie-breaking
        for attempt in range(10):
            if _time.time() - start > time_limit:
                break
            # Shuffle lists slightly to break symmetry
            rng = Random(attempt * 1000 + 7)
            shuffled_lists = []
            for lst in lists:
                s = list(lst)
                rng.shuffle(s)
                shuffled_lists.append(sorted(s))
            result = solve_list_coloring(n, edges, shuffled_lists, max(1.0, time_limit - (_time.time() - start)))
            if result is not None:
                return result
        return None

    def solve_instance(splits, seed, time_budget):
        n, edges, lists = instance(splits, seed)
        return solve_with_restart(n, edges, lists, time_budget)

    def verify(n, edges, lists, colour):
        if not isinstance(colour, (list, tuple)):
            return False
        if len(colour) != n:
            return False
        for i, c in enumerate(colour):
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    out = []
    start_time = _time.time()
    total_time = 110.0  # safety margin from 120s
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        elapsed = _time.time() - start_time
        remaining = total_time - elapsed
        if remaining < 2.0:
            # Not enough time, return what we have
            out.append(None)
            continue
        
        # Allocate time: more for larger instances
        # Instance sizes: 88, 168, 248, 408 vertices
        # Give proportional time but with minimum
        if splits == 20:
            budget = min(remaining * 0.15, 8.0)
        elif splits == 40:
            budget = min(remaining * 0.20, 12.0)
        elif splits == 60:
            budget = min(remaining * 0.25, 16.0)
        else:
            budget = min(remaining * 0.40, 25.0)
        
        budget = max(budget, 1.0)
        result = solve_instance(splits, seed, budget)
        
        if result is not None:
            # Convert to ensure proper types
            colour = [int(c) for c in result]
            out.append(colour)
        else:
            out.append(None)
    
    # If any instance failed, try to use remaining time on the hardest ones
    failed = [i for i in range(4) if out[i] is None]
    if failed:
        elapsed = _time.time() - start_time
        remaining = total_time - elapsed
        if remaining > 1.0:
            # Retry failed instances with remaining time
            splits_list = [20, 40, 60, 100]
            seed_list = [701, 702, 703, 704]
            for i in failed:
                if _time.time() - start_time > total_time - 1.0:
                    break
                budget = min(remaining / len(failed), 10.0)
                if budget < 0.5:
                    break
                result = solve_instance(splits_list[i], seed_list[i], budget)
                if result is not None:
                    colour = [int(c) for c in result]
                    out[i] = colour
    
    # Final: replace any None with a valid (but possibly wrong) coloring to avoid invalid output
    # Actually, if we can't solve, we should still return something of correct type
    # But the task says wrong colouring scores 0 for that instance, not invalid
    # So we need to return int colours in range. If we can't solve, return a dummy.
    for i in range(4):
        if out[i] is None:
            # Generate a dummy coloring (will score 0 for this instance)
            n, edges, lists = instance([20, 40, 60, 100][i], [701, 702, 703, 704][i])
            out[i] = [lists[v][0] for v in range(n)]
    
    # Ensure all are lists of Python ints
    final_out = []
    for colour in out:
        final_out.append([int(c) for c in colour])
    
    return final_out
# EVOLVE-BLOCK-END
