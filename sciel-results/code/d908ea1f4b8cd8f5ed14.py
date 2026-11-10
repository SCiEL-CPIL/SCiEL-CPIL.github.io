"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize domains
        domains = [set(lst) for lst in lists]

        # AC-3 Arc Consistency
        # Queue of arcs (x, y) where x and y are adjacent
        queue = []
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))

        def revise(x, y):
            """
            Revise domain of x based on constraint with y.
            Returns True if domain of x was revised, False otherwise.
            """
            revised = False
            dx = domains[x]
            dy = domains[y]
            # Remove colors from dx that are not compatible with any color in dy
            # Since constraint is x != y, a color c in dx is compatible if there exists d in dy such that c != d.
            # This is only false if dy == {c}.
            # So if c in dx and c in dy and len(dy) == 1, remove c from dx.
            # More generally: keep c in dx if there exists d in dy such that c != d.
            # This fails only if dy is a subset of {c}. Since dy is a set, this means dy == {c}.
            
            # Optimization: If len(dy) > 1, then for any c, there is always some d in dy with d != c.
            # So only if len(dy) == 1, we must check if c == the only element in dy.
            if len(dy) == 1:
                only_y = next(iter(dy))
                if only_y in dx:
                    dx.remove(only_y)
                    revised = True
            # If len(dy) > 1, no revision needed for != constraint.
            # If len(dy) == 0, it's already unsatisfiable, handled elsewhere or will be caught.
            
            return revised

        # Run AC-3
        while queue:
            x, y = queue.pop(0)
            if revise(x, y):
                if not domains[x]:
                    return None # Unsatisfiable
                # Add all arcs (z, x) where z is a neighbor of x, z != y
                for z in adj[x]:
                    if z != y:
                        queue.append((z, x))

        # Check if any domain is empty
        for i in range(n):
            if not domains[i]:
                return None

        # Backtracking Search with Forward Checking and MRV
        assignment = [None] * n
        
        def backtrack(assignment, domains):
            # Check time limit
            if time.time() > deadline:
                return None

            # Find unassigned variable with MRV
            unassigned = [i for i in range(n) if assignment[i] is None]
            if not unassigned:
                return assignment[:] # Success

            # MRV: Minimum Remaining Values
            # Tie-breaker: Degree (most constrained)
            min_size = float('inf')
            var = -1
            for u in unassigned:
                size = len(domains[u])
                if size < min_size:
                    min_size = size
                    var = u
                elif size == min_size:
                    # Tie-breaker: Higher degree
                    if len(adj[u]) > len(adj[var]):
                        var = u
            
            if var == -1:
                return None

            # Order values: Least Constraining Value (LCV) or just sorted
            # For simplicity and speed, we can just iterate through sorted values
            # Or LCV: choose value that rules out fewest choices for neighbors
            values = sorted(domains[var])
            
            for val in values:
                # Check if val is still in domain (should be, but safe)
                if val not in domains[var]:
                    continue
                
                # Forward Checking: check if assigning var=val makes any neighbor's domain empty
                consistent = True
                # Save old domains for neighbors to restore if needed? 
                # Or just simulate. Since domains are sets, we can create a copy or track changes.
                # Given small graph size, copying domains for neighbors is cheap.
                # But we are modifying domains in place in AC-3. 
                # In backtracking, we should not permanently modify domains unless we revert.
                # Let's use a recursive approach that passes modified domains or reverts changes.
                
                # Simpler: Check consistency without modifying, then assign and recurse.
                # But forward checking requires updating domains of neighbors.
                
                # Let's revert changes after recursion.
                changes = []
                for neighbor in adj[var]:
                    if assignment[neighbor] is not None:
                        # Already assigned, just check constraint
                        if assignment[neighbor] == val:
                            consistent = False
                            break
                    else:
                        # Unassigned, check if removing val from neighbor's domain empties it
                        if val in domains[neighbor]:
                            # We need to see if neighbor has other options
                            # If len(domains[neighbor]) == 1 and domains[neighbor] == {val}, then it becomes empty
                            if len(domains[neighbor]) == 1:
                                consistent = False
                                break
                            # Otherwise, we would remove val from domains[neighbor]
                            # For forward checking, we actually need to propagate this removal.
                            # Let's do full forward checking by modifying domains and reverting.
                            pass
                
                if not consistent:
                    continue
                
                # Perform Forward Checking properly
                # This requires modifying domains of neighbors and reverting
                # Let's implement a helper or inline logic
                
                # Assign var = val
                assignment[var] = val
                
                # Apply forward checking
                fc_changes = [] # list of (neighbor, old_domain_size) or similar
                fc_valid = True
                for neighbor in adj[var]:
                    if assignment[neighbor] is None:
                        if val in domains[neighbor]:
                            # Remove val from neighbor's domain
                            domains[neighbor].remove(val)
                            fc_changes.append(neighbor)
                            if not domains[neighbor]:
                                fc_valid = False
                                break
                
                if fc_valid:
                    # Recurse
                    result = backtrack(assignment, domains)
                    if result is not None:
                        return result
                
                # Revert forward checking changes
                for neighbor in fc_changes:
                    domains[neighbor].add(val)
                
                # Unassign var
                assignment[var] = None

            return None

        return backtrack(assignment, domains)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_csp(n, edges, lists)
        if colouring is None:
            # Fallback: shouldn't happen, but if it does, return empty list or invalid
            # The problem states all are colourable, so this is a safety net.
            # If we can't find one, we might try a simpler greedy as last resort?
            # No, better to return None and let it fail than return invalid.
            # But the task says "Maximize; all are colourable".
            # If our solver fails, we score 0 for this instance.
            # Let's just append None for now, but the return type must be list of ints.
            # If we return None, it might break the "list of ints" check if not handled.
            # We must return a list of ints.
            # If solver fails, we can't produce a valid coloring.
            # We will just return an empty list or a list of zeros? 
            # "An instance whose colouring breaks a list or an edge scores 0 for that instance"
            # So returning an invalid coloring is okay, it just scores 0 for that instance.
            # But "wrong length or type makes the whole result invalid".
            # So we must return a list of length n with ints.
            # If we can't solve, we return [0]*n, which will likely be invalid, scoring 0 for that instance.
            out.append([0] * n)
        else:
            # Ensure types are Python int
            out.append([int(c) for c in colouring])
            
    return out
# EVOLVE-BLOCK-END
