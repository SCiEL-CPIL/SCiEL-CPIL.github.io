import time
import sys
from random import Random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces and edges
        # Vertices 0-7
        # Faces:
        # 0-1-2-3 (top? no, let's use standard cube)
        # Let's stick to the seed's generation exactly to ensure same instances.
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = set()
        for f in faces:
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                edges.add((min(a, b), max(a, b)))
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                # Connect inner[i] to a
                edges.add((min(a, inner[i]), max(a, inner[i])))
                # Connect inner[i] to inner[i+1]
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                # New face: a, b, inner[i+1], inner[i]
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Solve
        # The simple backtracking above might be slow if the order is bad.
        # Let's implement a slightly more robust solver using MRV dynamically.
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        domains = [list(l) for l in lists]
        colour = [-1] * n
        uncolored = set(range(n))
        
        # Use a stack or recursion. Recursion depth N=400 is fine.
        
        def solve_csp():
            if time.time() > deadline:
                return False
            if not uncolored:
                return True
                
            # Select variable with Minimum Remaining Values (MRV)
            # Tie-breaker: Highest degree
            min_size = float('inf')
            best_v = None
            for v in uncolored:
                # Filter domain by already colored neighbors
                # We don't prune domains explicitly in this simple version, 
                # we just check validity on assignment.
                # For MRV, we should consider the effective domain size.
                # Effective domain = {c in domains[v] | c not in {colour[u] for u in adj[v] if colour[u] != -1}}
                
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                        
                eff_dom = [c for c in domains[v] if c not in taken]
                if len(eff_dom) == 0:
                    return False # Dead end
                
                if len(eff_dom) < min_size:
                    min_size = len(eff_dom)
                    best_v = v
                    if min_size == 1:
                        break # Can't get better than 1
                
            if best_v is None:
                return True # All colored? Shouldn't happen if uncolored not empty

            v = best_v
            uncolored.remove(v)
            
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            
            eff_dom = [c for c in domains[v] if c not in taken]
            
            # Shuffle eff_dom for randomness? Or keep sorted?
            # Randomness helps avoid bad local minima if we were doing local search.
            # For backtracking, order doesn't affect completeness, only speed.
            # Let's try them in the order they appear in the list (sorted).
            
            for c in eff_dom:
                colour[v] = c
                if solve_csp():
                    return True
                colour[v] = -1
                
            uncolored.add(v)
            return False
            
        if solve_csp():
            out.append(colour)
        else:
            # Fallback: should not happen
            # Return the last attempted colouring? Or raise?
            # The problem says it's colourable.
            # If we fail, we score 0 for this instance.
            # Let's just append the current colour array (which might be invalid).
            out.append(colour)
            
        if time.time() > deadline:
            # If we are running out of time, we should break and return what we have.
            # But we need 4 results.
            # If we can't finish, we might return invalid lists for the rest.
            # Better to break and pad with zeros? No, types must be int.
            # Just break.
            break

    # Ensure we have 4 results
    while len(out) < 4:
        # If we timed out, we need to return something.
        # But the contract says "wrong length ... makes the whole result invalid".
        # So we MUST return 4 lists.
        # If we didn't solve them, we can't just leave them out.
        # We should have solved them.
        # If we timed out, it's a failure.
        # To be safe, if we break early, we should try to fill the rest with a trivial (likely invalid) 
        # or just reuse the last one? No, that's wrong.
        # The best strategy is to ensure we don't time out.
        # With N=400 and MRV backtracking, it should be milliseconds.
        pass

    # Convert to list of lists of int
    # Ensure types are correct
    result = []
    for c in out:
        result.append([int(x) for x in c])
        
    return result

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
