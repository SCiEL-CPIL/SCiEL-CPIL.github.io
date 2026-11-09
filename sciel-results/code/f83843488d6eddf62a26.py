"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Increase recursion limit to be safe for backtracking depth
    sys.setrecursionlimit(5000)
    
    from random import Random
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for faster operations
        # Bit i set means color i is available
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        colour = [-1] * n
        uncolored = set(range(n))
        
        # Cache for popcount? Python 3.8+ has int.bit_count()
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Pick vertex with fewest available colors
            min_count = 5
            target = -1
            for v in uncolored:
                # domains[v] is the current available mask
                count = domains[v].bit_count()
                if count < min_count:
                    min_count = count
                    target = v
                    if count == 0:
                        return False
                    if count == 1:
                        break
            
            if target == -1:
                return False

            v = target
            uncolored.remove(v)
            
            # Iterate through available colors
            mask = domains[v]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                # Assign color c
                colour[v] = c
                
                # Forward checking: update neighbors' domains
                # We need to remember which bits were removed to restore them
                removed_bits = []
                valid = True
                for u in adj[v]:
                    if u in uncolored:
                        if domains[u] & (1 << c):
                            removed_bits.append((u, 1 << c))
                            new_domain = domains[u] ^ (1 << c)
                            domains[u] = new_domain
                            if new_domain == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                else:
                    # Undo neighbor changes before trying next color
                    for u, bit in removed_bits:
                        domains[u] |= bit
                    # If we broke out of neighbor loop early due to 0 domain,
                    # we must still restore all removed bits.
                    # Wait, the loop above breaks immediately. 
                    # We need to ensure restoration happens in all cases if we assigned c.
                    # Actually, if valid is False, we undo and continue.
                    # If valid is True and backtrack fails, we undo and continue.
                    
                # Undo neighbor changes
                for u, bit in removed_bits:
                    domains[u] |= bit
                # Undo color assignment
                colour[v] = -1
            
            # Restore vertex to uncolored set if all colors failed
            uncolored.add(v)
            return False

        if backtrack():
            return colour
        else:
            # Fallback: return a list of 0s (invalid coloring, scores 0 for this instance)
            # But must be valid type and length.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we are running out of time, fill remaining with dummy solutions
            # to ensure we return a valid structure (list of 4 lists of ints)
            # though they will score 0.
            while len(out) < 4:
                 out.append([0])
            break
        
        n, edges, lists = instance(splits, seed)
        # Ensure lists are lists of ints
        sol = solve_instance(n, edges, lists)
        # Convert to Python ints just in case
        # Ensure they are plain ints, not numpy or bool
        out.append([int(c) for c in sol])
        
    # If we exited early due to deadline, ensure we have 4 entries
    while len(out) < 4:
        out.append([0])
        
    return out
# EVOLVE-BLOCK-END
