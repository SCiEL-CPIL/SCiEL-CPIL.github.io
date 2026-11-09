"""Rewritten program using a bitmask CSP solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 10s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
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
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for fast operations
        # Colors are 0..4, so we can use bits 0..4
        init_masks = []
        for l in lists:
            mask = 0
            for c in l:
                mask |= (1 << c)
            init_masks.append(mask)
            
        color = [-1] * n
        avail = init_masks[:]
        
        # MRV Heuristic with Backtracking
        def backtrack():
            if time.time() > DEADLINE:
                return False, False # Timeout
            
            # Find unassigned vertex with min available colors (MRV)
            min_size = 5 # Max palette size
            min_v = -1
            for v in range(n):
                if color[v] == -1:
                    # Count bits in mask
                    size = bin(avail[v]).count('1')
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False, True # Failure
                        if size == 1:
                            break # Best case for MRV
            
            if min_v == -1:
                return True, True # All assigned
            
            # Try each color in the mask for min_v
            mask = avail[min_v]
            while mask:
                # Get lowest set bit
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit # Remove bit for next iteration
                
                # Assign c to min_v
                color[min_v] = c
                old_avail_min_v = avail[min_v]
                avail[min_v] = 0 # Mark as assigned (or just use color array)
                
                # Propagate: remove c from neighbors' avail
                valid = True
                removed_neighbors = []
                for nb in adj[min_v]:
                    if color[nb] == -1:
                        if avail[nb] & c_bit:
                            avail[nb] ^= c_bit
                            removed_neighbors.append(nb)
                            if avail[nb] == 0:
                                valid = False
                                break
                
                if valid:
                    success, timed_out = backtrack()
                    if success:
                        return True, True
                    if timed_out:
                        # Restore state
                        color[min_v] = -1
                        avail[min_v] = old_avail_min_v
                        for nb in removed_neighbors:
                            avail[nb] |= c_bit
                        return False, True
                else:
                    # Restore state for failed branch
                    color[min_v] = -1
                    avail[min_v] = old_avail_min_v
                    for nb in removed_neighbors:
                        avail[nb] |= c_bit
            
            # All colors failed
            return False, True

        success, timed_out = backtrack()
        
        if success:
            # Ensure types are correct: int
            return [int(c) for c in color]
        else:
            # Fallback: return a list of 0s (invalid coloring, scores 0 for this instance)
            # This ensures the return type is correct (list of ints)
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Check time again
        if time.time() > DEADLINE:
             out.append([0] * n)
             continue
             
        sol = solve_instance(n, edges, lists)
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
