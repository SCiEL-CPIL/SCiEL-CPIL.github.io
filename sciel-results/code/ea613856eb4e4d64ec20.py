"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys
    from random import Random
    from collections import deque

    # Keep the exact generator under the name instance
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
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute list masks and initial available masks
        # Palette is 0..4, so we can use 5-bit integers
        list_masks = []
        for lst in lists:
            mask = 0
            for c in lst:
                mask |= (1 << c)
            list_masks.append(mask)
        
        # initial_avail[v] is the mask of colours allowed by list[v]
        # We will maintain avail[v] which is list_mask[v] & ~used_neighbours_mask[v]
        avail = list_masks[:]
        colour = [-1] * n
        uncoloured = set(range(n))
        
        # For efficient MRV, we can just iterate uncoloured.
        # Since n is small (max 8 + 4*100 = 408), O(n) scan is fine.
        
        def check_deadline():
            return time.time() < deadline

        def backtrack():
            if not check_deadline():
                return False

            # Find uncoloured vertex with MRV
            best_v = -1
            best_count = float('inf')
            # Iterate over a copy of uncoloured to allow removal? No, we just find min.
            for v in uncoloured:
                # avail[v] is already updated by forward checking
                count = bin(avail[v]).count('1')
                if count < best_count:
                    best_count = count
                    best_v = v
                    if best_count == 0:
                        return False
                    if best_count == 1:
                        break

            if best_v == -1:
                return True  # All coloured

            # Get available colours for best_v
            mask = avail[best_v]
            # Iterate through set bits
            # We need to try each colour
            # Unpack bits
            c = 0
            temp_mask = mask
            while temp_mask:
                if temp_mask & 1:
                    # Try colour c
                    if not check_deadline():
                        return False
                    
                    colour[best_v] = c
                    uncoloured.remove(best_v)
                    
                    # Forward checking
                    valid = True
                    undo_stack = []
                    for u in adj[best_v]:
                        if u in uncoloured:
                            # u is uncoloured, update its avail
                            old_avail = avail[u]
                            new_avail = old_avail & ~(1 << c)
                            if new_avail == 0:
                                valid = False
                                break
                            avail[u] = new_avail
                            undo_stack.append((u, old_avail))
                    
                    if valid:
                        if backtrack():
                            return True
                    else:
                        # Restore if valid was false due to break? 
                        # Actually, if valid is false, we didn't recurse, but we might have modified avail for some neighbours before breaking.
                        # We must restore them.
                        pass 
                    
                    # Restore state
                    uncoloured.add(best_v)
                    colour[best_v] = -1
                    for u, old_a in undo_stack:
                        avail[u] = old_a

                c += 1
                temp_mask >>= 1

            return False

        if backtrack():
            return colour
        return None

    deadline = time.time() + 110.0  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is None:
            # Fallback: return a default (shouldn't happen per Alon-Tarsi)
            result = [lists[v][0] for v in range(n)]
        out.append(result)

    # Ensure all values are Python int and lists are list
    final_out = []
    for coloring in out:
        final_out.append([int(c) for c in coloring])

    return final_out
# EVOLVE-BLOCK-END
