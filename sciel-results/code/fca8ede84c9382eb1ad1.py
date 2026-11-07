"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

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

    def solve_instance_wrapper(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for fast operations
        # Colors are 0..4, so we need 5 bits.
        # mask bit i is 1 if color i is allowed.
        init_masks = [0] * n
        for v in range(n):
            m = 0
            for c in lists[v]:
                m |= (1 << c)
            init_masks[v] = m
            
        # State:
        # colors[v] = assigned color (-1 if unassigned)
        # masks[v] = current available colors (bitmask)
        colors = [-1] * n
        masks = init_masks[:]
        
        # Heap for MRV (Minimum Remaining Values)
        # (domain_size, vertex)
        import heapq
        heap = []
        for v in range(n):
            # Count bits in mask
            size = bin(masks[v]).count('1')
            heapq.heappush(heap, (size, v))
            
        def pop_valid_vertex():
            while heap:
                size, v = heapq.heappop(heap)
                if colors[v] != -1:
                    continue
                # Check if size is still accurate (might have changed)
                cur_size = bin(masks[v]).count('1')
                if size != cur_size:
                    heapq.heappush(heap, (cur_size, v))
                    continue
                return v
            return None

        def backtrack():
            v = pop_valid_vertex()
            if v is None:
                # All vertices assigned
                return True
            
            # Try each color in the mask
            m = masks[v]
            # Iterate through set bits
            while m:
                # Extract lowest set bit
                lsb = m & -m
                c = lsb.bit_length() - 1
                m ^= lsb
                
                # Check if color c is still in mask (it is, by construction)
                # Assign color c to v
                colors[v] = c
                old_mask_v = masks[v]
                masks[v] = lsb
                
                # Forward Checking: remove c from neighbors
                valid = True
                changes = [] # (neighbor, old_mask)
                for u in adj[v]:
                    if colors[u] == -1:
                        if masks[u] & lsb:
                            old_mu = masks[u]
                            new_mu = masks[u] & ~lsb
                            if new_mu == 0:
                                valid = False
                                # Undo changes so far
                                for (nu, omu) in changes:
                                    masks[nu] = omu
                                break
                            masks[u] = new_mu
                            changes.append((u, old_mu))
                            # Update heap for u
                            new_size = bin(new_mu).count('1')
                            heapq.heappush(heap, (new_size, u))
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                colors[v] = -1
                masks[v] = old_mask_v
                for (u, old_mu) in changes:
                    masks[u] = old_mu
        
        if backtrack():
            return colors
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance_wrapper(n, edges, lists)
        if coloring is None:
            # Fallback to greedy (should not happen if solver is correct)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            coloring = colour
        out.append(coloring)
    return out

# EVOLVE-BLOCK-END
