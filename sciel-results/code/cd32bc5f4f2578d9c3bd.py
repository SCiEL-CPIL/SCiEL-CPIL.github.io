"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    import time as _time

    # Set a deadline slightly before the 150s kill to ensure safe return
    deadline = _time.monotonic() + 140.0

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

    def solve_csp(n, edges, color_lists):
        """
        Solves the list coloring problem using backtracking with MRV and Forward Checking.
        Uses bitmasks for domains for speed.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute neighbor lists as tuples for faster iteration if needed, 
        # but lists are fine. 
        # Convert lists to bitmasks
        # Colors are 0..4, so we use 5 bits.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in color_lists[v]:
                mask |= (1 << c)
            domains[v] = mask
        
        assignment = [-1] * n
        # Track which vertices are assigned
        # We can use assignment[v] != -1 to check
        
        # To speed up MRV, we might need a way to quickly find min domain size.
        # Since n is small (< 500), a linear scan is acceptable.
        
        def backtrack(depth):
            if _time.monotonic() > deadline:
                return False
            
            # Find unassigned vertex with Minimum Remaining Values (MRV)
            min_v = -1
            min_size = 10  # Max possible size is 5 (or less)
            for v in range(n):
                if assignment[v] == -1:
                    # Count bits in domains[v]
                    # Python 3.10+ has int.bit_count(), but let's be safe or use bin().count()
                    # Actually, domains are small, we can just iterate or use a precomputed table for 0-31
                    # Let's use a helper or just bit_count if available. 
                    # Fallback: bin(domains[v]).count('1')
                    sz = bin(domains[v]).count('1')
                    if sz < min_size:
                        min_size = sz
                        min_v = v
                        if sz <= 1:
                            break
            
            if min_v == -1:
                return True # All assigned
            
            if min_size == 0:
                return False # Domain wipeout
            
            # Get colors for min_v
            d_mask = domains[min_v]
            # Iterate through set bits
            # We can extract colors:
            colors_to_try = []
            m = d_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                c = lsb.bit_length() - 1
                colors_to_try.append(c)
                m ^= lsb
            
            # Optional: Order colors by least constraining value (LCV)
            # For each color c, count how many neighbors have c in their domain.
            # Sort colors by this count (ascending).
            def lcv_score(c):
                count = 0
                for u in adj[min_v]:
                    if assignment[u] == -1:
                        if domains[u] & (1 << c):
                            count += 1
                return count
            
            colors_to_try.sort(key=lcv_score)
            
            for c in colors_to_try:
                assignment[min_v] = c
                c_bit = 1 << c
                
                # Forward Check: Remove c from neighbors' domains
                changes = []
                consistent = True
                for u in adj[min_v]:
                    if assignment[u] == -1:
                        old_dom = domains[u]
                        new_dom = old_dom & ~c_bit
                        if new_dom != old_dom:
                            if new_dom == 0:
                                # Wipeout
                                # Undo changes
                                for (uu, old_d) in changes:
                                    domains[uu] = old_d
                                consistent = False
                                break
                            changes.append((u, old_dom))
                            domains[u] = new_dom
                
                if consistent:
                    if backtrack(depth + 1):
                        return True
                
                # Undo changes
                for (u, old_d) in changes:
                    domains[u] = old_d
                assignment[min_v] = -1
            
            return False

        # Start solving
        if backtrack(0):
            return assignment[:]
        
        # If backtracking fails (should not happen for valid instances), 
        # return a dummy result to avoid crash, though it will score 0.
        # Or try a simple greedy as a last resort? 
        # Given the theorem, this should never be reached.
        return [color_lists[v][0] for v in range(n)]

    out = []
    instances_params = ((20, 701), (40, 702), (60, 703), (100, 704))
    
    for splits, seed in instances_params:
        if _time.monotonic() > deadline:
            # Not enough time, generate instance and return a trivial (likely invalid) coloring
            # to satisfy the interface, though it will score 0.
            n, edges, lists = instance(splits, seed)
            out.append([int(c) for c in lists])
            continue
            
        n, edges, lists = instance(splits, seed)
        result = solve_csp(n, edges, lists)
        
        # Ensure types are correct (Python int)
        # result contains ints from the solver.
        # Just in case, convert to list of ints.
        out.append([int(c) for c in result])

    return out
# EVOLVE-BLOCK-END
