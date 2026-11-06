"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256
    # Build conflict graph: edge (i,j) if i+j is a perfect square (including i=j)
    # We want max independent set in this graph.
    
    # Precompute squares up to 512 (max sum = 256+256=512)
    squares = set()
    for k in range(1, 23):  # 22^2 = 484, 23^2 = 529 > 512
        squares.add(k * k)
    
    # Build adjacency: conflict[i] = set of j such that i+j is a square
    # Using 0-indexed: element i+1 corresponds to index i
    conflict = [set() for _ in range(N)]
    for i in range(N):
        a = i + 1
        for j in range(N):
            b = j + 1
            if a + b in squares:
                conflict[i].add(j)
    
    # Convert to bitsets for fast operations
    # Use Python integers as bitsets
    conflict_bits = [0] * N
    for i in range(N):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        conflict_bits[i] = bits
    
    # Greedy coloring for upper bound (DSATUR-like)
    # For branch and bound: color the subgraph to get upper bound on independent set size
    
    def greedy_color_upper_bound(vertices, current_depth):
        """
        vertices: list of vertex indices still to consider
        Returns an upper bound on the size of independent set from these vertices.
        Uses greedy coloring: color the induced subgraph, number of colors is upper bound.
        """
        if not vertices:
            return 0
        # Simple greedy coloring on the conflict subgraph induced by vertices
        # Order vertices by degree in the subgraph (descending) for tighter bound
        verts_set = set(vertices)
        degs = []
        for v in vertices:
            deg = len(conflict_bits[v] & ((1 << N) - 1))  # not quite right, let me fix
            # Actually compute degree within vertices
            mask = 0
            for u in vertices:
                if u != v:
                    if (1 << u) & conflict_bits[v]:
                        mask |= (1 << u)
            degs.append((len(bin(mask).count('1')), v))
        
        degs.sort(reverse=True)
        
        # Greedy coloring
        color_of = {}
        num_colors = 0
        for _, v in degs:
            used_colors = set()
            for u in vertices:
                if u != v and (1 << u) & conflict_bits[v]:
                    if u in color_of:
                        used_colors.add(color_of[u])
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors
    
    # Better: use a proper branch-and-bound with greedy coloring upper bound
    # We'll use a recursive B&B where at each node we have a set of candidates
    # and a current independent set. Upper bound = |current_set| + greedy_color_bound(remaining)
    
    best_set = []
    best_size = 0
    global_deadline = deadline
    
    def is_deadline():
        return time.time() > global_deadline
    
    # Initial greedy solution
    def greedy_is():
        """Greedy: repeatedly pick vertex with fewest conflicts among remaining."""
        remaining = set(range(N))
        result = []
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (1 << u) & conflict_bits[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            result.append(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v
            to_remove = set()
            for u in remaining:
                if (1 << u) & conflict_bits[best_v]:
                    to_remove.add(u)
            remaining -= to_remove
        return result
    
    initial = greedy_is()
    best_set = initial
    best_size = len(initial)
    
    # Branch and bound
    # State: (candidates_bitset, current_set)
    # candidates: bitmask of vertices still available
    # We branch: include or exclude a chosen vertex
    
    def branch_and_bound(candidates_mask, current_set, depth=0):
        nonlocal best_set, best_size, global_deadline
        
        if is_deadline():
            return
        
        # Upper bound: |current_set| + greedy coloring bound on candidates
        # For speed, use a simpler bound first: just count of candidates
        # candidates_count = bin(candidates_mask).count('1')
        # if len(current_set) + candidates_count <= best_size:
        #     return
        
        # Extract candidate list
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Greedy coloring upper bound
        # For efficiency, use a quick bound: the number of vertices
        # If that's not enough to prune, do coloring
        quick_bound = len(candidates)
        if len(current_set) + quick_bound <= best_size:
            return
        
        # Do greedy coloring for tighter bound
        # Order by degree in candidate subgraph
        cand_set = set(candidates)
        deg_list = []
        for v in candidates:
            deg = 0
            m2 = candidates_mask
            # Count neighbors of v in candidates
            nb_mask = conflict_bits[v] & candidates_mask
            deg = bin(nb_mask).count('1')
            deg_list.append((deg, v))
        
        deg_list.sort(reverse=True)
        
        # Greedy coloring
        color_of = {}
        num_colors = 0
        for _, v in deg_list:
            used_colors = set()
            nb_mask = conflict_bits[v] & candidates_mask
            # Iterate over neighbors in candidates
            m3 = nb_mask
            while m3:
                lsb2 = m3 & (-m3)
                u = lsb2.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                m3 ^= lsb2
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            num_colors = max(num_colors, c + 1)
        
        if len(current_set) + num_colors <= best_size:
            return
        
        # Choose branching vertex: pick one with highest degree (most constraining)
        # Or use a heuristic: pick vertex that appears in most color classes
        branch_v = deg_list[0][1]
        
        # Branch 1: include branch_v
        new_set = current_set + [branch_v]
        # Remove branch_v and its neighbors from candidates
        new_mask = candidates_mask & ~(1 << branch_v) & ~conflict_bits[branch_v]
        branch_and_bound(new_mask, new_set, depth + 1)
        
        if is_deadline():
            return
        
        # Branch 2: exclude branch_v
        new_mask2 = candidates_mask & ~(1 << branch_v)
        branch_and_bound(new_mask2, current_set, depth + 1)
    
    # Run B&B
    full_mask = (1 << N) - 1
    branch_and_bound(full_mask, [])
    
    if is_deadline():
        return [v + 1 for v in best_set]
    
    # Local search polish: try to add more vertices or swap
    # Try to add any vertex not in best_set that doesn't conflict
    best_set_set = set(best_set)
    improved = True
    while improved and not is_deadline():
        improved = False
        for v in range(N):
            if v in best_set_set:
                continue
            # Check if v conflicts with any in best_set
            ok = True
            for u in best_set_set:
                if (1 << u) & conflict_bits[v]:
                    ok = False
                    break
            if ok:
                best_set_set.add(v)
                improved = True
                break
    
    # Try swap-based local search
    best_set_list = list(best_set_set)
    improved = True
    while improved and not is_deadline():
        improved = False
        for i, v in enumerate(best_set_list):
            for u in range(N):
                if u in best_set_set:
                    continue
                # Try swapping v with u
                # Check if u conflicts with any other in set (excluding v)
                ok = True
                for w in best_set_set:
                    if w == v:
                        continue
                    if (1 << w) & conflict_bits[u]:
                        ok = False
                        break
                if ok:
                    best_set_set.remove(v)
                    best_set_set.add(u)
                    improved = True
                    break
            if improved:
                break
        best_set_list = list(best_set_set)
    
    # Multiple restarts with different greedy orderings
    if not is_deadline():
        import random
        rng = random.Random(42)
        for restart in range(50):
            if is_deadline():
                break
            # Randomized greedy
            remaining = list(range(N))
            rng.shuffle(remaining)
            # But use degree-based ordering with random tie-breaking
            result = []
            rem_set = set(remaining)
            while rem_set and not is_deadline():
                best_v = None
                best_score = float('inf')
                for v in rem_set:
                    deg = bin(conflict_bits[v] & ((1 << N) - 1)).count('1')
                    # Actually compute degree in rem_set
                    deg2 = 0
                    m = conflict_bits[v]
                    # Count bits in m that are in rem_set
                    # For speed, just count all conflicts
                    deg2 = bin(conflict_bits[v]).count('1')
                    score = (deg2, rng.random())
                    if score < best_score:
                        best_score = score
                        best_v = v
                if best_v is None:
                    break
                result.append(best_v)
                rem_set.discard(best_v)
                # Remove neighbors
                to_remove = set()
                for u in rem_set:
                    if (1 << u) & conflict_bits[best_v]:
                        to_remove.add(u)
                rem_set -= to_remove
            
            if len(result) > best_size:
                best_size = len(result)
                best_set_set = set(result)
                # Polish
                improved2 = True
                while improved2 and not is_deadline():
                    improved2 = False
                    for v in range(N):
                        if v in best_set_set:
                            continue
                        ok = True
                        for u in best_set_set:
                            if (1 << u) & conflict_bits[v]:
                                ok = False
                                break
                        if ok:
                            best_set_set.add(v)
                            improved2 = True
                            break
    
    result = [v + 1 for v in sorted(best_set_set)]
    return result if result else [1]
# EVOLVE-BLOCK-END
