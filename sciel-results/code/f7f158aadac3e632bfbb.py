import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback result
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger n
    for n in range(15, 9, -1):
        if time.time() > deadline:
            break
        
        # Precompute all C6s for this n
        # A C6 is a sequence of 6 distinct vertices (v0, ..., v5) such that 
        # edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) exist.
        # In K_n, all edges exist. We just need the set of 6 vertices to form a cycle.
        # Actually, any 6 distinct vertices in K_n contain many C6s.
        # We need to check if the specific cycle is monochromatic.
        # The number of distinct C6 subgraphs (cycles) in K_n is n! / (6 * 2) ? No.
        # Number of cyclic permutations of 6 vertices is (6-1)! / 2 = 60.
        # Total C6s = C(n, 6) * 60.
        # For n=15, C(15,6) = 5005. 5005 * 60 = 300,300.
        # For n=17, C(17,6) = 12376. 12376 * 60 = 742,560.
        
        # We represent a C6 by the tuple of 6 vertices in order.
        # To avoid duplicates, we fix v0 to be the smallest vertex in the cycle.
        # And we fix the direction (v1 < v5 or similar? No, just enumerate cycles).
        # Simpler: Iterate over all combinations of 6 vertices.
        # For each combination, there are 60 distinct cycles.
        
        cycles = []
        cycles_by_edge = {}
        
        # Map edge (i,j) to index
        def edge_idx(i, j):
            if i > j: i, j = j, i
            # i < j
            return i * n + j # Unique enough for n<=20? 20*20=400. Yes.
            
        # Precompute cycles
        # This might take a second or two for n=17.
        if n <= 17:
            for combo in itertools.combinations(range(n), 6):
                # Generate all 60 cycles from these 6 vertices
                # Fix v0 = min(combo)
                v0 = min(combo)
                rest = [v for v in combo if v != v0]
                # Permutations of rest
                for perm in itertools.permutations(rest):
                    # Cycle: v0, p0, p1, p2, p3, p4, v0
                    # To avoid reverse duplicates, we can enforce p0 < p4? 
                    # Or just generate all and deduplicate? 
                    # Better: generate all 120 permutations, divide by 2 for direction?
                    # Let's just generate all cycles and store them.
                    # 5! = 120 permutations.
                    # Cycle: (v0, p[0], p[1], p[2], p[3], p[4])
                    # This is a specific cycle.
                    # The reverse cycle (v0, p[4], p[3], p[2], p[1], p[0]) is also a cycle.
                    # Are they the same set of edges? Yes.
                    # So we should count each undirected cycle once.
                    # We can enforce p[0] < p[4] to halve the count.
                    if perm[0] < perm[4]:
                        cycle = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
                        cycles.append(cycle)
                        # Add to cycles_by_edge
                        edges_in_cycle = [
                            (cycle[0], cycle[1]),
                            (cycle[1], cycle[2]),
                            (cycle[2], cycle[3]),
                            (cycle[3], cycle[4]),
                            (cycle[4], cycle[5]),
                            (cycle[5], cycle[0])
                        ]
                        for u, v in edges_in_cycle:
                            idx = edge_idx(u, v)
                            if idx not in cycles_by_edge:
                                cycles_by_edge[idx] = []
                            cycles_by_edge[idx].append(len(cycles) - 1)
        
        if not cycles:
            continue
            
        num_cycles = len(cycles)
        
        # Local Search
        # State: colors array of length n*n (only i<j used)
        # We use a flat array for colors, indexed by i*n + j (i<j)
        colors = [0] * (n * n)
        # Initialize randomly
        for i in range(n):
            for j in range(i + 1, n):
                colors[i * n + j] = random.randint(0, 3)
                
        # Count monochromatic C6s
        mono_count = 0
        for c in cycles:
            e0 = edge_idx(c[0], c[1])
            e1 = edge_idx(c[1], c[2])
            e2 = edge_idx(c[2], c[3])
            e3 = edge_idx(c[3], c[4])
            e4 = edge_idx(c[4], c[5])
            e5 = edge_idx(c[5], c[0])
            col = colors[e0]
            if (colors[e1] == col and colors[e2] == col and 
                colors[e3] == col and colors[e4] == col and colors[e5] == col):
                mono_count += 1
                
        best_mono = mono_count
        best_state = colors[:]
        
        search_deadline = time.time() + 20 # 20s per n
        
        while time.time() < search_deadline and mono_count > 0:
            # Pick a random edge to change
            # Focus on edges in monochromatic cycles?
            # For simplicity, pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(i + 1, n - 1)
            idx = i * n + j
            
            old_color = colors[idx]
            if old_color == 0:
                new_color = random.randint(1, 3)
            else:
                new_color = random.randint(0, old_color - 1)
                
            # Update mono_count
            # Check cycles containing this edge
            affected_cycles = cycles_by_edge.get(idx, [])
            delta = 0
            for c_idx in affected_cycles:
                c = cycles[c_idx]
                # Get all edges in this cycle
                edges_in_cycle = [
                    edge_idx(c[0], c[1]),
                    edge_idx(c[1], c[2]),
                    edge_idx(c[2], c[3]),
                    edge_idx(c[3], c[4]),
                    edge_idx(c[4], c[5]),
                    edge_idx(c[5], c[0])
                ]
                
                # Check if it was monochromatic with old_color
                # And if it becomes monochromatic with new_color
                # Note: Only this edge changed.
                
                # Before change:
                was_mono = True
                for e in edges_in_cycle:
                    if e == idx:
                        continue
                    if colors[e] != old_color:
                        was_mono = False
                        break
                # Since colors[idx] was old_color, if was_mono, then it was mono.
                
                # After change:
                is_mono = True
                for e in edges_in_cycle:
                    if e == idx:
                        continue
                    if colors[e] != new_color:
                        is_mono = False
                        break
                # If is_mono, then with new_color it is mono.
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
                # If was_mono and is_mono (impossible if old!=new and other edges match both? 
                # No, other edges must match old_color AND new_color? Impossible if old!=new.
                # Unless the cycle has only 1 edge? No.
                # So was_mono and is_mono is False.
                
            mono_count += delta
            colors[idx] = new_color
            
            if mono_count < best_mono:
                best_mono = mono_count
                best_state = colors[:]
                if mono_count == 0:
                    break
                    
            # If stuck, random restart?
            # For now, just keep going.
            
        if best_mono == 0:
            # Extract colors in lexicographic order
            final_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    final_colors.append(best_state[i * n + j])
            return {'n': n, 'colors': final_colors}
            
    # Fallback to seed
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
