import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback valid coloring: n=5
    # Any coloring of K5 is valid because C6 requires 6 vertices.
    # Let's just use all 0s.
    def get_fallback(n):
        # For n=5, 10 edges.
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    # Target n values to try in descending order of ambition
    # We will try to find a valid coloring for these n.
    # If we find one, we return it immediately (as it's better than smaller n).
    # We allocate time to each.
    
    targets = [12, 11, 10]
    time_per_target = 30.0
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        # Precompute C6s for this n
        # A C6 is a set of 6 distinct vertices.
        # We need to check if the edges in the cycle are all the same color.
        # Cycle: v0-v1-v2-v3-v4-v5-v0
        # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
        
        # Generate all 6-combinations of vertices
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            # Map (u,v) to index 0..m-1
            # u < v
            # index = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
            # Or simpler: precomputed map
            return edge_map[(u, v)]
        
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        import itertools
        for combo in itertools.combinations(vertices, 6):
            # There are (6-1)!/2 = 60 distinct cycles on these 6 vertices
            # We can generate them by permuting the 6 vertices and taking the first 6
            # But many permutations are redundant.
            # Standard way: fix v0. Permute v1..v5.
            # v0 is fixed.
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                # Cycle: v0, p0, p1, p2, p3, p4, v0
                # This generates each cycle twice (forward and reverse) if we don't care about direction?
                # The task says "six-cycle". Usually undirected.
                # If we treat (v0,p0,p1,p2,p3,p4) and (v0,p4,p3,p2,p1,p0) as the same cycle,
                # we have 60 unique cycles.
                # The number of permutations of 5 elements is 120.
                # So each undirected cycle is counted twice.
                # We can just use all 120 permutations, it's fine for counting "monochromatic" property.
                # If a cycle is monochromatic, both directions are monochromatic.
                # We just need to know if AT LEAST one C6 exists.
                # So we can just iterate over all 120 permutations for each 6-combo.
                # Total checks: C(n,6) * 120.
                # For n=10: 210 * 120 = 25200.
                # For n=12: 924 * 120 = 110880.
                # This is manageable.
                
                path = [v0] + list(perm)
                # Edges: (path[0], path[1]), ..., (path[4], path[5]), (path[5], path[0])
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v))
                cycles.append(cycle_edges)
                
                # Map edge to cycles
                for e in cycle_edges:
                    edge_to_cycles[e].append(len(cycles) - 1)
        
        # Initialize random coloring
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Function to count monochromatic C6s
        # We can compute this from scratch initially, then update incrementally.
        # For simplicity and speed in Python, let's compute from scratch for each move?
        # No, that's O(NumCycles * 6). NumCycles ~ 100k. 600k ops per move.
        # 1000 moves -> 600M ops. Might be too slow in Python.
        
        # Better: Maintain a count `bad_count`.
        # When changing color of edge e from c_old to c_new:
        # 1. For each cycle containing e:
        #    - Check if it was monochromatic with c_old. If yes, decrement bad_count.
        #    - Check if it will be monochromatic with c_new. If yes, increment bad_count.
        
        # Helper: check if a cycle is monochromatic given the current colors
        # This requires checking 6 edges.
        
        def is_mono(cycle_edges, target_color):
            for e in cycle_edges:
                if colors[e] != target_color:
                    return False
            return True
        
        # Initial count
        bad_count = 0
        for ce in cycles:
            # Check if all 6 edges have the same color
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce[1:]):
                bad_count += 1
        
        # Local Search
        # Parameters
        max_iter = 20000
        # We will use a simple hill climbing with random restarts or just long run.
        # Given the landscape, simple hill climbing might get stuck.
        # Let's use Simulated Annealing.
        
        T = 5.0
        cool_rate = 0.99
        
        current_colors = colors[:]
        current_bad = bad_count
        
        best_colors = current_colors[:]
        best_bad = current_bad
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = current_colors[e]
            
            # Pick new color different from old
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            delta = 0
            # Iterate through cycles containing e
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                # Check if cycle was mono with c_old
                # We know edge e has color c_old.
                # Check other 5 edges.
                was_mono = True
                for ed in ce:
                    if ed != e and current_colors[ed] != c_old:
                        was_mono = False
                        break
                
                # Check if cycle will be mono with c_new
                # We know edge e will have color c_new.
                # Check other 5 edges.
                will_be_mono = True
                for ed in ce:
                    if ed != e and current_colors[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            # Acceptance
            if delta < 0:
                # Always accept improvement
                current_colors[e] = c_new
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = current_colors[:]
                    if best_bad == 0:
                        # Found valid coloring!
                        return {'n': n, 'colors': best_colors}
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.718281828, -delta / T):
                    current_colors[e] = c_new
                    current_bad += delta
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = current_colors[:]
                        if best_bad == 0:
                            return {'n': n, 'colors': best_colors}
            
            T *= cool_rate
        
        # If we didn't find 0, this n failed.
        # We don't update best_result unless we found 0.
        # So we continue to next n (smaller).
        
    return best_result
