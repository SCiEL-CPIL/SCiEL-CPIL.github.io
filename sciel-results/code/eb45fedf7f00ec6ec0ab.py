"""
Strategy: Fast Bitmask Verification + Simulated Annealing Local Search
Instead of relying on potentially incorrect hardcoded cyclic constructions or slow backtracking,
we use a robust local search approach:
1. Start with a valid coloring for n=17 (found via quick local search from a random start).
2. Attempt to extend to n=18 and n=19 by adding vertices and fixing C6 violations via simulated annealing.
3. Use a fast bitmask-based C6 detector to guide the search.
"""

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Set random seed for reproducibility if needed, but random is better for search
    random.seed(42)
    
    # Helper: Fast C6 detection using bitmasks
    # Returns a list of C6s found (as tuples of vertices) for a given coloring
    def find_c6s(colors, n):
        # Build adjacency masks for each color
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        c6s = []
        # For each color, check for C6
        # A C6 is v0-v1-v2-v3-v4-v5-v0
        # To make it fast, we iterate over pairs (v0, v1) that are adjacent,
        # then look for paths of length 4 from v1 to v0 in the same color graph.
        # However, for n<=19, a simpler approach might be faster in Python if optimized,
        # or we can use the "common neighbors" trick.
        
        # Let's use a more direct but optimized check for small n.
        # For each color c:
        for c in range(4):
            # Get list of edges for this color to iterate faster?
            # Or just use the masks.
            # Check all 6-combinations? C(19,6) = 27132. 4 colors * 27132 = ~100k checks.
            # Each check involves 6 adjacency lookups. This is feasible.
            # But we can prune.
            
            # Let's stick to the combination method but optimized with bitmasks.
            # Actually, for local search, we need to know *which* edges to flip.
            # So returning the list of C6s is crucial.
            
            # Efficient C6 detection:
            # For each vertex v0:
            #   For each neighbor v1 of v0 in color c:
            #     For each neighbor v2 of v1 in color c (v2 != v0):
            #       For each neighbor v3 of v2 in color c (v3 not in {v0, v1}):
            #         For each neighbor v4 of v3 in color c (v4 not in {v0, v1, v2}):
            #           For each neighbor v5 of v4 in color c (v5 not in {v0..v3}):
            #             If v5 is neighbor of v0 in color c and v5 != v1:
            #               Found C6: (v0, v1, v2, v3, v4, v5)
                
            # This is O(E^3) roughly, which might be slow in Python if E is large.
            # For n=19, max edges per color is ~171. 171^3 is too high.
            
            # Let's use the combination method with early termination and caching.
            # Since n is small (<=19), C(19,6) is small.
            from itertools import combinations
            for combo in combinations(range(n), 6):
                # Check if this set of 6 vertices contains a C6 in color c
                # There are 60 distinct cycles. We can check a subset or all.
                # To be safe, check all 60.
                # Optimization: If any vertex has degree < 2 in the induced subgraph, skip.
                # Let's just check the cycles.
                
                v = list(combo)
                # Fix v0 = v[0]
                v0 = v[0]
                rest = v[1:]
                
                # Permute rest
                # To reduce permutations, we can fix v1 to be the smallest neighbor of v0 in the induced graph?
                # No, just iterate.
                found_cycle = False
                for p in range(5):
                    v1 = rest[p]
                    # Check v0-v1 edge
                    if not ((adj[c][v0] >> v1) & 1):
                        continue
                    
                    # Now we need a path v1-v2-v3-v4-v5-v0 of length 4
                    # v2, v3, v4, v5 are the remaining 4 vertices.
                    # There are 4! = 24 permutations for the rest.
                    # We can iterate over them.
                    import itertools
                    for perm in itertools.permutations([x for x in rest if x != v1]):
                        v2, v3, v4, v5 = perm
                        # Check edges: v1-v2, v2-v3, v3-v4, v4-v5, v5-v0
                        if ((adj[c][v1] >> v2) & 1 and
                            (adj[c][v2] >> v3) & 1 and
                            (adj[c][v3] >> v4) & 1 and
                            (adj[c][v4] >> v5) & 1 and
                            (adj[c][v5] >> v0) & 1):
                            # Ensure it's a simple cycle (vertices are distinct by construction of combo)
                            # And v5 != v1 (guaranteed by distinctness)
                            found_cycle = True
                            break
                    if found_cycle:
                        break
                
                if found_cycle:
                    c6s.append(combo)
                    # For local search, we might not need ALL c6s, just one or a few.
                    # But returning all helps with "energy" calculation.
                    
        return c6s

    def count_c6s(colors, n):
        """Count the number of C6s. Faster than finding all if we just need the count."""
        # Reuse find_c6s but count length?
        # For local search, we want to minimize the number of C6s.
        # We can just return len(find_c6s(colors, n))
        # But find_c6s is expensive. Let's optimize it.
        
        # Optimized count:
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        count = 0
        from itertools import combinations
        for c in range(4):
            for combo in combinations(range(n), 6):
                v0 = combo[0]
                rest = list(combo[1:])
                found = False
                for p in range(5):
                    v1 = rest[p]
                    if not ((adj[c][v0] >> v1) & 1):
                        continue
                    # Check remaining 4
                    others = [x for x in rest if x != v1]
                    # 24 perms
                    import itertools
                    for perm in itertools.permutations(others):
                        v2, v3, v4, v5 = perm
                        if ((adj[c][v1] >> v2) & 1 and
                            (adj[c][v2] >> v3) & 1 and
                            (adj[c][v3] >> v4) & 1 and
                            (adj[c][v4] >> v5) & 1 and
                            (adj[c][v5] >> v0) & 1):
                            found = True
                            break
                    if found:
                        break
                if found:
                    count += 1
        return count

    # Helper: Local Search (Simulated Annealing)
    def local_search(colors, n, time_limit, initial_temp=1.0):
        current_colors = colors[:]
        current_energy = count_c6s(current_colors, n)
        
        if current_energy == 0:
            return current_colors, 0
        
        best_colors = current_colors[:]
        best_energy = current_energy
        
        temp = initial_temp
        cooling_rate = 0.995
        
        while time.time() < time_limit and temp > 0.01:
            # Pick a random edge to flip
            edge_idx = random.randint(0, len(current_colors) - 1)
            old_color = current_colors[edge_idx]
            new_color = (old_color + 1) % 4  # Or random different color
            # Better: pick a random different color
            new_color = random.choice([c for c in range(4) if c != old_color])
            
            current_colors[edge_idx] = new_color
            
            new_energy = count_c6s(current_colors, n)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = current_colors[:]
                    if best_energy == 0:
                        return best_colors, 0
            else:
                current_colors[edge_idx] = old_color
            
            temp *= cooling_rate
        
        return best_colors, best_energy

    # Step 1: Find a valid coloring for n=17
    n_target = 17
    # Initialize with random coloring
    colors_17 = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
    
    # Run local search for n=17
    time_for_17 = 10  # 10 seconds
    colors_17, energy_17 = local_search(colors_17, n_target, start_time + time_for_17)
    
    if energy_17 != 0:
        # Try again with different seed or more time
        # Or just accept it and try to extend? No, must be valid to extend.
        # If we can't find n=17, we return a smaller valid one.
        # Let's try a few more restarts
        for _ in range(3):
            colors_17 = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
            colors_17, energy_17 = local_search(colors_17, n_target, start_time + time_for_17)
            if energy_17 == 0:
                break
    
    if energy_17 != 0:
        # Fallback: return a known smaller valid coloring if we can't find 17
        # For now, try n=16
        n_target = 16
        colors_16 = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
        colors_16, energy_16 = local_search(colors_16, n_target, start_time + 5)
        if energy_16 == 0:
            return {'n': 16, 'colors': colors_16}
        
        # If even 16 fails, return a trivial valid small graph
        # K_5 is always 4-colorable without C6 (no C6 in K5)
        n = 5
        colors = [random.randint(0, 3) for _ in range(10)]
        return {'n': n, 'colors': colors}
    
    # Step 2: Try to extend to n=18
    n_try = 18
    # Extend colors_17 to n=18
    # New vertex 17. Edges (17, 0..16) need colors.
    # Initialize new edges randomly.
    colors_18 = colors_17[:]
    for i in range(17):
        colors_18.append(random.randint(0, 3))
    
    time_for_18 = 20
    colors_18, energy_18 = local_search(colors_18, n_try, start_time + time_for_18)
    
    if energy_18 == 0:
        return {'n': n_try, 'colors': colors_18}
    
    # Step 3: Try to extend to n=19
    if time.time() < start_time + 90:
        n_try = 19
        # Extend from best 18 (even if invalid) or 17?
        # Better to extend from the best 18 attempt, or restart from 17?
        # Let's try extending the best 18 found.
        colors_19 = colors_18[:]
        for i in range(18):
            colors_19.append(random.randint(0, 3))
        
        time_for_19 = start_time + 110
        colors_19, energy_19 = local_search(colors_19, n_try, time_for_19)
        
        if energy_19 == 0:
            return {'n': n_try, 'colors': colors_19}
        
        # If 19 failed, return best 18 if valid? No, 18 was invalid.
        # Return 17 which is valid.
    
    return {'n': 17, 'colors': colors_17}

if __name__ == "__main__":
    result = construct()
    print(f"Found coloring with n={result['n']}")
    print(f"Number of edges: {len(result['colors'])}")
