import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Known valid coloring for n=9 (circular distance 1,2,3)
    # Colors: distance 1 -> 0, distance 2 -> 1, distance 3 -> 2
    def make_circular_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = min(j - i, n - j + i)
                # For n=9: distances 1,2,3
                # Map distances to colors: d%3
                colors.append(d % 3)
        return {'n': n, 'colors': colors}
    
    # Verify a coloring has no C6 in any color class
    def has_c6(colors, n):
        # Build adjacency for each color
        for c in range(4):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            # Check for 6-cycles in this color class
            # A 6-cycle: v0-v1-v2-v3-v4-v5-v0, all distinct, all edges in adj
            # We can check by looking for 6-tuples
            for v0 in range(n):
                for v1 in adj[v0]:
                    if v1 <= v0:
                        continue
                    for v2 in adj[v1]:
                        if v2 <= v0 or v2 == v1:
                            continue
                        for v3 in adj[v2]:
                            if v3 in (v0, v1, v2):
                                continue
                            for v4 in adj[v3]:
                                if v4 in (v0, v1, v2, v3):
                                    continue
                                for v5 in adj[v4]:
                                    if v5 in (v0, v1, v2, v3, v4):
                                        continue
                                    if v0 in adj[v5]:
                                        return True
        return False
    
    def is_valid(result):
        n = result['n']
        colors = result['colors']
        expected_len = n * (n - 1) // 2
        if len(colors) != expected_len:
            return False
        if any(c not in (0, 1, 2, 3) for c in colors):
            return False
        return not has_c6(colors, n)
    
    # Start with known valid n=9
    best_result = make_circular_coloring(9)
    if not is_valid(best_result):
        # Fallback to a trivial valid coloring
        best_result = {'n': 1, 'colors': []}
    
    # Try to find better colorings using backtracking
    def try_find_coloring(n, time_budget):
        """Try to find a 4-coloring of K_n with no C6 in any color class."""
        if time.time() > time_budget:
            return None
        
        # Use backtracking with symmetry breaking
        # Assign colors to edges in lexicographic order
        # Use memoization of partial constraints
        
        # For efficiency, use a simpler approach:
        # Start from a known valid smaller coloring and try to extend
        
        # Try random restarts with local search
        import random
        random.seed(42)
        
        best_local = None
        attempts = 0
        max_attempts = 1000
        while time.time() < time_budget and attempts < max_attempts:
            attempts += 1
            
            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Local search: try to fix violations
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                # Check for violations and try to fix
                for c in range(4):
                    adj = [set() for _ in range(n)]
                    idx = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if colors[idx] == c:
                                adj[i].add(j)
                                adj[j].add(i)
                            idx += 1
                    
                    # Find 6-cycles
                    found_cycle = None
                    for v0 in range(n):
                        if found_cycle:
                            break
                        for v1 in adj[v0]:
                            if v1 <= v0 or found_cycle:
                                continue
                            for v2 in adj[v1]:
                                if v2 <= v0 or v2 == v1 or found_cycle:
                                    continue
                                for v3 in adj[v2]:
                                    if v3 in (v0, v1, v2) or found_cycle:
                                        continue
                                    for v4 in adj[v3]:
                                        if v4 in (v0, v1, v2, v3) or found_cycle:
                                            continue
                                        for v5 in adj[v4]:
                                            if v5 in (v0, v1, v2, v3, v4) or found_cycle:
                                                continue
                                            if v0 in adj[v5]:
                                                found_cycle = (v0, v1, v2, v3, v4, v5)
                                                break
                    
                    if found_cycle:
                        v0, v1, v2, v3, v4, v5 = found_cycle
                        # Try changing one edge to break the cycle
                        edges_in_cycle = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                        for (a, b) in edges_in_cycle:
                            if a > b:
                                a, b = b, a
                            # Find index
                            idx = 0
                            found_idx = -1
                            for i in range(n):
                                for j in range(i+1, n):
                                    if i == a and j == b:
                                        found_idx = idx
                                        break
                                    idx += 1
                                if found_idx >= 0:
                                    break
                            
                            if found_idx >= 0:
                                old_color = colors[found_idx]
                                new_color = (old_color + 1) % 4
                                colors[found_idx] = new_color
                                improved = True
                                break
                        if improved:
                            continue
                    else:
                        # No C6 in color c
                        pass
                        break  # Need to check all colors properly
            
            # Check if valid
            if not has_c6(colors, n):
                best_local = {'n': n, 'colors': colors.copy()}
                break
        
        return best_local
    
    # Try increasing n from 10 to 20
    for n in range(10, 21):
        if time.time() > deadline - 10:
            break
        
        budget = min(deadline - 15, time.time() + 5)
        result = try_find_coloring(n, budget)
        if result and is_valid(result):
            best_result = result
            # Don't continue to higher n if we found a good one and time is tight
            if n >= 17:
                break
    
    return best_result
