"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Fallback: the known K9 construction
    def k9_fallback():
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    def build_adj(color_assignment, n):
        """Build adjacency lists for each color class."""
        adj = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = color_assignment[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        return adj

    def count_c6s(adj, n, color):
        """Count all C6s in a given color class."""
        adj_c = adj[color]
        count = 0
        for a0 in range(n):
            for a1 in adj_c[a0]:
                if a1 <= a0:
                    continue
                for a2 in adj_c[a1]:
                    if a2 == a0:
                        continue
                    for a3 in adj_c[a2]:
                        if a3 == a0 or a3 == a1:
                            continue
                        for a4 in adj_c[a3]:
                            if a4 in (a0, a1, a2):
                                continue
                            for a5 in adj_c[a4]:
                                if a5 in (a0, a1, a2, a3):
                                    continue
                                if a0 in adj_c[a5]:
                                    count += 1
        return count

    def find_c6(adj, n, color):
        """Find one C6 in a given color class, return tuple of 6 vertices or None."""
        adj_c = adj[color]
        for a0 in range(n):
            for a1 in adj_c[a0]:
                if a1 <= a0:
                    continue
                for a2 in adj_c[a1]:
                    if a2 == a0:
                        continue
                    for a3 in adj_c[a2]:
                        if a3 == a0 or a3 == a1:
                            continue
                        for a4 in adj_c[a3]:
                            if a4 in (a0, a1, a2):
                                continue
                            for a5 in adj_c[a4]:
                                if a5 in (a0, a1, a2, a3):
                                    continue
                                if a0 in adj_c[a5]:
                                    return (a0, a1, a2, a3, a4, a5)
        return None

    def get_pair_index(i, j, n):
        """Get the index in lexicographic order for pair (i, j) where i < j."""
        idx = 0
        for ii in range(i):
            idx += (n - ii - 1)
        idx += (j - i - 1)
        return idx

    def local_search(n, color_assignment, time_limit, max_iter=50000):
        """Local search: flip edges to eliminate C6 violations."""
        adj = build_adj(color_assignment, n)
        
        for iteration in range(max_iter):
            if time.time() > time_limit:
                return color_assignment, False
            
            # Find a C6 in any color
            found_cycle = None
            found_color = -1
            for c in range(4):
                cycle = find_c6(adj, n, c)
                if cycle is not None:
                    found_cycle = cycle
                    found_color = c
                    break
            
            if found_cycle is None:
                return color_assignment, True
            
            # We have a C6: v0-v1-v2-v3-v4-v5-v0 in color found_color
            # Flip one edge in the cycle to break it
            # Choose the edge that minimizes new C6s created
            cycle = found_cycle
            best_edge = None
            best_score = float('inf')
            best_new_color = None
            
            for k in range(6):
                u = cycle[k]
                v = cycle[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                
                # Try changing this edge from found_color to another color
                for new_c in range(4):
                    if new_c == found_color:
                        continue
                    
                    # Check if this creates a C6 in new_c
                    # We need to check if adding edge (u,v) to color new_c creates a C6
                    # That requires a path of length 4 from u to v in new_c avoiding this edge
                    
                    adj_new = adj[new_c]
                    # Check for path u -> x1 -> x2 -> x3 -> x4 -> v in adj_new
                    # where none of x1..x4 are u or v
                    creates_c6 = False
                    for x1 in adj_new[u]:
                        if x1 == v:
                            continue
                        for x2 in adj_new[x1]:
                            if x2 == u or x2 == v or x2 == x1:
                                continue
                            for x3 in adj_new[x2]:
                                if x3 == u or x3 == v or x3 == x1 or x3 == x2:
                                    continue
                                for x4 in adj_new[x3]:
                                    if x4 == u or x4 == v or x4 == x1 or x4 == x2 or x4 == x3:
                                        continue
                                    if v in adj_new[x4]:
                                        creates_c6 = True
                                        break
                                if creates_c6:
                                    break
                            if creates_c6:
                                break
                        if creates_c6:
                            break
                    if creates_c6:
                        continue
                    
                    # Count how many C6s this flip would create in new_c
                    # Approximate: just use 0 if no immediate C6
                    score = 0
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
                        best_new_color = new_c
            
            if best_edge is None:
                # No valid flip found, try random flip
                idx = random.randint(0, n * (n - 1) // 2 - 1)
                old_color = color_assignment[idx]
                new_color = random.choice([c for c in range(4) if c != old_color])
                color_assignment[idx] = new_color
                # Update adjacency
                # Find pair
                i = 0
                remaining = idx
                while remaining >= (n - i - 1):
                    remaining -= (n - i - 1)
                    i += 1
                j = i + 1 + remaining
                adj[old_color][i].discard(j)
                adj[old_color][j].discard(i)
                adj[new_color][i].add(j)
                adj[new_color][j].add(i)
            else:
                u, v = best_edge
                idx = get_pair_index(u, v, n)
                old_color = color_assignment[idx]
                new_color = best_new_color
                color_assignment[idx] = new_color
                adj[old_color][u].discard(v)
                adj[old_color][v].discard(u)
                adj[new_color][u].add(v)
                adj[new_color][v].add(u)
        
        return color_assignment, False

    def greedy_init(n):
        """Greedy initialization: assign colors to minimize immediate C6 risk."""
        color_assignment = [0] * (n * (n - 1) // 2)
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Choose color that minimizes degree in that color class
                best_c = 0
                best_deg = float('inf')
                for c in range(4):
                    deg = len(adj[c][i]) + len(adj[c][j])
                    if deg < best_deg:
                        best_deg = deg
                        best_c = c
                color_assignment[idx] = best_c
                adj[best_c][i].add(j)
                adj[best_c][j].add(i)
                idx += 1
        
        return color_assignment

    def random_init(n):
        """Random initialization."""
        return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]

    # Try increasing n from high to low, using local search
    best_result = k9_fallback()
    best_n = 9
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple strategies
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            if attempt < 5:
                # Random init
                color_assignment = random_init(n)
            else:
                # Greedy init
                color_assignment = greedy_init(n)
            
            result, success = local_search(n, color_assignment, deadline - 5, max_iter=20000)
            if success:
                # Verify
                adj = build_adj(result, n)
                valid = True
                for c in range(4):
                    if find_c6(adj, n, c) is not None:
                        valid = False
                        break
                if valid:
                    best_result = {'n': n, 'colors': result}
                    best_n = n
                    found = True
                    break
        
        if found:
            # Try to go higher
            continue
        else:
            # Couldn't find for this n, stop
            break
    
    # If we found n=17, try n=18
    if best_n >= 17 and time.time() < deadline - 20:
        for n in range(best_n + 1, 21):
            if time.time() > deadline - 5:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 3:
                    break
                
                color_assignment = random_init(n)
                result, success = local_search(n, color_assignment, deadline - 3, max_iter=10000)
                if success:
                    adj = build_adj(result, n)
                    valid = True
                    for c in range(4):
                        if find_c6(adj, n, c) is not None:
                            valid = False
                            break
                    if valid:
                        best_result = {'n': n, 'colors': result}
                        best_n = n
                        found = True
                        break
            
            if not found:
                break
    
    return best_result
# EVOLVE-BLOCK-END
