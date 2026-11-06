import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper to check if a specific color class (adjacency list) contains a C6
    def has_c6(adj, n):
        # adj is a list of lists or bitsets
        # Check all 6-subsets
        for combo in itertools.combinations(range(n), 6):
            # Check if the induced subgraph on combo contains a C6
            # A C6 exists if there is a cycle of length 6.
            # Since there are only 6 vertices, the cycle must use all of them.
            # So we check if the subgraph has a Hamiltonian cycle.
            # We can do this by fixing one vertex and checking permutations of the rest?
            # Or DFS.
            # Let's use a simple check: 
            # A graph on 6 vertices has a C6 iff it is connected and has a cycle of length 6.
            # Actually, just checking for a Hamiltonian cycle is sufficient.
            
            # Optimization: if any vertex has degree < 2 in the subgraph, no C6.
            # Build local adjacency for the 6 vertices
            local_adj = [0] * 6
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[combo[i]][combo[j]]:
                        local_adj[i] |= (1 << j)
                        local_adj[j] |= (1 << i)
            
            # Check degrees
            for i in range(6):
                if bin(local_adj[i]).count('1') < 2:
                    break
                elif i == 5:
                    # All degrees >= 2. Check for C6.
                    # Try to find a cycle of length 6.
                    # Fix vertex 0.
                    # Next vertex must be a neighbor of 0.
                    # We can just check all permutations of 1..5? 5! = 120.
                    # Or use DFS.
                    # Let's just check if there is a cycle.
                    # Since it's small, we can try:
                    # Does there exist a permutation p of 1..5 such that 
                    # 0-p[0]-p[1]-p[2]-p[3]-p[4]-0 is a cycle?
                    # We need edges (0,p[0]), (p[0],p[1]), ..., (p[4],0).
                    
                    # Let's iterate permutations
                    found = False
                    for p in itertools.permutations(range(1, 6)):
                        if (local_adj[0] >> p[0]) & 1 and \
                           (local_adj[p[0]] >> p[1]) & 1 and \
                           (local_adj[p[1]] >> p[2]) & 1 and \
                           (local_adj[p[2]] >> p[3]) & 1 and \
                           (local_adj[p[3]] >> p[4]) & 1 and \
                           (local_adj[p[4]] >> 0) & 1:
                            found = True
                            break
                    if found:
                        return True
        return False

    def count_c6s(colors, n):
        # Returns number of C6s in the coloring
        # colors is a list of length n*(n-1)//2
        # Reconstruct adjacency for each color
        total_c6s = 0
        for c in range(4):
            adj = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    idx += 1
            if has_c6(adj, n):
                # Just return a large number or count?
                # For local search, we want a gradient.
                # Counting exact C6s is hard.
                # Let's just return 1 if any, else 0?
                # Better: return number of C6s?
                # My has_c6 returns bool.
                # Let's modify has_c6 to count?
                # Or just use 1/0 for now.
                total_c6s += 1
        return total_c6s

    def get_best_n():
        # Try to find valid coloring for n from 17 down to 10
        for n in range(17, 9, -1):
            if time.time() > deadline:
                break
            # Try random restarts
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Initialize random colors
                num_edges = n * (n - 1) // 2
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                
                # Local search
                # Try to minimize count_c6s
                # If 0, return
                current_cost = count_c6s(colors, n)
                if current_cost == 0:
                    return n, colors
                
                # Simple hill climbing
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Pick random edge
                    for _ in range(100):
                        if time.time() > deadline:
                            return None, None
                        idx = random.randint(0, num_edges - 1)
                        old_c = colors[idx]
                        best_c = old_c
                        best_cost = current_cost
                        for c in range(4):
                            if c == old_c: continue
                            colors[idx] = c
                            new_cost = count_c6s(colors, n)
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_c = c
                        colors[idx] = best_c
                        if best_cost < current_cost:
                            current_cost = best_cost
                            improved = True
                            if current_cost == 0:
                                return n, colors
        return None, None

    result = get_best_n()
    if result[0] is not None:
        return {'n': result[0], 'colors': result[1]}
    
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
