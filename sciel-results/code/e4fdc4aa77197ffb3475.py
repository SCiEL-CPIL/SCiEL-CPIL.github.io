import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    # Use bitmasks for adjacency: adj_mask[v] is a bitmask of neighbors of v
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                
    best_solution = []
    best_size = 0
    
    def popcount(x):
        """Count set bits in x"""
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def greedy_color(P):
        """
        P is a list of vertices.
        Returns: colors list (same length as P) and max_color
        Uses bitmask-based neighbor lookup for speed.
        """
        if not P:
            return [], 0
        colors = [0] * len(P)
        # color_map: vertex -> color, stored in a dict for O(1) lookup
        color_map = {}
        # For each color class, maintain a bitmask of vertices assigned that color
        color_masks = {}  # color -> bitmask
        
        for i, v in enumerate(P):
            # Find which colors are used by neighbors of v
            used_colors = set()
            am = adj_mask[v]
            for c, mask in color_masks.items():
                if am & mask:
                    used_colors.add(c)
            c = 1
            while c in used_colors:
                c += 1
            colors[i] = c
            color_map[v] = c
            if c not in color_masks:
                color_masks[c] = 1 << v
            else:
                color_masks[c] |= (1 << v)
        
        max_c = max(colors) if colors else 0
        return colors, max_c

    def greedy_mis_on_subset(vertices):
        """Greedy MIS on a subset of vertices using bitmasks"""
        remaining_mask = 0
        for v in vertices:
            remaining_mask |= (1 << v)
        result = []
        while remaining_mask:
            # Find vertex with minimum degree within remaining
            min_v = None
            min_deg = float('inf')
            temp = remaining_mask
            while temp:
                # Extract lowest set bit
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = popcount(adj_mask[v] & remaining_mask)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                temp ^= lsb
            result.append(min_v)
            remaining_mask &= ~(1 << min_v)
            remaining_mask &= ~adj_mask[min_v]
        return result

    def initial_solution():
        """Build a good initial solution: all odds + compatible evens"""
        odds = [i for i in range(1, n + 1) if i % 2 == 1]
        odds_mask = 0
        for o in odds:
            odds_mask |= (1 << o)
        
        # Check which evens are compatible with ALL odds
        compatible_evens = []
        for e in range(2, n + 1, 2):
            if adj_mask[e] & odds_mask:
                continue
            compatible_evens.append(e)
        
        # Now find max independent set among compatible evens
        even_mis = greedy_mis_on_subset(compatible_evens)
        
        return odds + even_mis

    # Get initial solution
    init_sol = initial_solution()
    if len(init_sol) > best_size:
        best_size = len(init_sol)
        best_solution = init_sol[:]
    
    # Also try a pure greedy MIS on all vertices as another starting point
    greedy_sol = greedy_mis_on_subset(range(1, n + 1))
    if len(greedy_sol) > best_size:
        best_size = len(greedy_sol)
        best_solution = greedy_sol[:]
    
    def expand(P, sol):
        nonlocal best_solution, best_size
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol.copy()
            return
            
        # Compute UB using greedy coloring
        colors, max_color = greedy_color(P)
        
        if len(sol) + max_color <= best_size:
            return
            
        # Order P by color (ascending) then by degree (descending)
        # Vertices with lower colors should be considered first
        P_with_colors = list(zip(P, colors))
        P_with_colors.sort(key=lambda x: (x[1], -popcount(adj_mask[x[0]])))
        P = [v for v, c in P_with_colors]
        colors = [c for v, c in P_with_colors]
        
        for i in range(len(P)):
            if time.time() - start_time > timeout:
                return
            v = P[i]
            v_mask = 1 << v
            
            # Include v: remove v and its neighbors from P
            new_P = []
            for j in range(len(P)):
                u = P[j]
                if u != v and not (adj_mask[v] & (1 << u)):
                    new_P.append(u)
            
            # Recurse with v included
            expand(new_P, sol + [v])
            
            # Exclude v: remove v from P
            P.pop(i)
            i -= 1
            
            # Check if remaining P can beat best
            if P:
                _, ub_remaining = greedy_color(P)
                if len(sol) + ub_remaining <= best_size:
                    break

    initial_P = list(range(1, n + 1))
    expand(initial_P, [])
    
    if not best_solution:
        best_solution = greedy_mis_on_subset(range(1, n + 1))

    return list(best_solution)
