"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Build conflict: a conflicts with b if a+b is a perfect square (a != b)
    # Self-conflict: 2a is a perfect square -> a is forbidden
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N and b != a:
                if b in num_to_idx:
                    j = num_to_idx[b]
                    adj[i] |= (1 << j)

    # Tomita's algorithm for Maximum Independent Set
    # We work on the complement graph? No, we work on the conflict graph.
    # Independent Set in G = Clique in G_complement.
    # Or just use B&B on G for Independent Set.
    
    # Let's stick to Independent Set on G (conflict graph).
    # Vertices: 0..M-1
    # adj[i] is bitmask of neighbors of i in G.
    
    # We want to find a subset S such that no two elements in S are adjacent in G.
    
    # Algorithm:
    # 1. Greedy coloring on the subgraph induced by candidates P.
    #    Colors provide an upper bound on the size of the independent set in P.
    # 2. Branch on vertices in reverse color order.
    
    # To make it fast:
    # - Represent P as a bitmask (int).
    # - Greedy coloring on a bitmask is tricky but doable if we iterate bits.
    # - However, standard Tomita uses lists. For M~200, lists are okay if optimized.
    # - Better: Use bitsets for adjacency checks.
    
    # Let's implement a bitset-based B&B.
    
    # Pre-compute degrees for initial ordering
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    
    # Initial greedy solution to set lower bound
    # Order by degree (ascending)
    order = sorted(range(M), key=lambda x: degrees[x])
    
    def greedy_iset(vertices):
        """Greedy Independent Set from a list of vertices."""
        selected = 0
        for v in vertices:
            if (adj[v] & selected) == 0:
                selected |= (1 << v)
        return selected

    best_mask = greedy_iset(order)
    best_size = bin(best_mask).count('1')
    
    # Try a few random restarts for a better lower bound
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        mask = greedy_iset(random_order)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # B&B State
    # We will use a recursive function but with explicit stack or deep recursion limit.
    # Python recursion limit is usually 1000, which is enough for depth 200.
    sys.setrecursionlimit(3000)
    
    # Global variables for B&B
    current_best_size = best_size
    current_best_set = best_mask
    
    # Helper: Greedy coloring on a set of vertices given as a bitmask.
    # Returns: (max_color, colors_dict) where colors_dict maps vertex -> color.
    # We need to process vertices in a specific order (e.g., degree within P).
    
    def greedy_coloring(P_mask):
        """
        Greedy coloring of the subgraph induced by P_mask.
        Returns (num_colors, color_of_vertex_dict).
        We order vertices by degree in P (ascending) for better coloring.
        """
        # Get list of vertices in P
        verts = []
        tmp = P_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            tmp ^= lsb
            
        if not verts:
            return 0, {}
        
        # Compute degree within P for each vertex
        # degree(v) = popcount(adj[v] & P_mask)
        # Sort by degree ascending
        verts.sort(key=lambda v: (adj[v] & P_mask).bit_count())
        
        colors = {}
        max_color = 0
        
        for v in verts:
            # Find smallest color not used by neighbors of v in P
            # Neighbors of v in P: adj[v] & P_mask
            # We need to check colors of already colored neighbors.
            # Since we color in order, we only need to check neighbors that are earlier in 'verts'?
            # No, standard greedy coloring: when coloring v, its neighbors that are ALREADY colored matter.
            # If we process in fixed order, we check colors of neighbors in that order.
            
            neighbor_colors = set()
            # Iterate through neighbors of v in P
            # To be efficient, we can just check the colors of all neighbors in P that have been assigned a color.
            # Since we process sequentially, we can maintain a set of colors for neighbors.
            
            # Optimization: Instead of checking all neighbors, we can just check the specific neighbors.
            # For M=200, checking ~20 neighbors per vertex is fine.
            
            # Get neighbors of v in P
            neigh_mask = adj[v] & P_mask
            # Iterate bits in neigh_mask
            tmp_neigh = neigh_mask
            while tmp_neigh:
                lsb = tmp_neigh & -tmp_neigh
                u = lsb.bit_length() - 1
                if u in colors:
                    neighbor_colors.add(colors[u])
                tmp_neigh ^= lsb
                
            c = 1
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        return max_color, colors

    def branch_and_bound(P_mask, current_size, current_set_mask):
        nonlocal current_best_size, current_best_set, deadline
        
        if time.time() > deadline:
            return
        
        if not P_mask:
            if current_size > current_best_size:
                current_best_size = current_size
                current_best_set = current_set_mask
            return
            
        # Upper bound via greedy coloring
        color_count, colors = greedy_coloring(P_mask)
        
        if current_size + color_count <= current_best_size:
            return
            
        # Branching
        # We process vertices in reverse order of their color assignment from greedy_coloring.
        # Vertices with higher color are processed first?
        # Tomita: Process in reverse order of the coloring list.
        # The coloring list is sorted by degree.
        # Let's retrieve the order used in coloring.
        # Actually, we need the order. Let's modify greedy_coloring to return the order too.
        
        # Re-implement greedy_coloring to return order
        verts = []
        tmp = P_mask
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            tmp ^= lsb
            
        if not verts:
            return

        verts.sort(key=lambda v: (adj[v] & P_mask).bit_count())
        
        # Re-color to get colors consistent with this order
        colors = {}
        for v in verts:
            neighbor_colors = set()
            neigh_mask = adj[v] & P_mask
            tmp_neigh = neigh_mask
            while tmp_neigh:
                lsb = tmp_neigh & -tmp_neigh
                u = lsb.bit_length() - 1
                if u in colors:
                    neighbor_colors.add(colors[u])
                tmp_neigh ^= lsb
                
            c = 1
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            
        # Now, iterate through verts in reverse order
        # For each vertex v at index i in verts (0-based):
        # The upper bound for the subgraph induced by verts[0..i] is i+1 (since colors are 1..i+1 max)
        # Wait, the bound is: current_size + (number of vertices remaining in this branch)
        # Specifically, if we are at vertex v which is the (i+1)-th in the list,
        # and we include v, the remaining candidates are a subset of verts[0..i] that are not neighbors of v.
        
        for i in range(len(verts) - 1, -1, -1):
            v = verts[i]
            
            # Upper bound check:
            # The max size we can get from verts[0..i] is i+1 (trivial bound)
            # But we can use the color bound for the subgraph verts[0..i]?
            # Tomita's bound: If we are processing v (which has color c_v in the full coloring),
            # and we are at index i, the remaining vertices are verts[0..i].
            # The max clique size in verts[0..i] is at most i+1.
            # So if current_size + i + 1 <= current_best_size, we can prune the whole loop.
            if current_size + i + 1 <= current_best_size:
                return
                
            # Include v
            # New candidates: verts[0..i] that are NOT neighbors of v
            new_P_mask = 0
            for j in range(i):
                u = verts[j]
                if not ((adj[v] >> u) & 1):
                    new_P_mask |= (1 << u)
                    
            new_size = current_size + 1
            new_set_mask = current_set_mask | (1 << v)
            
            # Recurse
            branch_and_bound(new_P_mask, new_size, new_set_mask)
            
            # Exclude v: loop continues to i-1

    # Run B&B
    # Initial P is all vertices
    all_mask = (1 << M) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # If B&B didn't find anything better (or timed out), use best_mask
    result_mask = current_best_set
    if bin(result_mask).count('1') < bin(best_mask).count('1'):
        result_mask = best_mask
        
    # Convert mask to list of numbers
    result = []
    for i in range(M):
        if (result_mask >> i) & 1:
            result.append(valid_numbers[i])
            
    if not result:
        # Fallback
        result = [valid_numbers[0]]
        
    return sorted(result)

# EVOLVE-BLOCK-END
