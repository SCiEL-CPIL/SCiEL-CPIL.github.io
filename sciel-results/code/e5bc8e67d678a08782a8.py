import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks for all pairs
    # conflicts[i] is a bitmask of j such that numbers[i] and numbers[j] conflict
    conflicts = [0] * N
    
    def inc(a, b):
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            if inc(a, b):
                mask |= (1 << j)
        conflicts[i] = mask

    # 1. Identify odds and evens
    odds_indices = [i for i in range(N) if numbers[i] % 2 == 1]
    evens_indices = [i for i in range(N) if numbers[i] % 2 == 0]

    # 2. Filter evens that are compatible with ALL odds
    # An even e is compatible with an odd o if (conflicts[e] & (1 << o)) == 0
    odds_ok_indices = []
    for e in evens_indices:
        ok = True
        for o in odds_indices:
            if (conflicts[e] >> o) & 1:
                ok = False
                break
        if ok:
            odds_ok_indices.append(e)

    # 3. Solve MIS on the subgraph induced by odds_ok_indices
    # Map global indices to local indices for the MIS solver
    n_ev = len(odds_ok_indices)
    
    if n_ev == 0:
        return [numbers[i] for i in odds_indices]

    # Build local conflict masks
    # local_conflicts[k] is bitmask of local neighbors
    local_conflicts = [0] * n_ev
    for i in range(n_ev):
        global_i = odds_ok_indices[i]
        mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            global_j = odds_ok_indices[j]
            if (conflicts[global_i] >> global_j) & 1:
                mask |= (1 << j)
        local_conflicts[i] = mask

    # Tomita's algorithm for Maximum Independent Set
    # Using bitsets for efficiency
    # We reorder vertices by degree to improve pruning
    degrees = [bin(local_conflicts[i]).count('1') for i in range(n_ev)]
    order = sorted(range(n_ev), key=lambda x: -degrees[x])
    
    # Remap
    remap = [0] * n_ev
    for new_i, old_i in enumerate(order):
        remap[new_i] = old_i
        
    new_masks = [0] * n_ev
    for i in range(n_ev):
        old_i = order[i]
        mask = local_conflicts[old_i]
        new_mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            old_j = order[j]
            if (mask >> old_j) & 1:
                new_mask |= (1 << j)
        new_masks[i] = new_mask

    best_size = [0]
    best_set = [frozenset()]
    
    import sys
    sys.setrecursionlimit(10000)

    def popcount(x):
        return bin(x).count('1')

    # Greedy coloring for upper bound
    # This is a standard technique in Tomita's algorithm
    # We color the subgraph induced by available vertices.
    # The number of colors is an upper bound on the size of the MIS.
    # Actually, for MIS, we color the *complement* graph? No.
    # Standard Tomita: 
    # 1. Order vertices.
    # 2. Recursively include/exclude.
    # 3. Use a greedy coloring of the *remaining* vertices to bound the search.
    #    If we have `k` colors, the max independent set size in the remaining graph is <= k.
    #    Wait, the coloring is on the graph itself. An independent set in G is a clique in G'.
    #    The chromatic number of G is an upper bound on the size of a clique in G'.
    #    So yes, greedy coloring of the conflict graph gives an upper bound.
    
    def greedy_color(available, masks):
        """
        Returns a list of colors for vertices in available.
        Color c means the vertex is in color class c.
        The number of colors used is the upper bound.
        """
        color_of = {}
        next_color = 0
        rem = available
        # We process vertices in some order. 
        # For the bound to be tight, we usually process in the order they are considered.
        # But a simple greedy pass works.
        # To get a good bound, we should assign the smallest possible color.
        # We need to know which colors are "used" by neighbors.
        
        # A more efficient bound calculation for Tomita:
        # The algorithm typically maintains a list of vertices sorted by color.
        # Here we just compute the number of colors needed.
        
        # Let's implement a simple greedy coloring count
        # This is O(n^2) per call, which might be slow if called frequently.
        # However, n_ev is small (< 20 usually), so it's fine.
        
        colors = [0] * n_ev # 0 means uncolored
        max_color = 0
        temp_rem = available
        # We should color in the order of 'available' bits? 
        # Standard Tomita colors the subgraph induced by the available set.
        
        # To make it fast, we just count the colors used by a greedy assignment
        # on the vertices in 'available'.
        
        # Get list of vertices
        verts = []
        temp = available
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
            
        # Greedy coloring
        used_colors = [False] * (len(verts) + 1)
        num_colors = 0
        for v in verts:
            # Check which colors are forbidden by neighbors in 'available'
            # Actually, we just need to find the smallest color not used by neighbors
            forbidden = [False] * (num_colors + 2)
            neighbors = masks[v] & available
            temp_n = neighbors
            while temp_n:
                u = (temp_n & (-temp_n)).bit_length() - 1
                if colors[u] > 0:
                    forbidden[colors[u]] = True
                temp_n &= temp_n - 1
            
            # Find smallest available color
            c = 1
            while c <= num_colors and forbidden[c]:
                c += 1
            if c > num_colors:
                num_colors = c
            colors[v] = c
            
        return num_colors

    def solve(available, current_size, current_set):
        if elapsed() > deadline - 2.0:
            return

        if available == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = frozenset(current_set)
            return

        # Upper bound via greedy coloring
        # Note: The coloring bound is valid for the subgraph induced by 'available'.
        ub = greedy_color(available, new_masks)
        
        if current_size + ub <= best_size[0]:
            return

        # Select a vertex to branch on
        # Heuristic: pick vertex with max degree in available? 
        # Or just the first one? Tomita usually picks the first in the ordered list.
        # Let's pick the one with highest degree for better pruning in this simple version.
        temp = available
        best_v = -1
        best_deg = -1
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            deg = popcount(new_masks[v] & available)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= temp - 1

        if best_v < 0:
            return

        # Branch 1: Include best_v
        new_available = available & ~(1 << best_v) & ~new_masks[best_v]
        new_set = set(current_set)
        new_set.add(best_v)
        solve(new_available, current_size + 1, frozenset(new_set))

        # Branch 2: Exclude best_v
        new_available = available & ~(1 << best_v)
        solve(new_available, current_size, current_set)

    initial_available = (1 << n_ev) - 1
    solve(initial_available, 0, frozenset())

    # Map back to global indices
    final_indices = set(odds_indices)
    for idx in best_set[0]:
        final_indices.add(odds_ok_indices[remap[idx]])

    # Convert to values
    final_values = [numbers[i] for i in final_indices]
    
    # Verification
    # The problem guarantees a valid set exists. 
    # Our construction is structurally valid:
    # 1. All odds are mutually compatible.
    # 2. Chosen evens are compatible with all odds.
    # 3. Chosen evens form an independent set (mutually compatible).
    # So the result is guaranteed valid.
    
    return final_values

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(len(res))
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
    print(valid)
