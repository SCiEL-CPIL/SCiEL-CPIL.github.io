import time
import random

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # Helper to multiply in GF(16)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Log table for GF(16)
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Build the base 32-vertex coloring
    # Two copies of the 16-vertex Clebsch-based coloring, cross edges color 3
    n_base = 32
    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                # Within the same block (0-15 or 16-31)
                # Use log[i^j] % 3
                # i^j is non-zero since i != j
                base_colors.append(log[i ^ j] % 3)
            else:
                # Between blocks
                base_colors.append(3)
    
    best_n = n_base
    best_colors = base_colors[:]

    # Target n=50
    n_target = 50
    if time.time() - start_time > deadline - safety_margin:
        return {'n': best_n, 'colors': best_colors}

    # We will attempt to find a valid 50-vertex coloring using local search.
    # We start with the 32-vertex base and add 18 vertices.
    # We represent the graph as a list of colors for edges (i,j) with i<j.
    # Index for (i,j) in a list of size n(n-1)/2:
    # idx = i*(n-1) - i*(i-1)//2 + (j-i-1) ? 
    # Let's use a simpler 2D array or a dictionary for flexibility during search, 
    # but for speed, a flat list with precomputed indices is best.
    
    # Precompute edge indices for n=50
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # Number of edges before row i
        # Row 0 has n-1 edges
        # Row 1 has n-2 edges
        # ...
        # Row i has n-1-i edges
        # Sum of edges in rows 0 to i-1:
        # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Then add offset (j - i - 1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Initialize n=50 coloring
    # Copy base colors
    colors = [0] * (n_target * (n_target - 1) // 2)
    
    # Fill base part
    for i in range(n_base):
        for j in range(i + 1, n_base):
            idx = get_edge_index(i, j, n_target)
            colors[idx] = base_colors[i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)]

    # Fill new edges (involving vertices >= 32) with random colors
    for i in range(n_base):
        for j in range(i + 1, n_target):
            if j >= n_base:
                idx = get_edge_index(i, j, n_target)
                colors[idx] = random.randint(0, 3)
    for i in range(n_base, n_target):
        for j in range(i + 1, n_target):
            idx = get_edge_index(i, j, n_target)
            colors[idx] = random.randint(0, 3)

    # Data structure for fast triangle counting
    # adj[c][v] = list of neighbors u such that edge (v,u) has color c
    # Or better, for each vertex v and color c, a set of neighbors.
    # Since n=50, we can use bitsets (integers) for speed.
    
    def build_adjacency(colors, n):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_edge_index(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def count_triangles(adj, n):
        total = 0
        for c in range(4):
            for i in range(n):
                # For each neighbor j of i in color c
                neighbors = adj[c][i]
                # Count common neighbors of i and j in color c
                # This counts each triangle 3 times (once for each vertex)
                # We just need the sum.
                # Actually, iterating over neighbors and checking intersection is O(deg^2)
                # With n=50, max deg is 49. 50*49*49 ~ 120k ops per color. 4 colors ~ 500k.
                # This is too slow if done repeatedly.
                # We need incremental updates.
                pass
        return 0

    # Let's implement the incremental update logic directly in the search loop.
    # State: colors list, adj bitsets.
    # Energy E = sum over all triangles (1 if mono, 0 else).
    # Actually, let's just count the number of monochromatic triangles.
    
    def calculate_energy(adj, n):
        energy = 0
        for c in range(4):
            for i in range(n):
                neighbors = adj[c][i]
                # Iterate through bits in neighbors
                # For each j in neighbors, count common neighbors of i and j
                # Common neighbors = adj[c][i] & adj[c][j]
                # Number of bits in common neighbors
                # Sum over all j > i to avoid double counting?
                # If we sum over all i and all j in adj[c][i], we count each triangle 3 times.
                # So energy = (1/3) * sum_{c,i} popcount(adj[c][i] & adj[c][j]) for j in adj[c][i]?
                # No, simpler:
                # For each color c, for each pair (i,j) with edge color c,
                # count k such that (i,k) and (j,k) are color c.
                # This counts each triangle exactly 3 times (once for each edge).
                # So total triangles = (1/3) * sum_{c} sum_{(i,j) in E_c} |N_c(i) \cap N_c(j)|
                
                # Let's just do a simple O(n^3) check initially to get a baseline,
                # but for the search, we need O(n) updates.
                
                # O(n^3) check for n=50 is 125,000 * 4 = 500,000 ops.
                # This is fast enough for a few restarts, but not for 100k iterations.
                
                pass
        return 0

    # Efficient Energy Calculation and Update
    # Let's maintain `energy` as the total number of monochromatic triangles.
    # When edge (u,v) changes from c_old to c_new:
    # 1. Remove contribution of (u,v) in color c_old.
    #    Triangles involving (u,v) with color c_old:
    #    Any w such that (u,w) is c_old and (v,w) is c_old.
    #    Count = popcount(adj[c_old][u] & adj[c_old][v])
    #    Note: adj includes u and v? No, adj[c][u] has bits for neighbors.
    #    If we remove (u,v) from adj, the intersection might change?
    #    The triangles counted are those where all 3 edges are c_old.
    #    So we count w in N_c_old(u) \cap N_c_old(v).
    #    Let D_old = popcount(adj[c_old][u] & adj[c_old][v]).
    #    These D_old triangles are broken. So energy -= D_old.
    #
    # 2. Add contribution of (u,v) in color c_new.
    #    Triangles involving (u,v) with color c_new:
    #    Any w such that (u,w) is c_new and (v,w) is c_new.
    #    Count = popcount(adj[c_new][u] & adj[c_new][v]).
    #    Let D_new = popcount(adj[c_new][u] & adj[c_new][v]).
    #    These D_new triangles are created. So energy += D_new.
    #
    # 3. Update adj structures.
    #    adj[c_old][u] ^= (1 << v)
    #    adj[c_old][v] ^= (1 << u)
    #    adj[c_new][u] ^= (1 << v)
    #    adj[c_new][v] ^= (1 << u)
    #
    # Wait, the intersection `adj[c][u] & adj[c][v]` includes the nodes themselves?
    # No, adj[c][u] contains v if edge (u,v) is c.
    # If we calculate intersection BEFORE updating adj, does it include u or v?
    # adj[c][u] has bit v. adj[c][v] has bit u.
    # Intersection of {v, ...} and {u, ...} does not include u or v (unless u=v, which is false).
    # So the count D is exactly the number of w distinct from u,v.
    # So the logic holds.
    
    def initial_energy_and_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_edge_index(i, j, n)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        energy = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if edge (i,j) is color c
                    # If so, count common neighbors
                    if (adj[c][i] >> j) & 1:
                        common = adj[c][i] & adj[c][j]
                        energy += common.bit_count()
        # Each triangle is counted 3 times (for each of its 3 edges)
        energy //= 3
        return energy, adj

    # Search Loop
    # We will try multiple restarts if we don't find a solution quickly.
    
    found = False
    max_restarts = 10
    max_iters_per_restart = 20000
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - safety_margin:
            break
            
        # Re-initialize random part for new restart?
        # Or just continue from current state?
        # Let's re-randomize the new edges for a fresh start.
        if restart > 0:
            for i in range(n_base):
                for j in range(i + 1, n_target):
                    if j >= n_base:
                        idx = get_edge_index(i, j, n_target)
                        colors[idx] = random.randint(0, 3)
            for i in range(n_base, n_target):
                for j in range(i + 1, n_target):
                    idx = get_edge_index(i, j, n_target)
                    colors[idx] = random.randint(0, 3)
        
        energy, adj = initial_energy_and_adj(colors, n_target)
        
        if energy == 0:
            found = True
            break
            
        for it in range(max_iters_per_restart):
            if time.time() - start_time > deadline - safety_margin:
                break
                
            # Pick a random edge
            idx = random.randint(0, len(colors) - 1)
            
            # Find (u, v) from idx
            # Invert the index formula
            # idx = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # This is tricky to invert directly.
            # Precompute a list of (i,j) for all idx?
            # n=50, 1225 edges. This is small.
            
            # Let's precompute edge_pairs
            if not hasattr(construct, '_edge_pairs_50'):
                pairs = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        pairs.append((i, j))
                construct._edge_pairs_50 = pairs
                
            u, v = construct._edge_pairs_50[idx]
            
            old_c = colors[idx]
            
            # Try changing to a new color
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Calculate Delta Energy
            # D_old: triangles broken
            # Triangles (u, v, w) with color old_c.
            # w must be in N_old(u) \cap N_old(v).
            # Note: adj[old_c][u] contains v. adj[old_c][v] contains u.
            # Intersection does not contain u or v.
            common_old = adj[old_c][u] & adj[old_c][v]
            D_old = common_old.bit_count()
            
            # D_new: triangles created
            # Triangles (u, v, w) with color new_c.
            # w must be in N_new(u) \cap N_new(v).
            common_new = adj[new_c][u] & adj[new_c][v]
            D_new = common_new.bit_count()
            
            delta_e = D_new - D_old
            
            if delta_e < 0:
                # Accept
                colors[idx] = new_c
                energy += delta_e
                
                # Update adj
                adj[old_c][u] ^= (1 << v)
                adj[old_c][v] ^= (1 << u)
                adj[new_c][u] ^= (1 << v)
                adj[new_c][v] ^= (1 << u)
                
                if energy == 0:
                    found = True
                    break
            else:
                # Reject (or accept with small probability for simulated annealing, 
                # but hill climbing is usually sufficient for this structured problem)
                # We can add a small random kick if stuck
                pass
        
        if found:
            break
            
    if found:
        best_n = n_target
        best_colors = colors[:]
        
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
