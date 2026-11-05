import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict graph using bitsets for speed
    # conflict_mask[v] is a bitset where bit u is set if v conflicts with u
    # We use bit (v-1) to represent vertex v, so bits 0..255
    conflict_mask = [0] * (N + 1)
    
    # Build conflict masks
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Check if (i*j) % (i+j) == 0
            # This is equivalent to (i/g) + (j/g) dividing g, where g = gcd(i,j)
            # Let g = gcd(i,j), i = g*u, j = g*v.
            # (i*j) % (i+j) == 0  <=>  g^2*u*v % (g*(u+v)) == 0  <=>  g*u*v % (u+v) == 0
            # Since gcd(u,v)=1, gcd(u+v, u)=1 and gcd(u+v, v)=1, so gcd(u+v, u*v)=1.
            # Thus (u+v) must divide g.
            g = math.gcd(i, j)
            u = i // g
            v = j // g
            if g % (u + v) == 0:
                conflict_mask[i] |= (1 << (j - 1))
                conflict_mask[j] |= (1 << (i - 1))

    # Bitset for all vertices
    all_mask = 0
    for v in vertices:
        all_mask |= (1 << (v - 1))

    def can_add(v, s_mask):
        """Check if vertex v can be added to set represented by s_mask."""
        # v can be added if it doesn't conflict with any vertex in s_mask
        return (conflict_mask[v] & s_mask) == 0

    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            bit = 1 << (v - 1)
            if result_mask & bit:
                continue
            if can_add(v, result_mask):
                result_mask |= bit
        return result_mask

    def mask_to_list(mask):
        return [v for v in vertices if (mask >> (v - 1)) & 1]

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    # --- Initial Solution Generation ---
    best_mask = 0

    # Degrees for heuristics
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    
    # Heuristic orders
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    
    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
    ]
    
    # Random shuffles
    for _ in range(100):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))
        
    for c in candidates:
        if bin(c).count('1') > bin(best_mask).count('1'):
            best_mask = c

    # --- Upper Bound Calculation (Complement Coloring) ---
    # For a graph G, the size of the Max Independent Set (MIS) is bounded by the chromatic number of the complement graph \bar{G}.
    # We compute a greedy coloring of \bar{G} to get an upper bound.
    # \bar{G} has an edge (u,v) if u and v are NOT adjacent in G (and u!=v).
    # So, a color class in \bar{G} is a set of vertices that form a Clique in G.
    # Wait, if u,v are adjacent in \bar{G}, they are NOT adjacent in G.
    # A proper coloring of \bar{G} assigns colors such that adjacent vertices in \bar{G} have different colors.
    # This means vertices with the same color in \bar{G} are NOT adjacent in \bar{G}, which means they ARE adjacent in G?
    # No.
    # Let's re-verify.
    # Independent Set in G: Set S such that no two vertices in S are adjacent in G.
    # This is equivalent to a Clique in \bar{G}.
    # We want to find the largest Clique in \bar{G}.
    # The size of the largest Clique in a graph is bounded by the number of colors in a proper coloring of that graph?
    # No, the size of the largest Clique is bounded by the Chromatic Number of the graph?
    # Yes, because a clique requires all distinct colors in a proper coloring.
    # So, MIS(G) = MaxClique(\bar{G}) <= ChromaticNumber(\bar{G}).
    
    # We need to compute ChromaticNumber(\bar{G}) or an upper bound on it (i.e., a valid coloring with k colors).
    # Greedy coloring of \bar{G}:
    # Adjacency in \bar{G}: (1 << (u-1)) is in adj_bar[v] if u != v and NOT (conflict_mask[v] has bit u).
    # Note: conflict_mask[v] has bit (u-1) set if u conflicts with v in G.
    # So in \bar{G}, v is connected to u if NOT (conflict_mask[v] >> (u-1)) & 1.

    def get_upper_bound(s_mask):
        """
        Returns an upper bound on the size of MIS in the subgraph induced by s_mask.
        Uses greedy coloring on the complement graph induced by s_mask.
        """
        if s_mask == 0:
            return 0
        
        remaining = s_mask
        colors = 0
        
        while remaining:
            colors += 1
            # Find a maximal independent set in \bar{G} induced by remaining.
            # This is equivalent to finding a maximal clique in G induced by remaining.
            # We want to greedily pick vertices for this color class.
            # A vertex v can be added to the current color class if it is NOT adjacent to any vertex already in the class in \bar{G}.
            # In \bar{G}, v is adjacent to u if they are NOT adjacent in G.
            # So v can be added to the color class if it IS adjacent to all vertices already in the class in G?
            # No, that's for clique in G.
            # Let's stick to the definition:
            # Color class in \bar{G} is an Independent Set in \bar{G}.
            # Independent Set in \bar{G} is a set of vertices no two of which are adjacent in \bar{G}.
            # u, v not adjacent in \bar{G} means u, v ARE adjacent in G (or u==v).
            # So a color class in \bar{G} is a Clique in G.
            # We want to find a maximal Clique in G induced by remaining.
            
            color_class = 0
            # To find a maximal clique in G, we can greedily add vertices.
            # A vertex v can be added to the clique if it is adjacent to ALL vertices in the current clique.
            # In G, v is adjacent to u if (conflict_mask[v] >> (u-1)) & 1.
            
            # Heuristic for order: pick vertex with highest degree in G within remaining?
            # Or just iterate?
            # Let's iterate through remaining vertices.
            # To make it fast, we can precompute "neighbors in G" mask.
            # neighbors_in_G[v] = conflict_mask[v]
            
            # We need to check if v is connected to all in color_class.
            # This is equivalent to: color_class is a subset of neighbors_in_G[v].
            # i.e., color_class & ~neighbors_in_G[v] == 0.
            
            # We iterate through remaining vertices.
            # To improve efficiency, we can sort by degree in G descending?
            # High degree vertices are more likely to connect to others.
            
            temp_remaining = remaining
            # We might want to process vertices in a specific order.
            # Let's just iterate through bits.
            # Optimization: If we pick a vertex, we remove it from candidates for this color class?
            # No, we remove it from 'remaining' for the NEXT color.
            # But for the current color class, we just need to ensure connectivity.
            
            # Let's try a simple greedy:
            # Pick the vertex in remaining with the highest degree in G (within remaining? or global?).
            # Global degree is precomputed.
            # Let's just iterate.
            
            # Actually, finding a maximal clique is NP-hard, but greedy is fast.
            # Greedy:
            # 1. Start with empty clique.
            # 2. Pick a vertex v from remaining.
            # 3. Add v to clique.
            # 4. Remove from 'candidates for this color' all vertices not connected to v.
            # 5. Repeat until no candidates.
            
            candidates_for_color = remaining
            clique = 0
            
            while candidates_for_color:
                # Pick a vertex. Heuristic: highest degree in G?
                # Let's just pick the one with the lowest index for determinism/speed, or random?
                # Random might help avoid bad local minima in coloring.
                # But we want a TIGHT bound, so we want LARGE color classes (large cliques in G).
                # Picking high-degree vertices first might help form larger cliques.
                
                # Let's find the vertex in candidates_for_color with max degree in G.
                # This is O(N) per pick, O(N^2) per color, O(N^3) total. 256^3 is 16M, ok.
                
                best_v = 0
                best_deg = -1
                c = candidates_for_color
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = degrees[v] # Global degree is a proxy
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    c ^= lsb
                    
                if best_v == 0:
                    break
                    
                # Add best_v to clique
                clique |= (1 << (best_v - 1))
                
                # Update candidates: keep only those connected to best_v
                # connected to best_v in G means bit (best_v-1) is set in conflict_mask[u]
                # So we want u such that (conflict_mask[u] >> (best_v-1)) & 1
                # This is equivalent to: u is in the set of neighbors of best_v.
                # neighbors_of_best_v = conflict_mask[best_v]
                # We need candidates_for_color & neighbors_of_best_v
                candidates_for_color &= conflict_mask[best_v]
                
                # Also remove best_v from candidates (it's already in clique)
                # It's effectively removed because it's not connected to itself (no self-loops in conflict_mask usually, but let's be safe)
                # conflict_mask[best_v] does not have bit (best_v-1) set.
                
            # Remove clique from remaining
            remaining &= ~clique
            
        return colors

    # --- Local Search ---
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try adding each excluded vertex
            excluded_mask = all_mask & ~s_mask
            excluded = mask_to_list(excluded_mask)
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << (v - 1))
                    improved = True
            
            # Phase 2: Try 1-for-2 swaps
            if time.time() < time_limit:
                s_list = mask_to_list(s_mask)
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    bit_u = 1 << (u - 1)
                    temp_mask = s_mask & ~bit_u
                    # Find addable vertices
                    addable_mask = all_mask & ~s_mask
                    addable = []
                    # Iterate through addable_mask
                    m = addable_mask
                    while m:
                        lsb = m & -m
                        v_idx = lsb.bit_length() - 1
                        v = v_idx + 1
                        if can_add(v, temp_mask):
                            addable.append(v)
                        m ^= lsb
                        
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 30)
                        for i in range(limit):
                            v1 = addable[i]
                            for j in range(i + 1, limit):
                                v2 = addable[j]
                                # Check if v1 and v2 conflict
                                if (conflict_mask[v1] >> (v2 - 1)) & 1 == 0:
                                    s_mask = temp_mask | (1 << (v1 - 1)) | (1 << (v2 - 1))
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                                
            # Phase 3: Try 2-for-3 swaps (if time permits)
            if time.time() < time_limit - 5.0:
                s_list = mask_to_list(s_mask)
                random.shuffle(s_list)
                # Limit checks to avoid slowdown
                checks = 0
                for i in range(len(s_list)):
                    if time.time() >= time_limit:
                        break
                    if checks > 50:
                        break
                    u1 = s_list[i]
                    for j in range(i + 1, len(s_list)):
                        if checks > 50:
                            break
                        u2 = s_list[j]
                        checks += 1
                        temp_mask = s_mask & ~(1 << (u1 - 1)) & ~(1 << (u2 - 1))
                        
                        addable_mask = all_mask & ~s_mask
                        addable = []
                        m = addable_mask
                        while m:
                            lsb = m & -m
                            v_idx = lsb.bit_length() - 1
                            v = v_idx + 1
                            if can_add(v, temp_mask):
                                addable.append(v)
                            m ^= lsb
                            
                        if len(addable) >= 3:
                            random.shuffle(addable)
                            found = False
                            limit = min(len(addable), 20)
                            # Find 3 non-conflicting vertices
                            for i1 in range(limit):
                                v1 = addable[i1]
                                for i2 in range(i1 + 1, limit):
                                    v2 = addable[i2]
                                    if (conflict_mask[v1] >> (v2 - 1)) & 1:
                                        continue
                                    for i3 in range(i2 + 1, limit):
                                        v3 = addable[i3]
                                        if (conflict_mask[v1] >> (v3 - 1)) & 1:
                                            continue
                                        if (conflict_mask[v2] >> (v3 - 1)) & 1:
                                            continue
                                        s_mask = temp_mask | (1 << (v1 - 1)) | (1 << (v2 - 1)) | (1 << (v3 - 1))
                                        improved = True
                                        found = True
                                        break
                                    if found: break
                                if found: break
                            if found: break
        return s_mask

    # Run local search from best candidate
    time_limit = deadline - 5.0
    best_mask = local_search(best_mask, time_limit)
    
    best_size = bin(best_mask).count('1')
    
    # --- Branch and Bound / Simulated Annealing Hybrid ---
    # We use the upper bound to prune.
    # We perform a search that tries to improve the solution.
    
    # Simulated Annealing style restarts
    while time.time() < deadline - 5.0:
        # Create perturbation
        perturbed_mask = best_mask
        s_list = mask_to_list(perturbed_mask)
        if s_list:
            # Remove 1-5 random vertices
            n_remove = min(random.randint(1, 5), len(s_list))
            to_remove = random.sample(s_list, n_remove)
            for v in to_remove:
                perturbed_mask &= ~(1 << (v - 1))
        
        # Run local search on perturbed
        result_mask = local_search(perturbed_mask, deadline - 5.0)
        
        if bin(result_mask).count('1') > best_size:
            best_mask = result_mask
            best_size = bin(best_size).count('1') # Fix: best_size = bin(result_mask).count('1')
            best_size = bin(result_mask).count('1')
            
            # If we hit the upper bound, we can stop?
            # Let's check upper bound on the full set once.
            # Actually, let's check upper bound on the current best to see if we can stop.
            # If best_size == UB, we are optimal.
            ub = get_upper_bound(all_mask)
            if best_size >= ub:
                break

    # Final validation
    result_list = mask_to_list(best_mask)
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                # Fallback to a known valid set (e.g., all odds) if validation fails
                return list(range(1, 257, 2))

    return result_list
# EVOLVE-BLOCK-END
