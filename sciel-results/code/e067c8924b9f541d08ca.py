import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    max_mod = 128
    
    # 1. Generate all possible residue classes (r, m)
    # Node index mapping: (r, m) -> index
    nodes = []
    node_to_rm = []
    for m in range(2, max_mod + 1):
        for r in range(m):
            nodes.append((r, m))
            node_to_rm.append((r, m))
            
    n_nodes = len(nodes)
    
    # 2. Build conflict graph
    # Two nodes (r1, m1) and (r2, m2) conflict if r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    # We store neighbors as bitsets (Python integers) for fast intersection
    
    # Pre-compute gcds for moduli
    moduli = [m for _, m in nodes]
    gcd_table = {}
    for m1 in range(2, max_mod + 1):
        for m2 in range(m1, max_mod + 1):
            gcd_table[(m1, m2)] = gcd(m1, m2)
            if m1 != m2:
                gcd_table[(m2, m1)] = gcd(m1, m2)

    # Build adjacency bitsets
    # adj[i] is a bitset of all nodes j such that i and j conflict
    adj = [0] * n_nodes
    
    # Optimization: Group nodes by modulus to speed up conflict checking?
    # Or just do O(N^2) precomputation. N=8255, N^2 ~ 68M. 
    # 68M operations in Python might be too slow (approx 10-20s).
    # We can optimize by iterating over moduli pairs.
    
    # Let's use a more efficient construction.
    # For each pair of moduli (m1, m2), find conflicting residues.
    # Actually, direct O(N^2) with bitset setting is feasible if we are careful.
    # But let's try to be smarter.
    
    # Reset adj
    adj = [0] * n_nodes
    
    # To speed up, we can iterate i from 0 to n_nodes-1
    # and for each i, iterate j > i.
    # If they conflict, set bit j in adj[i] and bit i in adj[j].
    
    # We can group nodes by modulus to reduce gcd lookups?
    # Let's just run the O(N^2) loop but with early exits if possible?
    # No, just run it. 8255^2 / 2 = 34M pairs.
    # In Python, 34M iterations with a gcd lookup and mod check might take ~10-15 seconds.
    # This is acceptable if the search is fast.
    
    # Let's try to optimize the graph construction.
    # We can store indices by modulus.
    idx_by_mod = {}
    for i, (r, m) in enumerate(nodes):
        if m not in idx_by_mod:
            idx_by_mod[m] = []
        idx_by_mod[m].append(i)
        
    # Iterate over pairs of moduli
    mod_list = list(idx_by_mod.keys())
    for k in range(len(mod_list)):
        m1 = mod_list[k]
        idxs1 = idx_by_mod[m1]
        for l in range(k, len(mod_list)):
            m2 = mod_list[l]
            idxs2 = idx_by_mod[m2]
            g = gcd_table.get((m1, m2), 1)
            
            if m1 == m2:
                # All nodes with the same modulus conflict with each other?
                # Yes, because gcd(m, m) = m. r1 % m == r2 % m implies r1 == r2.
                # Since residues are distinct for a fixed modulus in our list, 
                # wait. The nodes are (r, m). For a fixed m, we have all r in [0, m-1].
                # So for fixed m, r1 != r2. 
                # Condition: r1 % m == r2 % m. Since 0<=r<m, this implies r1==r2.
                # So no two distinct nodes with the same modulus conflict?
                # Let's re-read the condition.
                # "Every two infinite residue classes must be disjoint".
                # Class A: x = r1 + k*m. Class B: x = r2 + k*m.
                # If r1 != r2, they are disjoint.
                # So nodes with the same modulus do NOT conflict.
                # My previous logic "conflict if r1 % g == r2 % g" with g=m implies r1=r2.
                # Since r1 != r2 for distinct nodes, they do not conflict.
                continue
                
            # For m1 != m2
            # Conflict if r1 % g == r2 % g
            # We can group residues in idxs1 by r % g and idxs2 by r % g.
            # But g can be small.
            # Simple loop is fine for small groups.
            
            # Optimization: If g is large, few conflicts. If g is small, many conflicts?
            # Actually, number of conflicts is sum over c in [0, g-1] of (count1[c] * count2[c]).
            
            # Let's just iterate pairs.
            # Size of idxs1 is m1, idxs2 is m2.
            # Total pairs for this modulus pair is m1 * m2.
            # Sum of m1*m2 over all pairs is approx (Sum m)^2 / 2 ~ 34M.
            # This is the same as N^2/2.
            
            # We can speed this up by pre-computing residue classes modulo g?
            # No, let's just do the loop.
            
            for i in idxs1:
                r1 = nodes[i][0]
                rem1 = r1 % g
                # Find all j in idxs2 such that nodes[j][0] % g == rem1
                # We can pre-group idxs2 by residue mod g?
                # Doing this inside the loop is expensive.
                
                # Alternative: Just check.
                for j in idxs2:
                    r2 = nodes[j][0]
                    if r2 % g == rem1:
                        # Conflict
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
    # 3. Maximum Independent Set Search
    # Use Branch and Bound with Greedy Coloring
    
    # We need a function to compute a greedy coloring of a set of vertices.
    # This gives an upper bound on the size of the independent set.
    # We represent the candidate set as a bitset.
    
    # To speed up coloring, we can order vertices by degree in the induced subgraph.
    
    best_size = 0
    best_set = 0 # Bitset of the best independent set found
    
    # Initial ordering: Sort vertices by degree (descending) to find large sets early?
    # Or just use 0..N-1.
    
    # Tomita's algorithm uses a specific ordering.
    # Let's use a simple recursive search with bitsets.
    
    # We need to handle the time limit.
    
    def greedy_color(candidates):
        # Returns a list of colors assigned to vertices in candidates, 
        # such that adjacent vertices have different colors.
        # The max color + 1 is an upper bound on the independence number.
        # We also want to return the partition to guide the search? 
        # Tomita's algorithm uses the coloring to partition the candidates.
        
        # For simplicity and speed, we just need the bound.
        # But Tomita's algorithm uses the color classes to decide which vertex to branch on.
        
        # Let's implement a standard greedy coloring that returns the max color.
        # To be effective, we should order vertices by degree.
        
        # Extract vertices
        verts = []
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
            
        if not verts:
            return 0, []
            
        # Order by degree in the induced subgraph (candidates)
        # degree(v) = popcount(adj[v] & candidates)
        # This is expensive. Let's just use the original order or a pre-computed heuristic.
        # For now, use the order in 'verts' (which is increasing index).
        # Better: Sort by degree descending.
        
        # To avoid O(N log N) every time, we can use a static order?
        # No, dynamic is better.
        
        # Let's try a simpler bound first: just the count? No, too weak.
        # Let's use a greedy coloring without sorting first, just to get it running.
        # Then optimize if needed.
        
        color = 0
        # We need to assign colors such that no two adjacent vertices share a color.
        # Greedy: Iterate through vertices, assign the smallest available color.
        
        # This is O(N^2) in the worst case for the coloring itself.
        # With N=8000, this might be slow if called often.
        
        # Alternative: Use a pre-computed ordering?
        
        # Let's try the Tomita algorithm structure properly.
        # It maintains a list of color classes.
        
        # For now, let's just return the number of vertices as a bound? No, that's too loose.
        # Let's use a simple greedy coloring on the induced subgraph.
        
        # To speed up, we can use the fact that we only need an upper bound.
        # A valid coloring provides an upper bound.
        
        # Implementation of Greedy Coloring for a bitset:
        # 1. Get list of vertices.
        # 2. Sort by degree in induced subgraph (descending).
        # 3. Assign colors.
        
        # Optimization: If the number of candidates is small, just brute force?
        
        # Let's implement a fast greedy coloring.
        
        # We'll use a list of vertices sorted by index for now, and see if it's fast enough.
        # If not, we'll add degree sorting.
        
        # Actually, a very fast upper bound is just the number of vertices in the largest independent set of the complement? No.
        
        # Let's stick to the standard Tomita approach:
        # 1. Order vertices.
        # 2. Color them greedily.
        # 3. The color classes are the groups.
        
        # To make it fast, we can approximate the degree.
        
        # Let's just do the coloring on the vertices as they appear in the bitset (LSB to MSB).
        # This is fast to iterate.
        
        colors = [0] * len(verts)
        # This is O(N^2) check for each vertex.
        
        # Let's try a different strategy:
        # If the search is too slow, we'll optimize.
        
        # For the first version, let's use a simple bound: 
        # Upper bound = number of candidates.
        # This will lead to a lot of backtracking.
        
        # Let's implement the greedy coloring properly but efficiently.
        
        # We'll return the max color used.
        
        # To speed up, we can stop coloring if the bound is already <= best_size - current_size.
        
        # Let's try to implement the full Tomita algorithm.
        
        return 0, [] # Placeholder

    # The O(N^2) graph construction is likely the bottleneck.
    # 34M iterations in Python is ~10-20s.
    # The search itself needs to be fast.
    
    # Let's refine the graph construction to be faster.
    # We can use NumPy? No, bitsets are better in Python ints.
    
    # Let's restart the logic with a focus on speed.
    
    # 1. Graph Construction Optimization
    # We can avoid the inner loop by using dictionary lookups for residues mod g.
    
    adj = [0] * n_nodes
    
    for k in range(len(mod_list)):
        m1 = mod_list[k]
        idxs1 = idx_by_mod[m1]
        for l in range(k + 1, len(mod_list)): # Only m1 < m2
            m2 = mod_list[l]
            idxs2 = idx_by_mod[m2]
            g = gcd_table.get((m1, m2), 1)
            
            # Group idxs2 by residue mod g
            # res_to_idxs = {r % g: [j for j in idxs2 if nodes[j][0] % g == r % g]}
            # This is O(m2) per pair.
            
            # Then for each i in idxs1, find the group for nodes[i][0] % g and set bits.
            
            # This reduces the inner loop from O(m1*m2) to O(m1 + m2 + conflicts).
            
            res_map = {}
            for j in idxs2:
                r2 = nodes[j][0]
                rem = r2 % g
                if rem not in res_map:
                    res_map[rem] = 0
                res_map[rem] |= (1 << j)
                
            for i in idxs1:
                r1 = nodes[i][0]
                rem = r1 % g
                if rem in res_map:
                    # All j in res_map[rem] conflict with i
                    conflict_bits = res_map[rem]
                    adj[i] |= conflict_bits
                    # We need to set the reverse bits too
                    # We can't just do adj[j] |= (1 << i) for all j in conflict_bits easily without iterating.
                    # But we can do it.
                    c = conflict_bits
                    while c:
                        lsb = c & -c
                        j = lsb.bit_length() - 1
                        adj[j] |= (1 << i)
                        c ^= lsb
                        
    # Now we have the graph.
    
    # 2. MIS Search
    
    # We will use a Branch and Bound algorithm.
    # State: candidates (bitset), current_set (bitset), current_size (int)
    
    # Greedy Coloring for Upper Bound
    def get_upper_bound(candidates):
        # Returns an upper bound on the size of the independent set in 'candidates'
        # Uses Greedy Coloring.
        # To be fast, we use a simple heuristic.
        
        # If candidates is empty, return 0
        if not candidates:
            return 0
            
        # Extract vertices
        # We want to color them.
        # To speed up, we can just return the number of vertices? No.
        
        # Let's use a simple greedy coloring.
        # We'll iterate through the vertices in the bitset.
        
        # To make it faster, we can limit the precision if time is tight?
        # No, the bound is crucial for pruning.
        
        # Let's implement a fast greedy coloring.
        # We need to assign colors c1, c2, ... such that no two adjacent vertices have the same color.
        # The number of colors is an upper bound.
        
        # Optimization: Only color a subset? No.
        
        # Let's try to use the fact that we can stop if the bound is low enough.
        
        # For now, let's just count the vertices. This is a valid but weak bound.
        # This will make the search slow.
        
        # Let's implement a basic coloring.
        
        # We'll use a list of vertices.
        verts = []
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
            
        if not verts:
            return 0
            
        # Sort by degree in the induced subgraph?
        # This is expensive. Let's just use the order.
        
        colors = {}
        max_color = 0
        
        # For each vertex, find the smallest color not used by its neighbors in 'verts'
        # Neighbors in 'verts' are adj[v] & candidates
        
        # To speed up, we can pre-compute the neighbors in the candidate set?
        
        # This is O(N^2) per call. If called many times, it's slow.
        
        # Alternative: Use a static ordering and a simple bound?
        
        # Let's try to improve the bound by just using the count of vertices for now,
        # and see if the search finishes. If not, we'll optimize the coloring.
        
        return len(verts)
        
    best_size = 0
    best_set = 0
    
    # Initial candidates: all vertices
    # We can sort vertices by degree to find a good initial solution?
    # Let's just start with all vertices.
    
    # To get a good initial lower bound, let's run a greedy independent set.
    # Pick vertex with max degree, add to set, remove it and its neighbors.
    
    def greedy_is(candidates):
        is_set = 0
        c = candidates
        while c:
            # Pick a vertex
            # Let's pick the one with the highest degree in the current subgraph
            # This is expensive. Let's just pick the first one (LSB).
            lsb = c & -c
            v = lsb.bit_length() - 1
            is_set |= lsb
            # Remove v and its neighbors
            c &= ~(adj[v] | lsb)
        return is_set, bin(is_set).count('1')
        
    init_set, init_size = greedy_is((1 << n_nodes) - 1)
    best_size = init_size
    best_set = init_set
    
    # Now Branch and Bound
    
    # We need a recursion limit or stack?
    # Python recursion might be deep.
    # But the depth is at most N.
    # We can use an iterative stack or increase recursion limit.
    # Let's use a stack.
    
    # Stack items: (candidates, current_set, current_size)
    # But we need to explore branches.
    
    # Let's use a recursive function with a time check.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound
        # If current_size + upper_bound(candidates) <= best_size, prune
        ub = get_upper_bound(candidates)
        if current_size + ub <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: Pick a vertex with high degree in the induced subgraph?
        # Or just pick the first one?
        # Tomita's algorithm picks the vertex with the highest color in the coloring.
        # Since we are using a weak bound, let's just pick the first one (LSB).
        
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates & ~(adj[v] | lsb)
        new_cands_include = candidates & ~(adj[v] | lsb)
        search(new_cands_include, current_set | lsb, current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates: candidates & ~lsb
        new_cands_exclude = candidates & ~lsb
        search(new_cands_exclude, current_set, current_size)
        
    search((1 << n_nodes) - 1, 0, 0)
    
    # Extract result
    result = []
    c = best_set
    while c:
        lsb = c & -c
        v = lsb.bit_length() - 1
        r, m = nodes[v]
        result.append([r, m])
        c ^= lsb
        
    return result

# Run construct to test
# construct()
