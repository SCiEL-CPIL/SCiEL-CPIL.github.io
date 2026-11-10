import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 512
    
    # 1. Build Graph
    # compat[i] is bitset of neighbors of i (including i)
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        compat[i] = 1 << i
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                compat[i] |= 1 << j
                compat[j] |= 1 << i

    # 2. Greedy Seed
    best_clique = []
    best_size = 0
    
    # Simple greedy: pick node with max degree in current available set
    available = sum(1 << i for i in range(1, N + 1))
    while available:
        # Find node in available with max degree in available
        best_node = 0
        max_deg = -1
        # Iterate through bits in available
        temp = available
        while temp:
            lsb = temp & -temp
            node = lsb.bit_length() - 1
            deg = (compat[node] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_node = node
            temp ^= lsb
        
        if best_node == 0: break
        
        best_clique.append(best_node)
        available &= compat[best_node]
        # Remove best_node from available? 
        # Actually, we add it to clique, so we restrict available to its neighbors.
        # The next node must be neighbor of best_node.
        # So available = available & compat[best_node] is correct.
        # But we must ensure we don't pick the same node again?
        # The bit for best_node is in compat[best_node].
        # So we should remove it?
        # If we keep it, we might pick it again? No, we update available.
        # Let's remove the picked node from available to avoid self-loop issues in logic, 
        # though lcm(i,i)=i<=N so it is compatible.
        # But a clique is a set of distinct nodes.
        available &= ~(1 << best_node)
        
    best_size = len(best_clique)
    best_clique_list = best_clique

    # 3. Tomita Search
    # We need to track best_size and best_clique_list globally or via closure
    
    def color_sort(P):
        # Returns list of (node, color)
        # Greedy coloring of induced subgraph P
        # Colors are 1, 2, ...
        # We want to return sorted by color descending
        nodes_in_P = [i for i in range(1, N + 1) if (P >> i) & 1]
        if not nodes_in_P:
            return []
        
        # Sort by degree in P descending to help coloring
        nodes_in_P.sort(key=lambda x: (compat[x] & P).bit_count(), reverse=True)
        
        colors = {}
        used_colors = [False] * (len(nodes_in_P) + 1) # Max color is len(nodes)
        
        # Simple greedy:
        # Iterate nodes. Assign smallest color not used by neighbors in P.
        # This is O(V^2) in worst case, but V <= 512.
        # We can optimize by maintaining bitsets of used colors per node?
        # Or just check neighbors.
        
        # Let's do a standard greedy coloring
        color_of = {}
        max_color = 0
        # To speed up, we can process in order of degree
        for v in nodes_in_P:
            # Find neighbors of v in P that are already colored
            # Actually, we just need to know which colors are used by neighbors
            # Since we are coloring sequentially, we only care about neighbors earlier in the list?
            # No, greedy coloring usually processes in a specific order.
            # If we process in order of degree, we need to check all previously colored neighbors.
            
            # Let's maintain a set of forbidden colors for v
            forbidden = set()
            # Check all u in nodes_in_P such that u < v (in list order) and u is neighbor of v
            # This is inefficient if we don't precompute adjacency lists.
            
            # Alternative:
            # Use bitsets for color classes.
            # color_class[c] is bitset of nodes with color c
            # For node v, check if (compat[v] & color_class[c]) == 0 for c=1..max_color
            # If yes, assign color c. Else try next.
            
            color_classes = [0] * (len(nodes_in_P) + 1)
            # Re-initializing inside loop is bad.
            
        # Let's implement a more efficient coloring:
        # 1. Start with U = P
        # 2. Color = 1
        # 3. While U not empty:
        #    a. Find maximal independent set in U (greedily)
        #    b. Assign Color to these nodes
        #    c. Remove from U
        #    d. Color++
        
        # This produces a valid coloring.
        # The max color is an upper bound.
        
        U = P
        color = 1
        result = [] # (node, color)
        
        while U:
            # Find a node in U with max degree in U? Or just any?
            # To make independent set large, pick low degree?
            # Greedy MIS:
            MIS = 0
            temp_U = U
            # Pick node with max degree in U to add to MIS? No, max degree is bad for MIS.
            # Pick node with min degree in U?
            # Or just iterate and add if compatible with current MIS.
            
            # Let's just iterate through nodes in U and add to MIS if compatible with all in MIS
            # This finds *a* maximal independent set.
            
            # To be faster, we can pick a pivot.
            # But for N=512, simple loop is okay.
            
            # Let's try:
            # Iterate nodes in U.
            # If node is not adjacent to any node already in MIS, add to MIS.
            # This is O(V * |MIS|).
            
            # Actually, we can just pick one node, add to MIS, remove its neighbors from U?
            # No, that's for finding clique.
            
            # Let's use the standard Tomita coloring which is:
            # While U not empty:
            #   Pick v in U
            #   Add v to current color class
            #   Remove v and its neighbors from U? No, remove v from U.
            #   But we must ensure color class is independent.
            #   So we remove neighbors of v from the *candidate pool for this color*?
            #   Yes.
            
            # Let's implement:
            # Current Color Class C
            # Candidates for C: V_c
            # Initially V_c = U
            # While V_c not empty:
            #   Pick v from V_c (e.g., max degree in V_c)
            #   Add v to C
            #   Remove v and N(v) from V_c
            # After C is built, assign color to all in C
            # Remove C from U
            # Color++
            
            V_c = U
            C = []
            while V_c:
                # Pick v in V_c with max degree in V_c
                best_v = 0
                max_deg = -1
                temp = V_c
                while temp:
                    lsb = temp & -temp
                    node = lsb.bit_length() - 1
                    deg = (compat[node] & V_c).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_v = node
                    temp ^= lsb
                
                C.append(best_v)
                # Remove best_v and its neighbors from V_c
                V_c &= ~(compat[best_v]) # This removes best_v and all its neighbors
                # Wait, compat[best_v] includes best_v.
                # So V_c = V_c & ~compat[best_v] removes best_v and all neighbors.
                # This ensures C is independent.
            
            for v in C:
                result.append((v, color))
            
            # Remove C from U
            mask_C = 0
            for v in C:
                mask_C |= (1 << v)
            U &= ~mask_C
            
            color += 1
            
            if time.time() > deadline:
                return result # Return partial? Or break?
                # If we return partial, we might miss high colors.
                # But if we time out, we just stop search.
        
        # Sort result by color descending
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def expand(R_bits, P_bits, R_list):
        nonlocal best_size, best_clique_list
        
        if time.time() > deadline:
            return
            
        if P_bits == 0:
            if len(R_list) > best_size:
                best_size = len(R_list)
                best_clique_list = R_list[:]
            return
        
        # Prune: if len(R) + bit_count(P) <= best_size, return
        # This is a weak bound. Tomita uses color bound.
        
        colored = color_sort(P_bits)
        
        # colored is list of (v, c) sorted by c desc
        # Iterate
        for v, c in colored:
            if len(R_list) + c <= best_size:
                break # Since sorted desc, all subsequent have lower c
            
            new_R_list = R_list + [v]
            new_R_bits = R_bits | (1 << v)
            new_P_bits = P_bits & compat[v]
            
            expand(new_R_bits, new_P_bits, new_R_list)
            
            # Remove v from P_bits for subsequent iterations
            P_bits &= ~(1 << v)
            
            # Update R_bits? No, R_bits is passed by value (int is immutable).
            # But we need to ensure we don't reuse v.
            # The loop structure handles this by removing v from P_bits.
            
            if time.time() > deadline:
                return

    # Start search
    initial_P = sum(1 << i for i in range(1, N + 1))
    initial_R = 0
    initial_R_list = []
    
    # Use the greedy seed to initialize?
    # Tomita works best if we start from empty, but a good seed helps pruning.
    # We can start expand with the greedy clique as R.
    # But then P must be nodes compatible with all in R.
    
    seed_P = initial_P
    for v in best_clique_list:
        seed_P &= compat[v]
        # Also remove v? No, v is in R.
        # But v is compatible with itself.
        # We should remove nodes in R from P?
        # Yes, P should contain nodes not in R.
        
    # Actually, it's safer to run Tomita from scratch with the seed as a lower bound.
    # The seed provides the initial best_size.
    # The search starts from empty R.
    
    expand(0, initial_P, [])
    
    # Final check
    if not best_clique_list:
         # Fallback
         return [1]
         
    return best_clique_list
