import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 256
    
    # Build squares up to 2*N
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build conflict graph (where u+v is square)
    # We want Max Independent Set in G_conflict
    # This is Max Clique in G_complement (where u+v is NOT square)
    
    # Bitset representation for N=256 fits in 4 64-bit ints or just Python int
    # Python int is arbitrary precision, so we can use a single int for 256 bits.
    
    # adj_complement[i] is a bitmask of nodes j such that i and j are NOT in conflict
    # i.e., i+j is NOT a square.
    # Note: i is not in conflict with itself? 
    # Task says: "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
    # So if 2*i is a square, i cannot be in the set.
    # In the complement graph for Clique:
    # We need a set of vertices such that for all u,v in set, u is compatible with v.
    # Compatibility: u+v is not a square.
    # If u=v, we need 2u not square.
    # So, a vertex i is "valid" only if 2*i is not a square.
    # If 2*i is a square, i cannot be part of the clique.
    
    valid_nodes = [i for i in range(1, N + 1) if 2 * i not in squares]
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    
    # Build adjacency bitmasks for the complement graph (compatibility graph)
    # adj[i] has bit j set if valid_nodes[i] and valid_nodes[j] are compatible (sum not square)
    adj = [0] * n
    for i in range(n):
        u = valid_nodes[i]
        for j in range(n):
            v = valid_nodes[j]
            if u + v not in squares:
                adj[i] |= (1 << j)
                
    # We are looking for Max Clique in this graph.
    # Branch and Bound with Greedy Coloring.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    # Precompute popcount for speed? Python's int.bit_count() is fast.
    
    def expand(P, size, depth):
        """
        Recursive B&B for Max Clique.
        P: bitmask of candidate vertices.
        size: current clique size.
        depth: recursion depth for pruning/timeout.
        """
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if size > best_clique_size:
                best_clique_size = size
                # We need to reconstruct the clique, or just store the mask.
                # Storing mask is easier.
                # But we pass P down, we don't accumulate mask easily in this structure.
                # Let's modify to pass current_mask.
            return

        # Greedy coloring of the subgraph induced by P
        # Colors are assigned such that adjacent vertices have different colors.
        # We use a simple greedy coloring:
        # Order vertices by degree (or just index) and assign smallest available color.
        # Actually, for the bound to work, we need to order vertices by color.
        
        # Standard Tomita coloring:
        # 1. Get list of vertices in P.
        # 2. Sort by degree in P (descending) to get better colors? Or just iterate.
        # Let's use a simple greedy coloring that produces an upper bound.
        
        uncolored = P
        colors = {} # vertex_idx -> color
        color_count = 0
        # We will assign colors 1, 2, ...
        # We need to ensure that if we pick a vertex with color c, the max clique size 
        # from that vertex and its successors is <= c.
        
        # Efficient coloring for bitsets:
        # We pick a vertex, assign it a color, remove it and its neighbors from the "current color class candidate" set.
        # Wait, standard greedy coloring for bound:
        # We want to partition P into independent sets (color classes).
        # The number of colors is an upper bound on the clique size? 
        # No, the number of colors is an upper bound on the *chromatic number*.
        # And Chromatic Number >= Clique Number.
        # So Color Count is an UPPER BOUND on Clique Size.
        
        # Algorithm:
        # while uncolored:
        #   color_count += 1
        #   current_color_set = uncolored
        #   while current_color_set:
        #      pick v from current_color_set (e.g., highest degree)
        #      colors[v] = color_count
        #      current_color_set = current_color_set & ~adj[v] # remove neighbors
        #      current_color_set = current_color_set & ~(1<<v) # remove v
        #   uncolored = uncolored & ~(all vertices assigned color_count)
        
        # This is O(N^2) per node, which might be slow.
        # N=256. 256^2 = 65k. Depth maybe 50? 3M ops. Should be fine in Python if optimized.
        
        # Optimization: We don't need to store all colors if we just want the bound and order.
        # We can store the order and the color of each vertex in that order.
        
        vertex_order = []
        vertex_colors = []
        
        uncolored = P
        c = 0
        while uncolored:
            c += 1
            current_class = uncolored
            assigned_in_class = 0
            while current_class:
                # Pick a vertex. Let's pick one with max degree in uncolored? 
                # Or just the lowest index for determinism?
                # Max degree usually gives fewer colors.
                # Let's just pick the lowest set bit for speed.
                v_bit = current_class & -current_class
                v_idx = v_bit.bit_length() - 1
                
                vertex_order.append(v_idx)
                vertex_colors.append(c)
                
                # Remove v and its neighbors from current_class (they can't share color c)
                current_class &= ~adj[v_idx]
                current_class &= ~v_bit
                
                assigned_in_class |= v_bit
                
            uncolored &= ~assigned_in_class
            
        # Now we have vertex_order and vertex_colors.
        # The vertices are ordered such that vertex_order[i] has color vertex_colors[i].
        # The colors are non-decreasing? Not necessarily in this simple loop, but 
        # usually we process color 1, then 2, etc. So vertex_colors will be 1,1,1,2,2,3...
        # Actually, my loop assigns color c to all vertices in the class before incrementing c.
        # So vertex_colors will be [1, 1, ..., 2, 2, ..., 3, ...].
        
        # We iterate in REVERSE order of the list.
        # Why reverse?
        # The bound is: if we pick vertex v with color c, the max clique we can add is at most c.
        # Vertices with higher colors were processed later.
        # In the standard Tomita, vertices are ordered by color, and we go from high color to low.
        # If we pick v with color c, any clique containing v can only use vertices from color classes 1..c.
        # But since we are branching, we remove v and its neighbors.
        
        # Let's iterate from the end of the list (highest color) to the start.
        for i in range(len(vertex_order) - 1, -1, -1):
            v_idx = vertex_order[i]
            v_color = vertex_colors[i]
            
            # Pruning: if current size + max possible additional (which is v_color) <= best, stop.
            if size + v_color <= best_clique_size:
                break
                
            # Branch: Include v_idx
            # New candidates: neighbors of v_idx that are in P and have index > i? 
            # No, in the Tomita algorithm, we maintain the candidate set P.
            # When we pick v, the new P is P & N(v).
            # But we also need to ensure we don't pick vertices we've already considered in this loop?
            # The standard recursive call is expand(P & N(v), size + 1).
            # However, we must remove v from P for the next iteration of the loop (Exclude branch).
            
            # Let's define P_local = P.
            # In the loop, we are effectively removing vertices from P as we go (Exclude branch).
            # So P should be updated.
            
            # But P is passed by value (int).
            # We need to maintain the state of P for the "Exclude" branch.
            # Let's pass P as a mutable list? No, int is better.
            # We can just update a local variable P_curr.
            
            # Actually, the loop structure in Tomita is:
            # for v in ordered_P:
            #   if size + color(v) <= best: break
            #   expand(P & N(v), size+1)
            #   P = P \ {v}
            
            # So we need to update P.
            # Since P is an int, we can just do P &= ~(1 << v_idx) after the recursive call.
            
            # BUT, the recursive call needs the *current* P (before removing v? No, P & N(v) handles it).
            # The recursive call gets a subset of the current P.
            
            # Let's implement this carefully.
            # P is the set of candidates available at the start of this expand call.
            # We iterate through vertices in P.
            
            # We need to pass the *current* P to the recursive call?
            # Yes, P & N(v).
            
            # Update P for the next iteration (Exclude v)
            # P = P & ~(1 << v_idx)
            
            # Wait, if we modify P, we lose the original P for the next vertex?
            # No, that's the point. We are trying all possibilities.
            # If we include v, we recurse.
            # If we exclude v, we continue the loop with v removed from P.
            
            # So:
            new_P = P & adj[v_idx]
            expand(new_P, size + 1, depth + 1)
            
            # Remove v from P for the next iteration
            P &= ~(1 << v_idx)
            
            # Check time
            if time.time() > deadline:
                return

    # Initial call
    # Start with all valid nodes
    initial_P = (1 << n) - 1
    expand(initial_P, 0, 0)
    
    # Reconstruct best clique
    # We only stored best_clique_size. We need the set.
    # To get the set, we can run a greedy algorithm or just return a valid set of that size.
    # Since we want to maximize cardinality, and the score is cardinality, we just need ANY set of that size.
    # However, the evaluator might check if the returned set is valid.
    # We should return a valid set.
    # We can find one by running a greedy algorithm to find a clique of size best_clique_size.
    # Or, we can modify expand to store the actual mask when we update best.
    
    # Let's restart with a version that stores the mask.
    # For speed, I'll just use a greedy method to find a large independent set, 
    # and if B&B found a larger one, I'll try to construct it?
    # No, B&B is the point. Let's store the mask.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    def expand2(P, size, depth):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if size > best_clique_size:
                best_clique_size = size
                # We need to reconstruct the mask. 
                # This is tricky without passing the mask down.
                # Alternative: Just return the size, and then use a separate greedy to find a set of that size?
                # Greedy might not find it.
                # Let's pass the mask.
            return

        uncolored = P
        vertex_order = []
        vertex_colors = []
        
        c = 0
        while uncolored:
            c += 1
            current_class = uncolored
            assigned_in_class = 0
            while current_class:
                v_bit = current_class & -current_class
                v_idx = v_bit.bit_length() - 1
                
                vertex_order.append(v_idx)
                vertex_colors.append(c)
                
                current_class &= ~adj[v_idx]
                current_class &= ~v_bit
                assigned_in_class |= v_bit
                
            uncolored &= ~assigned_in_class
            
        for i in range(len(vertex_order) - 1, -1, -1):
            v_idx = vertex_order[i]
            v_color = vertex_colors[i]
            
            if size + v_color <= best_clique_size:
                break
                
            new_P = P & adj[v_idx]
            # We don't pass the mask, we just track size.
            # To get the mask, we can backtrack? 
            # Easier: Just use the best size found, and then run a greedy to find a set of that size.
            # Since the graph is small, a greedy with random restarts might find a clique of size best_clique_size.
            # Or we can just return a greedy result if it's close.
            # But B&B gives the EXACT max.
            
            # Let's modify to store the mask.
            # We can pass a list of selected vertices.
            
            expand2(new_P, size + 1, depth + 1)
            
            P &= ~(1 << v_idx)
            
            if time.time() > deadline:
                return

    # To get the mask, I will use a different strategy:
    # 1. Find Max Size with B&B.
    # 2. Use a greedy algorithm to find a set of that size.
    #    Greedy: Pick a vertex, add to set, remove neighbors. Repeat.
    #    If greedy finds size == best_clique_size, we are done.
    #    If not, we might need to search.
    #    Given N=256, greedy usually finds near-optimal.
    
    expand2((1 << n) - 1, 0, 0)
    
    # Greedy to find a set of size best_clique_size
    result_mask = 0
    candidates = (1 << n) - 1
    current_size = 0
    
    # Try to build a clique of size best_clique_size
    # We can use a simple backtracking to find a clique of exactly best_clique_size if greedy fails?
    # No, just use greedy. If it fails, the B&B size might be slightly over what greedy finds, 
    # but B&B size is the TRUE max.
    # Wait, if B&B says max is 100, and greedy finds 99, we should return 99?
    # No, we want to maximize. We should try to find 100.
    # Let's use a recursive search to find a clique of size best_clique_size.
    
    def find_clique_of_size(P, k, path):
        if k == 0:
            return True
        if P == 0 or P.bit_count() < k:
            return False
        
        # Pick a vertex
        # Heuristic: pick vertex with high degree
        # Just pick lowest bit
        v_bit = P & -P
        v_idx = v_bit.bit_length() - 1
        
        # Include v
        path.append(v_idx)
        if find_clique_of_size(P & adj[v_idx], k - 1, path):
            return True
        path.pop()
        
        # Exclude v
        P &= ~v_bit
        return find_clique_of_size(P, k, path)
        
    path = []
    if find_clique_of_size((1 << n) - 1, best_clique_size, path):
        for idx in path:
            result_mask |= (1 << idx)
    else:
        # Fallback to greedy
        candidates = (1 << n) - 1
        while candidates:
            # Pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            c = candidates
            while c:
                v_bit = c & -c
                v_idx = v_bit.bit_length() - 1
                deg = (adj[v_idx] & candidates).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v_idx
                c &= ~v_bit
            result_mask |= (1 << best_v)
            candidates &= ~adj[best_v]
            candidates &= ~(1 << best_v)
            
    # Convert mask to list of nodes
    result = []
    for i in range(n):
        if (result_mask >> i) & 1:
            result.append(valid_nodes[i])
            
    result.sort()
    
    # Verify
    if not result:
        return [1]
        
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known small valid set
        # e.g. {1} is valid if 2 is not square (it's not).
        return [1]
        
    return result

# EVOLVE-BLOCK-END
