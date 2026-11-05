import time
import random
import numpy as np

def construct():
    n = 40
    deadline = time.time() + 115.0
    
    # Bitset representation of adjacency
    adj = [0] * n
    # A2[i] is an integer where bit j is 1 if there is a path of length 2 from i to j
    A2 = [0] * n
    
    def update_A2():
        for i in range(n):
            mask = 0
            a = adj[i]
            while a:
                lsb = a & -a
                k = lsb.bit_length() - 1
                mask |= adj[k]
                a ^= lsb
            A2[i] = mask

    def check_c6_add(u, v):
        # Returns True if adding (u,v) creates a C6
        # A C6 is formed if there is a path of length 3 between u and v in the existing graph.
        # Path length 3: u-x-y-v. 
        # This is equivalent to checking if N(u) intersects A2(v) or N(v) intersects A2(u).
        # Actually, the condition for NO C6 when adding (u,v) is that there is NO path of length 3 between u and v.
        # Let's verify:
        # If we add (u,v), a cycle u-v-...-u of length 6 exists iff there is a path of length 5 between u and v in G.
        # Wait. A cycle of length 6 is u-v-w-x-y-z-u.
        # If we add edge (u,v), we need to ensure there is no path of length 5 between u and v in G.
        # Path length 5: u-a-b-c-d-v.
        # This is equivalent to N(u) intersecting A4(v) != 0?
        # Or simpler: 
        # Let's re-read the seed's check_c6_add.
        # Seed: 
        #   x in N(u), w in N(v). Check if path of length 3 between x and w.
        #   Path x-...-w of length 3: x-p-q-w.
        #   So cycle: u-x-p-q-w-v-u. Length 6.
        #   Yes, this is correct.
        #   Condition: N(u) contains x, N(v) contains w, and dist(x,w) <= 3? No, exactly 3?
        #   If dist(x,w) is 1, cycle is u-x-w-v-u (length 4). C4 is not forbidden? 
        #   Task says "no six-cycle". It does not say "no cycles of length < 6".
        #   However, usually "C6-free" means no induced C6? Or just no C6?
        #   "containing no six-cycle". This usually means no subgraph isomorphic to C6.
        #   If there is a C4, adding an edge might create a C6?
        #   Let's stick to the seed's logic which was scored 1.0 (valid).
        #   Seed logic: If exists x in N(u), w in N(v) such that there is a path of length 3 between x and w, then adding (u,v) creates a C6.
        #   Path of length 3 between x and w means dist(x,w) = 3? Or <= 3?
        #   If dist(x,w)=1, x-w is an edge. Cycle u-x-w-v-u is C4. Not C6.
        #   If dist(x,w)=2, x-p-w. Cycle u-x-p-w-v-u is C5. Not C6.
        #   If dist(x,w)=3, x-p-q-w. Cycle u-x-p-q-w-v-u is C6.
        #   So we need to check if there is a path of length EXACTLY 3?
        #   Or if there is a path of length 3, it implies dist <= 3.
        #   If dist < 3, no C6 is formed by that specific path extension.
        #   However, if dist < 3, the graph has shorter cycles.
        #   The seed checks `if adj[x] & A2[w]`.
        #   A2[w] contains nodes at distance 2 from w.
        #   adj[x] contains nodes at distance 1 from x.
        #   Intersection means there is a node z such that z ~ x and dist(z, w) = 2.
        #   So x-z-...-w is a path of length 1+2=3.
        #   So this checks for a path of length 3.
        #   It does NOT distinguish between dist=3 and dist<3.
        #   If dist(x,w)=1, then w in adj[x]. Is w in A2[x]? No, A2[x] is dist 2.
        #   Wait. `adj[x] & A2[w]`.
        #   If x and w are adjacent (dist 1), is there a z in adj[x] and A2[w]?
        #   If x-w is an edge, and x-z is an edge, and z-y-w is a path...
        #   This logic is slightly complex. Let's trust the seed's validity check as it passed.
        #   Actually, a more robust check for "no C6" when adding (u,v):
        #   There is no path of length 5 between u and v in G.
        #   Let's implement a faster check using bitsets.
        #   But the seed's check is O(D^2 * bitops). With D~5, it's very fast.
        #   Let's use the seed's check for safety, but optimize the loop.
        
        a_u = adj[u]
        a_v = adj[v]
        
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            # We need to know if there is a path of length 3 from x to any w in N(v).
            # This is equivalent to: (A2[x] & a_v) != 0?
            # A2[x] has bits for nodes at dist 2 from x.
            # If w is in N(v) and in A2[x], then dist(x,w)=2.
            # Then path u-x-...-w-v-u is length 1+2+1 = 4? No.
            # u-x (1), x-...-w (2), w-v (1). Total 4 edges. Cycle C5?
            # Wait.
            # Cycle: u - x - p - w - v - u.
            # Edges: (u,x), (x,p), (p,w), (w,v), (v,u). 5 edges. C5.
            # We want C6.
            # Cycle: u - x - p - q - w - v - u.
            # Edges: (u,x), (x,p), (p,q), (q,w), (w,v), (v,u). 6 edges.
            # So we need path of length 3 between x and w.
            # dist(x,w) = 3.
            # This means w is in A3[x].
            # We don't have A3.
            # Seed checks: `adj[x] & A2[w]`.
            # If z in adj[x] and z in A2[w], then dist(x,z)=1 and dist(z,w)=2.
            # So dist(x,w) <= 3.
            # If dist(x,w)=3, we have a C6.
            # If dist(x,w)=1, we have a C4. (u-x-w-v-u).
            # If dist(x,w)=2, we have a C5. (u-x-p-w-v-u).
            # Does the task forbid C4 and C5?
            # "containing no six-cycle".
            # Usually this means no C6 subgraph. C4 and C5 are allowed.
            # However, if the seed's check rejects C4/C5 creating edges, it might be overly conservative but valid.
            # If it rejects an edge that creates a C4, the resulting graph is still C6-free.
            # So it's a safe (though potentially suboptimal) check.
            # Let's stick to it to ensure validity.
            
            if adj[x] & A2[w_mask_placeholder]: # Placeholder for logic
                pass
        
        # Let's just copy the seed's exact logic for the check to be safe, 
        # but I will optimize the outer loop structure in the main search.
        
        # Re-implementing seed's check exactly:
        x_mask = a_u
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            w_mask = a_v
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                if adj[x] & A2[w]:
                    return True
        
        return False

    # 1. Initialization: Random 5-regular graph
    success = False
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        for i in range(n):
            adj[i] = 0
        stubs = []
        for i in range(n):
            for _ in range(5):
                stubs.append(i)
        random.shuffle(stubs)
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v or (adj[u] >> v) & 1:
                valid = False
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        if valid:
            success = True
            break
            
    if not success:
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    update_A2()
    
    best_edges = sum(bin(a).count("1") for a in adj) // 2
    best_adj = adj[:]

    # 2. Deterministic Greedy Set-Cover Search
    # Strategy:
    # Instead of random SA, we systematically find the "best" edge to add.
    # We define the "cost" of adding an edge (u,v) as the number of other candidate edges it blocks.
    # An edge (u,v) blocks (x,y) if adding (u,v) makes (x,y) invalid.
    # This is computationally expensive to calculate for all pairs.
    
    # Alternative Structural Strategy: 
    # Local Search with Deterministic Improvement.
    # 1. Try to add any valid edge. If found, add it.
    # 2. If no edge can be added, try to swap edges to create space.
    #    Swap: Remove (a,b), add (c,d) if valid.
    #    To be deterministic, we iterate through all possible swaps in a fixed order.
    #    If a swap allows a subsequent addition, we take it.
    
    # This is a "Hill Climbing with Deterministic Neighborhood Exploration".
    
    improved = True
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase A: Try to add edges
        # Iterate through all non-edges in a deterministic order (or random order for variety? 
        # Deterministic is safer for reproducibility, but random might escape local minima better.
        # Let's use a shuffled list of non-edges, but re-shuffle only if stuck.)
        
        # Actually, let's just scan all non-edges.
        # If we find one that is valid, add it and restart scan.
        # This is O(N^3) worst case per addition, but N=40 is small.
        
        found_addition = False
        for u in range(n):
            if time.time() > deadline - 2.0:
                break
            a_u = adj[u]
            # Iterate over non-neighbors of u
            # Mask of all nodes
            all_mask = (1 << n) - 1
            non_neighbors = all_mask & ~a_u & ~(1 << u)
            
            # We can iterate through bits of non_neighbors
            v_mask = non_neighbors
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask ^= lsb_v
                
                if u < v: # Avoid duplicates
                    if not check_c6_add(u, v):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        update_A2()
                        current_edges = sum(bin(a).count("1") for a in adj) // 2
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                        found_addition = True
                        improved = True
                        break
            if found_addition:
                break
                
        if found_addition:
            continue # Restart scan as graph changed
            
        # Phase B: No edges can be added. Try swaps.
        # We want to find a swap (remove a,b, add c,d) such that after swap, we can add more edges?
        # Or just such that the swap is valid and potentially leads to a better state?
        # Since we can't add edges, we are stuck.
        # A swap keeps edge count same. It only helps if it enables future additions.
        # How to evaluate if a swap enables future additions?
        # We can perform the swap, then run a quick "can we add any edge?" check.
        # If yes, keep the swap. If no, revert.
        
        swapped = False
        # Iterate through all edges (a,b)
        edges_list = []
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges_list.append((i, j))
                a ^= lsb
                
        random.shuffle(edges_list) # Add some randomness to avoid getting stuck in a deterministic loop
        
        for a, b in edges_list:
            if time.time() > deadline - 2.0:
                break
            
            # Backup
            adj_backup = adj[:]
            A2_backup = A2[:]
            
            # Remove (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            update_A2()
            
            # Try to add a random edge (c,d)
            # To be efficient, try a few random candidates
            c_addable = False
            for _ in range(20):
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d or (adj[c] >> d) & 1:
                    continue
                if not check_c6_add(c, d):
                    # Check if this swap leads to a state where we can add MORE edges?
                    # Actually, if we just swap, edge count is same.
                    # We only accept if it allows an immediate addition?
                    # No, that's too greedy.
                    # Let's accept if the new state is "better" in terms of potential.
                    # Proxy: Number of valid non-edges.
                    
                    # This is expensive to calculate.
                    # Let's just accept the swap if it's valid, and hope it helps?
                    # No, that's random walk.
                    
                    # Better: Accept swap if it allows adding at least one edge immediately.
                    # Remove (a,b), add (c,d).
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    update_A2()
                    
                    # Now check if we can add any other edge.
                    can_add = False
                    for u in range(n):
                        a_u = adj[u]
                        non_neighbors = all_mask & ~a_u & ~(1 << u)
                        v_mask = non_neighbors
                        while v_mask:
                            lsb_v = v_mask & -v_mask
                            v = lsb_v.bit_length() - 1
                            v_mask ^= lsb_v
                            if u < v:
                                if not check_c6_add(u, v):
                                    can_add = True
                                    break
                        if can_add:
                            break
                            
                    if can_add:
                        # Keep this swap
                        c_addable = True
                        improved = True
                        swapped = True
                        break
                    else:
                        # Revert to pre-swap state (after removing a,b but before adding c,d)
                        # Actually, we modified adj. We need to revert to adj_backup
                        adj = adj_backup
                        A2 = A2_backup
                        break # Try next (a,b)
                        
            if swapped:
                break
                
        if not swapped and not improved:
            # Stuck. Try a random restart with a slightly different initial graph?
            # Or just break.
            # Let's try a few random restarts if time permits.
            if time.time() < deadline - 10.0:
                # Restart with a new random 5-regular graph
                for _ in range(10):
                    if time.time() > deadline - 10.0:
                        break
                    for i in range(n):
                        adj[i] = 0
                    stubs = []
                    for i in range(n):
                        for _ in range(5):
                            stubs.append(i)
                    random.shuffle(stubs)
                    valid = True
                    for i in range(0, len(stubs), 2):
                        u = stubs[i]
                        v = stubs[i+1]
                        if u == v or (adj[u] >> v) & 1:
                            valid = False
                            break
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    if valid:
                        update_A2()
                        current_edges = sum(bin(a).count("1") for a in adj) // 2
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                        # Run greedy again
                        # (This restarts the while loop effectively)
                        improved = True
                        break

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        a = best_adj[u]
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lsb
            
    return edges
