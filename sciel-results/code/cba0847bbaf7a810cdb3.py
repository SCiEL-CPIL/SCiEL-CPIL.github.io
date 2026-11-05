import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to check C4-free
    def is_c4_free(adj, n):
        # adj[i] is a set of neighbors
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    if len(common) > 0:
                        return False
        return True

    # Helper to calculate independence number (approximate or exact for small n)
    # For n <= 43, we can use a simple backtracking or ILP if available, 
    # but standard library only. 
    # A simple greedy independent set gives a lower bound. 
    # We need an upper bound < 11. 
    # Actually, we need to ensure NO independent set of size 11.
    # This is equivalent to checking if the maximum independent set size is < 11.
    
    def max_independent_set(adj, n):
        # Simple branch and bound
        # This might be slow for n=40 if not optimized.
        # Let's use a heuristic first: Greedy.
        # If greedy finds < 11, it's not proof.
        # We need a more robust check.
        # For n=40, exact MIS is feasible with bitsets.
        
        # Bitset representation for speed
        # adj_bits[i] is a bitmask of neighbors of i
        
        # We'll implement a simple recursive search with pruning.
        # To speed it up, we can use the fact that we only care if MIS >= 11.
        
        def can_find_independent_set(k, available_mask):
            if k == 0:
                return True
            if available_mask == 0:
                return False
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in available subgraph?
            # Or just pick the first available.
            # Let's pick the first available vertex v.
            v_bit = 1
            while not (available_mask & v_bit):
                v_bit <<= 1
            
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            # New available mask: available_mask & ~adj_bits[v] & ~(v_bit)
            if can_find_independent_set(k - 1, available_mask & ~adj_bits[v] & ~v_bit):
                return True
            
            # Option 2: Exclude v
            if can_find_independent_set(k, available_mask & ~v_bit):
                return True
            
            return False

        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_bits[i] = mask
        
        full_mask = (1 << n) - 1
        return can_find_independent_set(11, full_mask)

    # Base Graph: Polarity Graph of PG(2,5)
    # q = 5
    # Points are (x, y, 1) and (x, 1, 0) and (1, 0, 0) ? 
    # Standard representation:
    # Points: (x, y) for x,y in F5 (25 points)
    #           (x) for x in F5 (5 points)
    #           (1) (1 point)
    # Total 31 points.
    
    # Polarity: (x, y) ~ (x', y') if x*x' + y*y' + 1*1 = 0 ? 
    # Or x*x' + y*y' + z*z' = 0 with z=1.
    # Let's use the standard construction for q=5.
    
    def build_pg2_polarity(q):
        # Field F_q
        # For q=5, it's just integers mod 5.
        # Points:
        # 0..24: (x, y, 1)
        # 25..29: (x, 1, 0)
        # 30: (1, 0, 0)
        
        n = q*q + q + 1
        adj = [set() for _ in range(n)]
        
        def point_coords(i):
            if i < q*q:
                return (i // q, i % q, 1)
            elif i < q*q + q:
                return ((i - q*q), 1, 0)
            else:
                return (1, 0, 0)
        
        # Polarity: P ~ Q if x1*x2 + y1*y2 + z1*z2 == 0 mod q
        for i in range(n):
            x1, y1, z1 = point_coords(i)
            for j in range(i + 1, n):
                x2, y2, z2 = point_coords(j)
                if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj

    adj = build_pg2_polarity(5)
    n = 31
    
    # Verify base graph
    if not is_c4_free(adj, n):
        # Fallback to triangles if PG construction fails
        adj = [set() for _ in range(30)]
        n = 30
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            adj[a].add(b); adj[b].add(a)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(c); adj[c].add(b)
    
    # Try to add vertices up to 43
    best_n = n
    best_adj = [set(s) for s in adj]
    
    for target_n in range(n + 1, 44):
        if time.time() > deadline - 10:
            break
            
        # Try to add vertex target_n - 1
        # Find valid neighbor sets
        # Incompatibility graph H: u ~_H w if N(u) & N(w) != empty
        H_adj = [set() for _ in range(n)]
        for u in range(n):
            for w in range(u + 1, n):
                if adj[u] & adj[w]:
                    H_adj[u].add(w)
                    H_adj[w].add(u)
        
        # We need an independent set S in H_adj.
        # To maximize chances of keeping alpha < 11, we want S to be "large" and "spread out".
        # Let's try a few random independent sets in H_adj.
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            # Greedy independent set in H_adj
            S = set()
            available = set(range(n))
            # Randomize order
            random.shuffle(list(available))
            for v in list(available):
                if v in available:
                    S.add(v)
                    available.difference_update(H_adj[v])
                    available.discard(v)
            
            # Try adding vertex with neighbors S
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for v in S:
                new_adj[v].add(n)
            
            new_n = n + 1
            if is_c4_free(new_adj, new_n):
                # Check independence
                # Use the MIS check
                if not max_independent_set(new_adj, new_n):
                    # alpha < 11
                    adj = new_adj
                    n = new_n
                    found = True
                    break
        
        if not found:
            # Try modifying existing graph? Or just stop.
            # For now, if we can't add, we stop.
            # But we might be able to add with a smaller S.
            # Let's try smaller S.
            pass
            
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
