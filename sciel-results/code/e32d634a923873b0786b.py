import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 40
    best_edges = []
    best_count = 0
    
    # adj_bits[i] is an integer with bit j set if (i,j) is an edge
    # adj_bits[i] bit i is always 0
    
    def creates_c6(adj_bits, u, v):
        # Check if adding (u, v) creates a C6
        # Need path u-a-b-c-v of length 4
        # a in N(u), c in N(v)
        # b in N(a) & N(c)
        # b != u, v, a, c
        # a != v, c != u, a != c
        
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        # Iterate a in N(u)
        a = nu & ~nu.bit_length() # No, iterate bits
        # Simple bit iteration
        a_mask = nu
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            if a == v:
                continue
                
            c_mask = nv
            while c_mask:
                lsb_c = c_mask & -c_mask
                c = lsb_c.bit_length() - 1
                c_mask ^= lsb_c
                
                if c == u or c == a:
                    continue
                
                # Common neighbors of a and c
                common = adj_bits[a] & adj_bits[c]
                # Exclude u, v, a, c
                mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                if common & ~mask:
                    return True
        return False

    def find_c6_edge(adj_bits):
        # Find an edge (u, v) that is in a C6
        for u in range(n):
            v_mask = adj_bits[u]
            # Only check v > u to avoid duplicates? 
            # But we just need ONE edge.
            # We can iterate v > u
            v_mask &= ~((1 << (u + 1)) - 1) # Clear bits 0..u
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask ^= lsb_v
                
                # Check if (u, v) is in a C6
                # Path u-a-b-c-v
                nu = adj_bits[u]
                nv = adj_bits[v]
                
                a_mask = nu
                while a_mask:
                    lsb_a = a_mask & -a_mask
                    a = lsb_a.bit_length() - 1
                    a_mask ^= lsb_a
                    if a == v:
                        continue
                    
                    c_mask = nv
                    while c_mask:
                        lsb_c = c_mask & -c_mask
                        c = lsb_c.bit_length() - 1
                        c_mask ^= lsb_c
                        
                        if c == u or c == a:
                            continue
                        
                        common = adj_bits[a] & adj_bits[c]
                        mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                        if common & ~mask:
                            return (u, v)
        return None

    for restart in range(100):
        if time.time() > deadline:
            break
        
        adj_bits = [0] * n
        
        # Random graph p=0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        # Clean
        while True:
            if time.time() > deadline:
                break
            c6_edge = find_c6_edge(adj_bits)
            if c6_edge is None:
                break
            u, v = c6_edge
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
        # Add edges greedily
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_bits[u] & (1 << v)):
                        if not creates_c6(adj_bits, u, v):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            improved = True
        
        # Count edges
        count = sum(bin(adj_bits[i]).count("1") for i in range(n)) // 2
        if count > best_count:
            best_count = count
            # Store copy
            best_adj_bits = adj_bits[:]
        
        # Swaps
        # Build list of edges and non-edges for random selection
        # This is O(N^2) per iteration, which is fine for N=40
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Pick random edge
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1: continue
            if u1 > v1: u1, v1 = v1, u1
            if not (adj_bits[u1] & (1 << v1)):
                continue # Not an edge
            
            # Pick random non-edge
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if (adj_bits[u2] & (1 << v2)):
                continue # Is an edge
            
            # Try swap
            adj_bits[u1] &= ~(1 << v1)
            adj_bits[v1] &= ~(1 << u1)
            
            if not creates_c6(adj_bits, u2, v2):
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                # Update best if improved? No, edge count is same.
                # But structure might be better for future swaps.
            else:
                # Revert
                adj_bits[u1] |= (1 << v1)
                adj_bits[v1] |= (1 << u1)
                
        # Check if this specific configuration is better (shouldn't be, same count)
        # But maybe we can do more swaps?
        # The count is the same, so best_count doesn't change.
        # But we keep the state for next restart? No, we restart.
        # Actually, we should keep the best_adj_bits.
        # Since count is same, we don't update best_count.
        # But if we find a higher count in a later restart, we update.

    # Convert best_adj_bits to list of edges
    if best_count > 0:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj_bits[i] & (1 << j):
                    edges.append([i, j])
        return edges
    else:
        return []

