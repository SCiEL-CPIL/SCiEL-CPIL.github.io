import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def has_k4(adj, n):
        # adj is list of int bitsets
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    common = ai & adj[j]
                    # Check if common has an edge
                    # Iterate bits in common
                    m = common
                    while m:
                        lsb = m & -m
                        k = lsb.bit_length() - 1
                        m ^= lsb
                        # Check if k is connected to any other in common?
                        # No, we just need if (i,j,k,l) is K4.
                        # i-j, i-k, j-k are edges.
                        # We need l in common such that i-l, j-l, k-l are edges.
                        # l must be in adj[i] & adj[j] & adj[k].
                        # Since k is in common, adj[i] & adj[j] contains k.
                        # So we just check if (adj[i] & adj[j] & adj[k]) has any bit?
                        # Yes, if there is any l, it forms a K4.
                        if adj[i] & adj[j] & adj[k]:
                            return True
        return False

    def has_k4_in_subgraph(adj, mask):
        # Check if induced subgraph on mask has K4
        # Get list of vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Map to local indices? Or just use global indices and adj.
        # Use global indices.
        # Iterate triples?
        # Or use the same logic as has_k4 but restricted to verts.
        # Since n is small, we can just iterate combinations of 4 from verts.
        # len(verts) <= 35.
        # 35 choose 4 is 52k.
        # This is called inside has_k6, which iterates pairs.
        # So it might be slow if called too often.
        
        # Better: Use the bitset logic restricted to mask.
        # For i in verts:
        #   ai = adj[i] & mask
        #   for j in verts where j > i:
        #      if ai & (1<<j):
        #         common = ai & adj[j] & mask
        #         # check if common has edge within mask
        #         m = common
        #         while m:
        #            lsb = m & -m
        #            k = lsb.bit_length() - 1
        #            m ^= lsb
        #            if (adj[i] & adj[j] & adj[k] & mask):
        #                return True
        # return False
        pass

    def has_k6(adj, n):
        # Check for K6 in adj
        # Exists i, j such that common neighborhood contains K4.
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    common = ai & adj[j]
                    # Check if common contains K4
                    # We need to check if the subgraph induced by 'common' has a K4.
                    # Use has_k4_in_subgraph(adj, common)
                    if has_k4_in_subgraph(adj, common):
                        return True
        return False

    # Implement has_k4_in_subgraph efficiently
    def has_k4_in_subgraph(adj, mask):
        # Iterate i in mask
        m1 = mask
        while m1:
            lsb1 = m1 & -m1
            i = lsb1.bit_length() - 1
            m1 ^= lsb1
            
            ai = adj[i] & mask
            m2 = mask & ~((1 << (i + 1)) - 1) # j > i? 
            # Actually, we can just iterate all j in mask, j != i.
            # But to avoid duplicates, j > i.
            # Let's just iterate all j in mask.
            m2 = mask
            while m2:
                lsb2 = m2 & -m2
                j = lsb2.bit_length() - 1
                m2 ^= lsb2
                if i == j: continue
                if ai & (1 << j):
                    common = ai & adj[j] & mask
                    # Check if common has an edge
                    m3 = common
                    while m3:
                        lsb3 = m3 & -m3
                        k = lsb3.bit_length() - 1
                        m3 ^= lsb3
                        if (adj[i] & adj[j] & adj[k] & mask):
                            return True
        return False

    # Now search
    for n in range(35, 0, -1):
        # Time limit per n?
        # Let's just run until deadline.
        
        # Random init
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        steps = 0
        while time.time() < deadline:
            # Check validity
            if not has_k4(adj, n) and not has_k6(adj, n):
                # Valid
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Repair
            # If K4 exists, remove edge.
            # If K6 in comp (i.e. I6 in G) exists, add edge.
            
            # Find a K4 to break
            # We can modify has_k4 to return the K4.
            # For now, just pick a random edge and flip it?
            # Better: pick a random edge in a K4.
            
            # Let's just do a random walk with bias.
            # Flip a random edge.
            i, j = random.sample(range(n), 2)
            if random.random() < 0.5:
                # Add
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                # Remove
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            
            steps += 1
            
            # If we are stuck, try more aggressive moves?
            # The random walk might be too slow.
            # Let's add a "repair" step.
            
            # Check if K4 exists
            # We need to find one.
            # Let's implement find_k4
            pass

    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
