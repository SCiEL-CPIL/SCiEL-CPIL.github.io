import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 40
    best_edges = []
    best_count = 0
    
    def has_path_length_5(u, v, adj_masks):
        # u, v are indices. adj_masks is list of ints.
        # S0 = {u}
        # S1 = N(u)
        # S2 = N(S1) \ (S0 U S1)
        # ...
        
        mask_u = 1 << u
        mask_v = 1 << v
        
        # S1
        s1 = adj_masks[u] & ~mask_u
        
        # S2
        # Union of neighbors of nodes in s1
        s2 = 0
        temp = s1
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s2 |= adj_masks[idx]
            temp -= lsb
        s2 &= ~mask_u & ~s1
        
        # S3
        s3 = 0
        temp = s2
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s3 |= adj_masks[idx]
            temp -= lsb
        s3 &= ~mask_u & ~s1 & ~s2
        
        # S4
        s4 = 0
        temp = s3
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s4 |= adj_masks[idx]
            temp -= lsb
        s4 &= ~mask_u & ~s1 & ~s2 & ~s3
        
        # S5
        s5 = 0
        temp = s4
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s5 |= adj_masks[idx]
            temp -= lsb
        s5 &= ~mask_u & ~s1 & ~s2 & ~s3 & ~s4
        
        return (s5 & mask_v) != 0

    # Greedy with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        adj_masks = [0] * n
        # Randomize order to avoid symmetry
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy add
        for u in order:
            for v in order:
                if u >= v:
                    continue
                if time.time() > deadline:
                    break
                
                # Check if edge (u,v) exists
                if (adj_masks[u] >> v) & 1:
                    continue
                
                # Check C6
                if not has_path_length_5(u, v, adj_masks):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        # Count edges
        count = 0
        for i in range(n):
            count += bin(adj_masks[i]).count('1')
        count //= 2
        
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_masks[u] >> v) & 1:
                        best_edges.append([u, v])
                        
    return best_edges
