import random
import time

def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_mask = 0
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110
    
    while time.time() < deadline:
        random.shuffle(edges)
        adj = [0] * n  # bitmask of neighbors
        current_count = 0
        
        for u, v in edges:
            if not creates_q3_mask(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_count += 1
                # Store mask? We need to reconstruct edges later.
                # Or just keep the list of added edges.
                
        # Reconstruct edges from adj?
        # Or keep a list.
        # Keeping a list is easier for return.
        
        # Let's keep a list of added edges in the loop
        # But I removed it in the sketch. Let's add it back.
        
    # Need to return the best list of edges.
    
    # Revised loop to track best list
    best_edges_list = []
    
    while time.time() < deadline:
        random.shuffle(edges)
        adj = [0] * n
        current_edges = []
        
        for u, v in edges:
            if not creates_q3_mask(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append([u, v])
                
        if len(current_edges) > len(best_edges_list):
            best_edges_list = current_edges
            
    return best_edges_list

def creates_q3_mask(u, v, adj):
    # adj is list of int bitmasks
    N_u = adj[u] & ~(1 << v)
    N_v = adj[v] & ~(1 << u)
    
    # Get list of neighbors
    # Helper to get set bits
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits
        
    N_u_list = get_bits(N_u)
    N_v_list = get_bits(N_v)
    
    if len(N_u_list) < 2 or len(N_v_list) < 2:
        return False
        
    for i in range(len(N_u_list)):
        x1 = N_u_list[i]
        for j in range(i + 1, len(N_u_list)):
            x2 = N_u_list[j]
            
            common_x = adj[x1] & adj[x2]
            if common_x == 0:
                continue
            
            # Iterate z2 in common_x
            z2_mask = common_x
            while z2_mask:
                lsb = z2_mask & -z2_mask
                z2 = lsb.bit_length() - 1
                z2_mask ^= lsb
                
                # Check y1, y2
                for k in range(len(N_v_list)):
                    y1 = N_v_list[k]
                    if not (adj[x1] & (1 << y1)):
                        continue
                    for l in range(k + 1, len(N_v_list)):
                        y2 = N_v_list[l]
                        if not (adj[x2] & (1 << y2)):
                            continue
                        
                        common_y = adj[y1] & adj[y2]
                        if common_y == 0:
                            continue
                        
                        # Check if any z1 in common_y is connected to z2
                        # z1 != z2
                        z1_mask = common_y & ~(1 << z2)
                        while z1_mask:
                            lsb1 = z1_mask & -z1_mask
                            z1 = lsb1.bit_length() - 1
                            z1_mask ^= lsb1
                            
                            if adj[z1] & (1 << z2):
                                return True
                                
    return False
