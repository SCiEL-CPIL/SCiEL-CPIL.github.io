import sys

def construct():
    N = 256
    adj_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                
    all_vertices = (1 << (N + 1)) - 2
    
    best_size = 0
    best_set = 0
    
    sys.setrecursionlimit(5000)
    
    def dfs(P, current_size, current_set_bits):
        nonlocal best_size, best_set
        
        if P == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set_bits
            return
            
        if current_size + P.bit_count() <= best_size:
            return
            
        # Heuristic: Select v in P with maximum degree in P
        max_deg = -1
        best_v = -1
        temp_P = P
        # Iterate through set bits
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            deg = (P & adj_mask[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_P ^= lsb
            
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # Remove v and its neighbors from P
        new_P = P & ~(adj_mask[v] | v_bit)
        dfs(new_P, current_size + 1, current_set_bits | v_bit)
        
        # Branch 2: Exclude v
        new_P = P & ~v_bit
        dfs(new_P, current_size, current_set_bits)
        
    dfs(all_vertices, 0, 0)
    
    result = []
    for i in range(1, N + 1):
        if (best_set >> i) & 1:
            result.append(i)
            
    return result
