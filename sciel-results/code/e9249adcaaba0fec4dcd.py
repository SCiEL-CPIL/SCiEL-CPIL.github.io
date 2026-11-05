import random
import time

def construct():
    n = 16
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_3_colorable(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            used = 0
            neighbors = adj[v]
            while neighbors:
                lsb = neighbors & -neighbors
                u = (lsb.bit_length() - 1)
                neighbors ^= lsb
                if colors[u] != -1:
                    used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return backtrack(0)

    # Construct Clebsch Graph
    even_parity_5bit = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            even_parity_5bit.append(i)
            
    val_map = even_parity_5bit
    clebsch_edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            u_val = val_map[i]
            v_val = val_map[j]
            xor_val = u_val ^ v_val
            if bin(xor_val).count('1') == 4:
                clebsch_edges.append([i, j])
                
    current_edges = clebsch_edges[:]
    current_adj = [0] * n
    for u, v in current_edges:
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
    best_edges = current_edges[:]
    best_count = len(current_edges)
    current_is_3c = is_3_colorable(current_edges)
    
    start_time = time.time()
    timeout = 110.0
    
    T = 5.0
    cooling = 0.995
    P = 10.0 # Penalty for 3-colorable
    
    rng = random.Random(42)
    
    while time.time() - start_time < timeout:
        if T < 0.01:
            break
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Add edge
            u, v = rng.choice(possible_edges)
            if current_adj[u] & (1 << v):
                continue
            if current_adj[u] & current_adj[v]:
                continue
            
            new_edges = current_edges + [[u, v]]
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_count = len(new_edges)
            new_is_3c = is_3_colorable(new_edges)
            
            curr_E = -len(current_edges) + (P if current_is_3c else 0)
            new_E = -new_count + (P if new_is_3c else 0)
            delta_E = new_E - curr_E
            
            if delta_E <= 0 or rng.random() < pow(2.71828, -delta_E / T):
                current_edges = new_edges
                current_adj = new_adj
                current_is_3c = new_is_3c
        else:
            # Remove edge
            if not current_edges:
                continue
            idx = rng.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            new_edges = current_edges[:]
            new_edges.pop(idx)
            
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            new_count = len(new_edges)
            new_is_3c = is_3_colorable(new_edges)
            
            curr_E = -len(current_edges) + (P if current_is_3c else 0)
            new_E = -new_count + (P if new_is_3c else 0)
            delta_E = new_E - curr_E
            
            if delta_E <= 0 or rng.random() < pow(2.71828, -delta_E / T):
                current_edges = new_edges
                current_adj = new_adj
                current_is_3c = new_is_3c
        
        if not current_is_3c and len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        T *= cooling

    return best_edges
