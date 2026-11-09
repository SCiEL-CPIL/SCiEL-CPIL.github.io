import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    compat = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
    
    # 2. Tomita's Algorithm with Bitmasks
    best_clique = []
    
    # Greedy Lower Bound
    def greedy_clique():
        degrees = [(i, bin(compat[i]).count('1')) for i in range(1, N+1)]
        degrees.sort(key=lambda x: -x[1])
        
        clique = []
        for node, deg in degrees:
            if time.time() > DEADLINE:
                break
            ok = True
            for c in clique:
                if not (compat[node] & (1 << (c - 1))):
                    ok = False
                    break
            if ok:
                clique.append(node)
        return clique

    init_clique = greedy_clique()
    if init_clique:
        best_clique = init_clique
    
    sys.setrecursionlimit(10000)
    
    def expand(P_mask, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        if depth + bin(P_mask).count('1') <= len(best_clique):
            return
            
        # Extract vertices from P_mask
        P = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            P.append(idx + 1)
            temp ^= lsb
            
        # Greedy Coloring
        # Sort by degree in P? 
        # For speed, sort by static degree descending.
        P.sort(key=lambda v: -bin(compat[v]).count('1'))
        
        color_list = []
        vertex_color = {}
        
        for v in P:
            assigned = False
            for k in range(len(color_list)):
                conflict = False
                # Check if v is compatible with any in color_list[k]
                # If yes, conflict.
                # We can use bitmask for color_list[k]?
                # Let's keep list for now, but check bits.
                for u in color_list[k]:
                    if compat[v] & (1 << (u - 1)):
                        conflict = True
                        break
                if not conflict:
                    color_list[k].append(v)
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_list.append([v])
                vertex_color[v] = len(color_list) - 1
        
        max_color = len(color_list)
        
        if depth + max_color <= len(best_clique):
            return
        
        # Branching
        # Iterate colors in reverse
        P_set = set(P) # For quick removal? Or just use bitmask logic.
        
        for k in range(max_color - 1, -1, -1):
            for v in color_list[k]:
                # Check if v is still in P_mask
                if not (P_mask & (1 << (v - 1))):
                    continue
                
                # Include v
                C.append(v)
                
                # New P: neighbors of v in P_mask \ {v}
                new_P_mask = P_mask & compat[v] & ~(1 << (v - 1))
                
                expand(new_P_mask, C, depth + 1)
                
                C.pop()
                
                # Exclude v
                P_mask &= ~(1 << (v - 1))
                
                if time.time() > DEADLINE:
                    return

    initial_P_mask = (1 << N) - 1
    # Note: Bit 0 is node 1. Bit N-1 is node N.
    # We want nodes 1..N.
    # (1 << N) - 1 has bits 0..N-1 set.
    # This corresponds to nodes 1..N.
    
    expand(initial_P_mask, [], 0)
    
    if not best_clique:
        return [1]
        
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
