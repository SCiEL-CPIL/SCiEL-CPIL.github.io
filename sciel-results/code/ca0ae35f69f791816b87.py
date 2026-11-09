import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 88.0  # Safety margin for 90s limit
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    # Note: Bit k in compat[i] corresponds to node k+1.
    # So node 1 is bit 0, node 512 is bit 511.
    compat = [0] * (N + 1)
    
    # Precompute gcd/lcm checks
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # j is compatible with i
                compat[i] |= (1 << (j - 1))
                # i is compatible with j
                compat[j] |= (1 << (i - 1))
    
    # 2. Tomita's Algorithm with Bitmasks and Dynamic Coloring
    
    best_clique = []
    
    # Greedy Lower Bound
    def greedy_clique(order=None):
        if order is None:
            # Sort by degree descending
            degrees = [(i, bin(compat[i]).count('1')) for i in range(1, N+1)]
            degrees.sort(key=lambda x: -x[1])
            order = [d[0] for d in degrees]
        
        clique = []
        for node in order:
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

    # Try a couple of different orderings for the initial bound
    init_clique1 = greedy_clique()
    if init_clique1:
        best_clique = init_clique1
        
    # Try ordering by value ascending? (Small numbers are more compatible)
    init_clique2 = greedy_clique(order=list(range(1, N+1)))
    if len(init_clique2) > len(best_clique):
        best_clique = init_clique2

    sys.setrecursionlimit(10000)
    
    def expand(P_mask, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        # Simple bound
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
            
        # Calculate dynamic degree for each vertex in P
        # degree(v) = popcount(compat[v] & P_mask)
        P_with_deg = []
        for v in P:
            deg = bin(compat[v] & P_mask).count('1')
            P_with_deg.append((deg, v))
            
        # Sort by degree descending (standard Tomita improvement)
        P_with_deg.sort(key=lambda x: -x[0])
        P = [v for _, v in P_with_deg]
        
        # Greedy Coloring
        # We assign colors to vertices in P such that no two adjacent vertices have the same color.
        # The number of colors used is an upper bound on the clique size in P.
        color_list = []  # List of lists, color_list[k] contains vertices with color k
        vertex_color = {}
        
        for v in P:
            assigned = False
            # Try to assign the lowest possible color
            for k in range(len(color_list)):
                conflict = False
                # Check if v is compatible with any vertex in color_list[k]
                # If it is compatible, they can't have the same color (in the complement graph logic? No.)
                # Wait, Tomita's coloring is on the graph G[P].
                # If u and v are adjacent in G, they must have different colors.
                # So, if v is compatible (adjacent) with any u in color_list[k], conflict.
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
        # Iterate colors in reverse order (highest color first)
        # This is a heuristic to find large cliques faster.
        
        for k in range(max_color - 1, -1, -1):
            # Vertices in this color class
            # We need to process them in the order they were added or sorted?
            # Standard: process vertices in the order they appear in the sorted P list.
            # But we need to remove them from P_mask as we go.
            
            # To be efficient, we iterate through the vertices in color_list[k].
            # However, we must ensure we are iterating in the order consistent with the sorting.
            # Actually, the specific order within a color class doesn't change the bound,
            # but the "Exclude" step removes vertices from P_mask.
            
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
