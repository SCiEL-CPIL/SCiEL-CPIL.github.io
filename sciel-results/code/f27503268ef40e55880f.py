import time
import random
import numpy as np

DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known 32-vertex solution as fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table, x = {}, 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
    
    best_n = 32
    best_colors = base_colors[:]

    # Simulated Annealing to find coloring for N vertices
    def find_coloring(N, max_time):
        if N <= 0:
            return None, None
        
        # C[i][j] is color of edge (i,j). -1 for diagonal.
        C = np.full((N, N), -1, dtype=np.int8)
        
        # Initialize randomly
        for i in range(N):
            for j in range(i+1, N):
                c = random.randint(0, 3)
                C[i, j] = c
                C[j, i] = c
        
        # Calculate initial energy
        def calc_energy(C):
            E = 0
            for c in range(4):
                # Count triangles in color c
                # For each i < j, check common neighbors k > j? No, just count all.
                # Triangle (i,j,k) with i<j<k.
                # It is monochromatic c if C[i,j]==c, C[j,k]==c, C[k,i]==c.
                for i in range(N):
                    for j in range(i+1, N):
                        if C[i, j] == c:
                            # Check k > j
                            # Common neighbors of i and j in color c, restricted to k > j
                            # This is O(N^3). For N=50, 125k * 4 = 500k ops. Fast.
                            for k in range(j+1, N):
                                if C[j, k] == c and C[k, i] == c:
                                    E += 1
            return E

        E = calc_energy(C)
        
        if E == 0:
            return C, E

        T = 1.0
        T_min = 0.001
        alpha = 0.9995 # cooling rate
        
        # Precompute edge list for faster access? 
        # Or just use indices.
        
        while time.time() < deadline - max_time:
            if E == 0:
                break
                
            # Find a monochromatic triangle to target
            # Randomly pick an edge, check if it's in a mono-triangle
            found_triangle = False
            for _ in range(100): # max attempts to find a bad edge
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                c = C[u, v]
                # Check if there is w such that (u,w) and (v,w) are color c
                # w != u, v
                # We can check all w
                bad_w = -1
                for w in range(N):
                    if w == u or w == v: continue
                    if C[u, w] == c and C[v, w] == c:
                        bad_w = w
                        break
                
                if bad_w != -1:
                    # We found triangle (u, v, bad_w) in color c
                    # Pick a random edge from the triangle to change
                    # Options: (u,v), (v,bad_w), (bad_w,u)
                    edges_in_tri = [(u, v), (v, bad_w), (bad_w, u)]
                    e1, e2 = random.choice(edges_in_tri)
                    if e1 > e2: e1, e2 = e2, e1
                    found_triangle = True
                    break
            
            if not found_triangle:
                # No bad edge found in 100 tries? Maybe E is very small or 0?
                # If E > 0 but we can't find a triangle, it's a bug in E calc or search.
                # Just pick random edge.
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v: continue
                if u > v: u, v = v, u
                e1, e2 = u, v
                
            # Try changing color of (e1, e2)
            c_old = C[e1, e2]
            
            best_delta = float('inf')
            best_new_c = c_old
            
            for c_new in range(4):
                if c_new == c_old: continue
                
                # Calculate Delta E
                # Delta E = (new triangles formed) - (old triangles broken)
                # Old triangles: count w such that C[e1,w]==c_old and C[e2,w]==c_old
                # New triangles: count w such that C[e1,w]==c_new and C[e2,w]==c_new
                
                # Exclude e1 and e2 from w
                # We can use numpy
                vec1 = C[e1]
                vec2 = C[e2]
                
                # Mask out e1 and e2
                # Since C is symmetric and diagonal is -1, 
                # C[e1, e1] = -1, so it won't match c_old (0..3).
                # So we don't need to mask diagonal.
                # But we need to make sure we don't count the edge (e1,e2) itself?
                # The triangle condition involves a third vertex w.
                # The edge (e1,e2) is not a triangle with itself.
                # So counting w such that C[e1,w]==c and C[e2,w]==c is correct.
                
                count_old = np.sum((vec1 == c_old) & (vec2 == c_old))
                count_new = np.sum((vec1 == c_new) & (vec2 == c_new))
                
                delta = count_new - count_old
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c_new
            
            # Acceptance
            if best_delta < 0:
                accept = True
            else:
                if T > T_min:
                    accept = random.random() < np.exp(-best_delta / T)
                else:
                    accept = False
            
            if accept:
                C[e1, e2] = best_new_c
                C[e2, e1] = best_new_c
                E += best_delta
                if E < 0: E = 0 # Should not happen if calc is correct
                
            T *= alpha
            
        return C, E

    # Try to find coloring for n=51 first, then 50
    for N in [51, 50]:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on N
        # If N=51, give it 30s. If N=50, give it 100s.
        time_budget = 30 if N == 51 else 100
        
        C, E = find_coloring(N, time_budget)
        
        if C is not None and E == 0:
            # Convert C to list of colors in lexicographic order
            colors = []
            for i in range(N):
                for j in range(i+1, N):
                    colors.append(int(C[i, j]))
            
            if N > best_n:
                best_n = N
                best_colors = colors
            
            # If we found 51, we can stop.
            if N == 51:
                break

    return {'n': best_n, 'colors': best_colors}
