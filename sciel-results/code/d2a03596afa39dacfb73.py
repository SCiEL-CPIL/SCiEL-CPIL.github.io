import math
import time
import random

def construct():
    N = 256
    deadline = time.time() + 110
    
    # 1. Precompute 4-APs
    # APs are (a, a+d, a+2d, a+3d) with 0-based indices
    # a from 0 to N-1
    # d from 1 to ... such that a+3d < N
    aps_at = [[] for _ in range(N)]
    
    # We need to iterate over all valid (a, d)
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = [a, a+d, a+2*d, a+3*d]
            # Add to adjacency lists
            # For each index in idxs, store the other 3
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)
                
    # 2. Initialization
    colors = [random.randint(0, 1) for _ in range(N)]
    
    # 3. Initial Cost Calculation
    # We can calculate this by iterating all APs
    # Or summing contributions?
    # Easier: iterate all APs generated above? 
    # We didn't store the list of all APs, only adjacency.
    # Let's store all APs in a list too for initial count and verification.
    all_aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            all_aps.append(idxs)
            
    def count_mono(cols):
        cnt = 0
        for ap in all_aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt
        
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors[:]
    
    # 4. Simulated Annealing
    T = 1.0
    cooling = 0.99995
    min_T = 0.01
    
    while time.time() < deadline:
        # Pick random index
        i = random.randint(0, N - 1)
        
        # Calculate delta
        delta = 0
        for others in aps_at[i]:
            j, k, l = others
            c_i = colors[i]
            c_j = colors[j]
            c_k = colors[k]
            c_l = colors[l]
            
            # Before
            if c_i == c_j == c_k == c_l:
                delta -= 1 # It was mono, now it might not be. Wait.
                # If it was mono, and we flip i, it is NO LONGER mono (since c_i changes).
                # So cost decreases by 1.
            # After
            if (1 - c_i) == c_j == c_k == c_l:
                delta += 1 # It becomes mono. Cost increases by 1.
                
        # Note: If it was mono, c_i == c_j... so (1-c_i) != c_j, so "After" is false.
        # If it was NOT mono, "Before" is false.
        # So delta is either -1, 0, or 1 for each AP?
        # Actually, for a specific AP containing i:
        # If it was mono: Before=1, After=0 => delta=-1
        # If it was not mono:
        #   If flipping i makes it mono: Before=0, After=1 => delta=1
        #   Else: Before=0, After=0 => delta=0
        
        # So logic holds.
        
        if delta < 0:
            # Always accept
            colors[i] = 1 - colors[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
        else:
            # Accept with probability exp(-delta / T)
            if delta == 0:
                # Indifferent, accept to explore
                colors[i] = 1 - colors[i]
                # current_cost unchanged
            else:
                r = random.random()
                if r < math.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
                        
        T *= cooling
        if T < min_T:
            T = min_T
            # Optional: restart or just continue
            # With N=256, min_T is fine.
            
    return best_colors
