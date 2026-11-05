import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s for safety
    
    # Fallback: Paley(29)
    n_paley = 29
    residues = set((x * x) % n_paley for x in range(1, n_paley))
    colors_paley = []
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            diff = (j - i) % n_paley
            colors_paley.append(0 if diff in residues else 1)
    
    best_n = n_paley
    best_colors = colors_paley
    
    # Try to find n=30
    n_target = 30
    
    # Helper to get index
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * n_target - (i * (i + 1)) // 2 + j - i - 1
    
    # Helper to count K2,2,2 violations
    # Returns number of "bad" pairs (v1, v2) for both colors
    def count_violations(colors):
        N0 = [0] * n_target
        N1 = [0] * n_target
        
        for i in range(n_target):
            for j in range(i + 1, n_target):
                c = colors[get_idx(i, j)]
                if c == 0:
                    N0[i] |= (1 << j)
                    N0[j] |= (1 << i)
                else:
                    N1[i] |= (1 << j)
                    N1[j] |= (1 << i)
        
        cost = 0
        for c in [0, 1]:
            N = N0 if c == 0 else N1
            for v1 in range(n_target):
                for v2 in range(v1 + 1, n_target):
                    S = N[v1] & N[v2]
                    if S == 0:
                        continue
                    # Get vertices in S
                    # Optimization: if bit_count < 4, cannot have K2,2
                    if S.bit_count() < 4:
                        continue
                    
                    # Check for K2,2 in S
                    # Iterate pairs in S
                    # We can just iterate all pairs in range(n) and check if in S
                    # Or extract list
                    verts = []
                    mask = S
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        mask ^= lsb
                    
                    L = len(verts)
                    # Check all pairs a1, a2 in verts
                    for k in range(L):
                        a1 = verts[k]
                        for m in range(k + 1, L):
                            a2 = verts[m]
                            T = N[a1] & N[a2] & S
                            if T.bit_count() >= 2:
                                cost += 1
                                # Early exit? No, we want the count for SA
                                # But for validity, we could return 1
                                # For SA, we need the magnitude
                                # However, if cost is 0, we are valid.
                                # We can break if we just want to know if it's > 0?
                                # No, SA needs gradient.
                                # But if cost is large, we just need to know it's bad.
                                # Let's just count.
        return cost

    # Initialize random coloring for n=30
    colors = [random.randint(0, 1) for _ in range(n_target * (n_target - 1) // 2)]
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.99
    best_cost = count_violations(colors)
    if best_cost == 0:
        return {'n': n_target, 'colors': colors}
    
    current_cost = best_cost
    current_colors = colors[:]
    
    while time.time() < deadline:
        # Pick random edge to flip
        idx = random.randint(0, len(current_colors) - 1)
        
        # Flip
        current_colors[idx] = 1 - current_colors[idx]
        
        # Calculate new cost
        new_cost = count_violations(current_colors)
        
        # Acceptance criterion
        if new_cost < current_cost or random.random() < pow(2.718281828, (current_cost - new_cost) / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                if best_cost == 0:
                    return {'n': n_target, 'colors': current_colors}
        else:
            # Revert
            current_colors[idx] = 1 - current_colors[idx]
            
        T *= cooling
        if T < 1e-4:
            T = 10.0 # Restart if stuck? Or just keep going
            # Maybe randomize a few edges?
            for _ in range(10):
                r_idx = random.randint(0, len(current_colors) - 1)
                current_colors[r_idx] = random.randint(0, 1)
            current_cost = count_violations(current_colors)

    return {'n': best_n, 'colors': best_colors}
