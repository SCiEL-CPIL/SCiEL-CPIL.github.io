import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # 1. Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # 2. Precompute interaction matrix A
    # A[i, j] = number of APs containing both i and j (i != j)
    # A[i, i] = number of APs containing i
    A = np.zeros((n, n), dtype=np.float64)
    for ap in aps_arr:
        i0, i1, i2, i3 = ap
        A[i0, i0] += 1
        A[i1, i1] += 1
        A[i2, i2] += 1
        A[i3, i3] += 1
        
        A[i0, i1] += 1; A[i1, i0] += 1
        A[i0, i2] += 1; A[i2, i0] += 1
        A[i0, i3] += 1; A[i3, i0] += 1
        A[i1, i2] += 1; A[i2, i1] += 1
        A[i1, i3] += 1; A[i3, i1] += 1
        A[i2, i3] += 1; A[i3, i2] += 1
        
    # 3. Helper functions
    def count_mono(c):
        # c is array of 0/1
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))

    def thue_morse(i):
        return bin(i).count('1') % 2
        
    def fast_delta(c, p):
        # c is 0/1 array, p is index to flip
        # Delta = (sum_{j != p} A[p, j] * (1 - 2*c[j]) * (2*c[p] - 1)) + 4*A[p, p]*(2*c[p] - 1)
        # Let s_j = 1 if c[j]=0, -1 if c[j]=1. s_p = 2*c[p] - 1.
        # (1 - 2*c[j]) = s_j. (2*c[p] - 1) = s_p.
        # Delta = s_p * (sum_{j != p} A[p, j] * s_j + 4*A[p, p]*s_p)
        # sum_{j != p} A[p, j] * s_j = sum_{all j} A[p, j] * s_j - A[p, p]*s_p
        # Delta = s_p * (A[p, :] @ s - A[p, p]*s_p + 4*A[p, p]*s_p)
        # Delta = s_p * (A[p, :] @ s + 3*A[p, p]*s_p)
        
        # Convert c to s (1 for 0, -1 for 1)
        # s = 1 - 2*c
        # But we can just compute the dot product part.
        # Let's precompute s vector for the whole array for efficiency in the loop?
        # No, we only flip one bit. We can maintain s.
        
        # For this function, let's assume we pass s (sign vector) and c (color vector) or just compute from c.
        # To be safe and simple in the function call:
        s = 1 - 2 * c
        s_p = s[p]
        
        # dot = A[p, :] @ s
        # Note: A[p, p] is included in dot.
        # delta = s_p * (dot - A[p, p]*s_p + 4*A[p, p]*s_p)
        # delta = s_p * (dot + 3*A[p, p]*s_p)
        
        dot = np.dot(A[p, :], s)
        delta = s_p * (dot + 3 * A[p, p] * s_p)
        return delta

    # 4. Simulated Annealing
    # We will run multiple restarts or a single long SA with restarts if stuck.
    # Given time constraints, a single robust SA with good cooling is often best.
    # Or, since we have fast delta, we can do many steps.
    
    # Initialize with Thue-Morse
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    s = 1 - 2 * colors  # sign vector
    
    current_score = count_mono(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9999
    
    # We need to update s when we flip.
    # s[p] = -s[p]
    # c[p] = 1 - c[p]
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random position
        p = np.random.randint(0, n)
        
        # Calculate delta
        # We need s vector.
        # dot = A[p, :] @ s
        # This is O(n) = O(256). Very fast.
        
        delta = fast_delta(colors, p)
        
        if delta < 0:
            # Accept
            colors[p] = 1 - colors[p]
            s[p] = -s[p]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            # Accept with probability exp(-delta / T)
            # delta > 0
            prob = np.exp(-delta / T)
            if np.random.random() < prob:
                colors[p] = 1 - colors[p]
                s[p] = -s[p]
                current_score += delta
                
        T *= cooling
        steps += 1
        
        # Optional: If we stagnate, restart?
        # For now, let it cool down. If it hits T_min, we stop.
        # With cooling 0.9999, it takes ~10000 steps to go from 10 to 0.01.
        # 10000 steps * 256 ops = 2.5M ops. Very fast.
        # We can likely run many restarts.
        
        # Let's implement a simple restart strategy:
        # If T < T_min, restart with a perturbed best_colors.
        if T < T_min:
            # Perturb best_colors
            colors = best_colors.copy()
            s = 1 - 2 * colors
            current_score = best_score
            
            # Flip 5% random bits
            num_flips = n // 20
            flip_idx = np.random.choice(n, num_flips, replace=False)
            for idx in flip_idx:
                colors[idx] = 1 - colors[idx]
                s[idx] = -s[idx]
            
            current_score = count_mono(colors)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                
            T = 10.0  # Reset temperature
            
    return [int(c) for c in best_colors]
