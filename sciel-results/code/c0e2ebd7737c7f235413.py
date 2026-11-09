import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps_list.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps_list, dtype=np.int32)
    M = len(aps_arr)
    
    # Precompute for each position p, the list of AP indices it belongs to
    # And for each such AP, the position index (0,1,2,3) within that AP
    # This allows fast delta calculation.
    pos_to_aps = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps_arr):
        for pos_idx in range(4):
            p = ap[pos_idx]
            pos_to_aps[p].append((ap_idx, pos_idx))
            
    # Convert to numpy arrays for faster access in vectorized ops if needed, 
    # but lists of tuples are often faster for small lists in Python loops.
    # However, for SA, we need to flip one bit and check delta.
    # Delta calculation for flipping bit p:
    # For each AP containing p, check if it was mono and becomes non-mono, or vice versa.
    
    def count_mono(c):
        # c is array of 0/1
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))

    def thue_morse(i):
        return bin(i).count('1') % 2
        
    # Generate multiple diverse initial colorings
    # 1. Thue-Morse
    # 2. Random
    # 3. Random with high/low density
    # 4. Chaotic (e.g., based on prime positions)
    
    initial_colors_list = []
    
    # Thue-Morse
    initial_colors_list.append(np.array([thue_morse(i) for i in range(n)], dtype=np.int8))
    
    # Random
    initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # Random biased
    initial_colors_list.append((np.random.rand(n) < 0.3).astype(np.int8))
    initial_colors_list.append((np.random.rand(n) < 0.7).astype(np.int8))
    
    # Structured: Alternating blocks
    initial_colors_list.append(np.array([i % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8))
    
    # Prime-based
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0: return False
        return True
    initial_colors_list.append(np.array([1 if is_prime(i+1) else 0 for i in range(n)], dtype=np.int8))
    
    # More randoms
    for _ in range(4):
        initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))
    
    best_score_global = float('inf')
    best_colors_global = None
    lock = threading.Lock()
    
    def run_sa(initial_colors, seed_idx):
        nonlocal best_score_global, best_colors_global
        np.random.seed(42 + seed_idx) # Deterministic per thread
        colors = initial_colors.copy()
        current_score = count_mono(colors)
        best_score_local = current_score
        best_colors_local = colors.copy()
        
        # Simulated Annealing parameters
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.9995
        steps = 20000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random position to flip
            p = np.random.randint(n)
            
            # Calculate delta
            delta = 0
            for ap_idx, pos_idx in pos_to_aps[p]:
                ap = aps_arr[ap_idx]
                v0 = colors[ap[0]]
                v1 = colors[ap[1]]
                v2 = colors[ap[2]]
                v3 = colors[ap[3]]
                
                was_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
                
                # Simulate flip at p
                # We know p is at pos_idx in this AP
                if pos_idx == 0:
                    new_v0 = 1 - v0
                    new_v1 = v1
                    new_v2 = v2
                    new_v3 = v3
                elif pos_idx == 1:
                    new_v0 = v0
                    new_v1 = 1 - v1
                    new_v2 = v2
                    new_v3 = v3
                elif pos_idx == 2:
                    new_v0 = v0
                    new_v1 = v1
                    new_v2 = 1 - v2
                    new_v3 = v3
                else:
                    new_v0 = v0
                    new_v1 = v1
                    new_v2 = v2
                    new_v3 = 1 - v3
                    
                is_mono = (new_v0 == new_v1) and (new_v1 == new_v2) and (new_v2 == new_v3)
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            # Acceptance criterion
            if delta < 0:
                colors[p] = 1 - colors[p]
                current_score += delta
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_colors_local = colors.copy()
            else:
                # Metropolis criterion
                if np.random.random() < np.exp(-delta / T_start):
                    colors[p] = 1 - colors[p]
                    current_score += delta
            
            T_start *= cooling_rate
            if T_start < T_end:
                T_start = T_end
        
        # Update global best
        with lock:
            if best_score_local < best_score_global:
                best_score_global = best_score_local
                best_colors_global = best_colors_local.copy()

    # Run SA in parallel threads
    # Number of threads: use all available CPUs, but limit to number of initializations
    n_threads = min(len(initial_colors_list), 8) # Cap threads to avoid overhead
    
    # Re-seed numpy for main thread consistency if needed, but threads use their own seeds
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i, init_c in enumerate(initial_colors_list):
            futures.append(executor.submit(run_sa, init_c, i))
            
        for future in as_completed(futures):
            future.result() # Raise exception if any
    
    if best_colors_global is None:
        # Fallback
        return [int(c) for c in initial_colors_list[0]]
        
    return [int(c) for c in best_colors_global]
