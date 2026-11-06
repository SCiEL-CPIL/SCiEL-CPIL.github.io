"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    n = 256
    
    # Fast discrepancy calculation using pure Python for low overhead
    def calc_disc(a):
        max_abs = 0
        for d in range(1, n + 1):
            for r in range(d):
                cur_pos = 0
                cur_neg = 0
                max_pos = 0
                max_neg = 0
                # Iterate through the progression
                for i in range(r, n, d):
                    v = a[i]
                    cp = cur_pos + v
                    if cp > 0:
                        cur_pos = cp
                    else:
                        cur_pos = 0
                    if cur_pos > max_pos:
                        max_pos = cur_pos
                        
                    cn = cur_neg + v
                    if cn < 0:
                        cur_neg = cn
                    else:
                        cur_neg = 0
                    if cur_neg < max_neg:
                        max_neg = cur_neg
                
                if max_pos > max_abs:
                    max_abs = max_pos
                neg_max = -max_neg
                if neg_max > max_abs:
                    max_abs = neg_max
        return max_abs

    # Initialize with Thue-Morse
    signs = []
    for i in range(n):
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    
    current_signs = signs[:]
    current_disc = calc_disc(current_signs)
    best_signs = current_signs[:]
    best_disc = current_disc
    
    # Simulated Annealing
    T = 10.0
    T_end = 0.001
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 55.0
    
    while time.time() - start_time < time_limit:
        # Flip a random bit
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_disc = calc_disc(current_signs)
        
        delta = new_disc - current_disc
        
        if delta < 0:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current_signs[:]
        else:
            # Metropolis criterion
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_disc = new_disc
                else:
                    # Revert
                    current_signs[idx] = old_val
        
        # Cool down
        T *= 0.9999 
        if T < T_end:
            T = T_end
            
    return best_signs

# EVOLVE-BLOCK-END
