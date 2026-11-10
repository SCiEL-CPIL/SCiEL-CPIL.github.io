import time
import numpy as np
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate divisors
    divisors = []
    for i in range(3, int(np.sqrt(N)) + 1, 2):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    # Add 1? No, m > 1.
    # Check if N itself is a divisor (it is, but we iterate up to sqrt, so N is added as N//1 if 1 was checked? 
    # My loop starts at 3. 
    # N is 51975.
    # Divisors of N:
    # 1, 3, 5, 7, 9, 11, 15, 21, 25, 27, 33, 35, 45, 55, 63, 75, 77, 99, 105, 135, 165, 175, 189, 231, 275, 315, 385, 441, 495, 525, 567, 693, 787.5 (no), 
    # Wait, 51975 / 3 = 17325.
    # 51975 / 5 = 10395.
    # 51975 / 7 = 7425.
    # 51975 / 9 = 5775.
    # 51975 / 11 = 4725.
    # 51975 / 15 = 3465.
    # 51975 / 21 = 2475.
    # 51975 / 25 = 2079.
    # 51975 / 27 = 1925.
    # 51975 / 33 = 1575.
    # 51975 / 35 = 1485.
    # 51975 / 45 = 1155.
    # 51975 / 55 = 945.
    # 51975 / 63 = 825.
    # 51975 / 75 = 693.
    # 51975 / 77 = 675.
    # 51975 / 99 = 525.
    # 51975 / 105 = 495.
    # 51975 / 135 = 385.
    # 51975 / 165 = 315.
    # 51975 / 175 = 297.
    # 51975 / 189 = 275.
    # 51975 / 231 = 225.
    # 51975 / 275 = 189.
    # 51975 / 315 = 165.
    # 51975 / 385 = 135.
    # 51975 / 441 = 117.8 (No, 441 * 117 = 51597, 441 * 118 = 52038. 51975 / 441 = 117.85... Wait. 441 = 21^2 = 3^2 * 7^2. N has only 3^3 * 7^1. So 441 does NOT divide N.)
    # Let's re-calculate divisors properly.
    # N = 3^3 * 5^2 * 7 * 11
    # Divisors are of form 3^a * 5^b * 7^c * 11^d
    # a in 0..3, b in 0..2, c in 0..1, d in 0..1
    
    divisors = set()
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    val = (3**a) * (5**b) * (7**c) * (11**d)
                    if val > 1:
                        divisors.add(val)
    
    moduli = list(divisors)
    
    best_classes = []
    best_uncovered = N
    
    # Try a few different greedy orders
    orders = [
        sorted(moduli, reverse=True), # Decreasing
        sorted(moduli),               # Increasing
        sorted(moduli, key=lambda x: N//x), # Same as decreasing
    ]
    
    # Add a couple of random orders
    for _ in range(2):
        random.shuffle(moduli)
        orders.append(moduli.copy())
    
    for order in orders:
        if time.time() > deadline:
            break
            
        # Greedy Phase
        cover_counts = np.zeros(N, dtype=np.int8)
        uncovered = N
        classes = []
        current_rs = []
        
        for m in order:
            # Find best r
            # gains[r] = sum(cover_counts[r::m] == 0)
            # Using a loop for clarity, might optimize if slow
            gains = np.zeros(m, dtype=np.int32)
            for r in range(m):
                gains[r] = np.sum(cover_counts[r::m] == 0)
            
            best_r = np.argmax(gains)
            best_gain = gains[best_r]
            
            # Update
            cover_counts[best_r::m] += 1
            uncovered -= best_gain
            classes.append([m, best_r])
            current_rs.append(best_r)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c.copy() for c in classes]
            best_cover_counts = cover_counts.copy()
            best_current_rs = current_rs.copy()
            best_order = order.copy()
        
        # Local Search Phase
        # We only do local search if we have time
        if time.time() > deadline - 10.0:
            continue
            
        cur_uncovered = uncovered
        cur_cover_counts = cover_counts.copy()
        cur_rs = current_rs.copy()
        cur_classes = [c.copy() for c in classes]
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random modulus to modify
            k = random.randint(0, len(moduli) - 1)
            m = order[k]
            r_old = cur_rs[k]
            
            # Calculate losses from removing r_old
            # Indices in I_old
            I_old = range(r_old, N, m)
            losses = 0
            for i in I_old:
                if cur_cover_counts[i] == 1:
                    losses += 1
            
            # Temporarily remove
            for i in I_old:
                cur_cover_counts[i] -= 1
            
            # Find best r_new
            # gains[r] = sum(cur_cover_counts[r::m] == 0)
            best_gain = -1
            best_r_new = r_old
            # Optimization: If m is large, this loop is long.
            # But N/m is small.
            for r in range(m):
                gain = np.sum(cur_cover_counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r_new = r
            
            net_change = best_gain - losses
            
            if net_change > 0:
                # Apply
                I_new = range(best_r_new, N, m)
                for i in I_new:
                    cur_cover_counts[i] += 1
                cur_rs[k] = best_r_new
                cur_classes[k][1] = best_r_new
                cur_uncovered -= net_change
                no_improve_count = 0
            elif net_change == 0 and random.random() < 0.1:
                # Accept neutral move with small probability to escape local minima
                I_new = range(best_r_new, N, m)
                for i in I_new:
                    cur_cover_counts[i] += 1
                cur_rs[k] = best_r_new
                cur_classes[k][1] = best_r_new
                # cur_uncovered unchanged
                no_improve_count += 1
            else:
                # Revert
                for i in I_old:
                    cur_cover_counts[i] += 1
                no_improve_count += 1
            
            if cur_uncovered < best_uncovered:
                best_uncovered = cur_uncovered
                best_classes = [c.copy() for c in cur_classes]
                # best_cover_counts = cur_cover_counts.copy() # Not needed if we return classes

    return best_classes

# EVOLVE-BLOCK-END
