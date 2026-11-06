import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Pre-compute constraints
        # red_k4s: list of tuples of 4 triple indices
        # blue_k5s: list of tuples of 10 triple indices
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        red_k4s = []
        for quad in combinations(range(n), 4):
            red_k4s.append(tuple(triple_index[t] for t in combinations(quad, 3)))
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            blue_k5s.append(tuple(triple_index[t] for t in combinations(quint, 3)))
            
        num_red_k4s = len(red_k4s)
        num_blue_k5s = len(blue_k5s)
        
        # Initialize colors
        # colors[i] = 0 (red) or 1 (blue)
        # We want to minimize violations.
        # Initial random coloring
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Pre-compute indices for vectorized updates
        # For each triple i, which red_k4s does it belong to?
        # For each triple i, which blue_k5s does it belong to?
        triple_to_red_k4s = [[] for _ in range(num_triples)]
        triple_to_blue_k5s = [[] for _ in range(num_triples)]
        
        for k4_idx, k4 in enumerate(red_k4s):
            for t_idx in k4:
                triple_to_red_k4s[t_idx].append(k4_idx)
                
        for k5_idx, k5 in enumerate(blue_k5s):
            for t_idx in k5:
                triple_to_blue_k5s[t_idx].append(k5_idx)
                
        # Convert to numpy arrays for faster iteration
        triple_to_red_k4s_arr = [np.array(lst, dtype=np.int32) for lst in triple_to_red_k4s]
        triple_to_blue_k5s_arr = [np.array(lst, dtype=np.int32) for lst in triple_to_blue_k5s]
        
        # Initialize violation counts
        # red_violations[k4_idx] = number of red triples in this k4 (0-4). Violation if 4.
        # blue_violations[k5_idx] = number of blue triples in this k5 (0-10). Violation if 10.
        
        def compute_violations(colors):
            red_v = np.zeros(num_red_k4s, dtype=np.int32)
            blue_v = np.zeros(num_blue_k5s, dtype=np.int32)
            for k4 in red_k4s:
                count = 0
                for t_idx in k4:
                    if colors[t_idx] == 0:
                        count += 1
                red_v[red_k4s.index(k4)] = count # This is slow, let's do it properly in init
                
            # Better init:
            red_v = np.zeros(num_red_k4s, dtype=np.int32)
            for k4_idx, k4 in enumerate(red_k4s):
                count = 0
                for t_idx in k4:
                    if colors[t_idx] == 0:
                        count += 1
                red_v[k4_idx] = count
                
            blue_v = np.zeros(num_blue_k5s, dtype=np.int32)
            for k5_idx, k5 in enumerate(blue_k5s):
                count = 0
                for t_idx in k5:
                    if colors[t_idx] == 1:
                        count += 1
                blue_v[k5_idx] = count
                
            return red_v, blue_v

        red_v, blue_v = compute_violations(colors)
        
        def total_violations(red_v, blue_v):
            # Count how many k4s have 4 red triples
            # Count how many k5s have 10 blue triples
            return np.sum(red_v == 4) + np.sum(blue_v == 10)
            
        current_violations = total_violations(red_v, blue_v)
        
        # Simulated Annealing
        T = 1000.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        
        while T > min_T and time.time() < deadline - 5:
            # Pick a random triple to flip
            i = np.random.randint(0, num_triples)
            
            # Calculate delta violations
            # If we flip i from 0 to 1:
            # - For red_k4s containing i: red count decreases by 1.
            #   If count was 4, violation removed (-1).
            #   If count was 5 (impossible, max 4), no change.
            #   If count was 3, no change (still not 4).
            #   Actually, if count was 4, it becomes 3 (violation gone).
            #   If count was 5? No, max is 4.
            #   So delta_red = -1 if red_v[k4]==4 else 0.
            
            # - For blue_k5s containing i: blue count increases by 1.
            #   If count was 9, violation created (+1) (becomes 10).
            #   If count was 10, violation removed (-1) (becomes 9).
            #   So delta_blue = +1 if blue_v[k5]==9 else (-1 if blue_v[k5]==10 else 0).
            
            # If we flip i from 1 to 0:
            # - For red_k4s containing i: red count increases by 1.
            #   If count was 3, violation created (+1) (becomes 4).
            #   If count was 4, violation removed (-1) (becomes 3).
            
            # - For blue_k5s containing i: blue count decreases by 1.
            #   If count was 10, violation removed (-1) (becomes 9).
            #   If count was 11 (impossible), no change.
            
            delta = 0
            if colors[i] == 0:
                # Flipping 0 -> 1
                for k4_idx in triple_to_red_k4s_arr[i]:
                    if red_v[k4_idx] == 4:
                        delta -= 1
                for k5_idx in triple_to_blue_k5s_arr[i]:
                    if blue_v[k5_idx] == 9:
                        delta += 1
                    elif blue_v[k5_idx] == 10:
                        delta -= 1
            else:
                # Flipping 1 -> 0
                for k4_idx in triple_to_red_k4s_arr[i]:
                    if red_v[k4_idx] == 3:
                        delta += 1
                    elif red_v[k4_idx] == 4:
                        delta -= 1
                for k5_idx in triple_to_blue_k5s_arr[i]:
                    if blue_v[k5_idx] == 10:
                        delta -= 1
            
            # Acceptance probability
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Apply flip
                colors[i] = 1 - colors[i]
                
                # Update counts
                if colors[i] == 1:
                    # Was 0, now 1
                    for k4_idx in triple_to_red_k4s_arr[i]:
                        red_v[k4_idx] -= 1
                    for k5_idx in triple_to_blue_k5s_arr[i]:
                        blue_v[k5_idx] += 1
                else:
                    # Was 1, now 0
                    for k4_idx in triple_to_red_k4s_arr[i]:
                        red_v[k4_idx] += 1
                    for k5_idx in triple_to_blue_k5s_arr[i]:
                        blue_v[k5_idx] -= 1
                        
                current_violations += delta
                
                if current_violations == 0:
                    # Found a valid coloring!
                    return {'n': n, 'colors': colors.tolist()}
                    
            T *= cooling
            steps += 1
            
            if steps % 1000 == 0 and time.time() > deadline - 10:
                break
                
        # If we didn't find 0 violations, try restarting with different initializations
        # But we have limited time. Let's try a few more random restarts quickly.
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            colors = np.random.randint(0, 2, size=num_triples)
            red_v, blue_v = compute_violations(colors)
            current_violations = total_violations(red_v, blue_v)
            if current_violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            # Short SA run
            T = 500.0
            for _ in range(5000):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(0, num_triples)
                delta = 0
                if colors[i] == 0:
                    for k4_idx in triple_to_red_k4s_arr[i]:
                        if red_v[k4_idx] == 4: delta -= 1
                    for k5_idx in triple_to_blue_k5s_arr[i]:
                        if blue_v[k5_idx] == 9: delta += 1
                        elif blue_v[k5_idx] == 10: delta -= 1
                else:
                    for k4_idx in triple_to_red_k4s_arr[i]:
                        if red_v[k4_idx] == 3: delta += 1
                        elif red_v[k4_idx] == 4: delta -= 1
                    for k5_idx in triple_to_blue_k5s_arr[i]:
                        if blue_v[k5_idx] == 10: delta -= 1
                        
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    if colors[i] == 1:
                        for k4_idx in triple_to_red_k4s_arr[i]: red_v[k4_idx] -= 1
                        for k5_idx in triple_to_blue_k5s_arr[i]: blue_v[k5_idx] += 1
                    else:
                        for k4_idx in triple_to_red_k4s_arr[i]: red_v[k4_idx] += 1
                        for k5_idx in triple_to_blue_k5s_arr[i]: blue_v[k5_idx] -= 1
                    current_violations += delta
                    if current_violations == 0:
                        return {'n': n, 'colors': colors.tolist()}
                T *= 0.999
                
    # Fallback: return a valid small coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
