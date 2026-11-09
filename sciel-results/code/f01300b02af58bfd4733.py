import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    limit = 55.0
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
        
    seed_val = A[29] - A[30]
    best_val = abs(seed_val)
    best_c = [0] * 30
    best_c[28] = 1
    best_c[29] = -1
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i+1]
        return s
        
    def try_update(new_c, new_sum):
        nonlocal best_val, best_c
        if new_sum != 0:
            new_abs = abs(new_sum)
            if new_abs < best_val:
                best_val = new_abs
                best_c = new_c[:]
                
    def local_search(c):
        current_sum = get_sum(c)
        
        if current_sum != 0:
            try_update(c, current_sum)
        
        steps = 0
        max_steps = 50000
        while time.time() - start_time < limit and steps < max_steps:
            steps += 1
            if current_sum == 0:
                # At zero, try to escape with a single flip
                found = False
                for idx in range(30):
                    old_val = c[idx]
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * A[idx+1]
                        new_sum = current_sum + delta
                        if new_sum != 0 and abs(new_sum) < best_val:
                            c[idx] = new_val
                            current_sum = new_sum
                            try_update(c, new_sum)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Try double flips to escape zero
                    found = False
                    for i in range(30):
                        if found:
                            break
                        for j in range(i+1, 30):
                            if found:
                                break
                            old_i = c[i]
                            old_j = c[j]
                            for new_i in [-1, 0, 1]:
                                if new_i == old_i:
                                    continue
                                for new_j in [-1, 0, 1]:
                                    if new_j == old_j:
                                        continue
                                    delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                                    new_sum = current_sum + delta
                                    if new_sum != 0 and abs(new_sum) < best_val:
                                        c[i] = new_i
                                        c[j] = new_j
                                        current_sum = new_sum
                                        try_update(c, new_sum)
                                        found = True
                                        break
                            if found:
                                break
                    if not found:
                        # Stuck at zero, restart
                        break
            else:
                # Normal hill climbing: try all single flips
                improved = False
                best_move = None
                best_new_sum = current_sum
                for idx in range(30):
                    old_val = c[idx]
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * A[idx+1]
                        new_sum = current_sum + delta
                        if new_sum != 0 and abs(new_sum) < abs(current_sum):
                            if abs(new_sum) < abs(best_new_sum):
                                best_new_sum = new_sum
                                best_move = (idx, new_val)
                if best_move:
                    idx, new_val = best_move
                    c[idx] = new_val
                    current_sum = best_new_sum
                    try_update(c, current_sum)
                    improved = True
                else:
                    # Try double flips
                    found = False
                    for i in range(30):
                        if found:
                            break
                        for j in range(i+1, 30):
                            if found:
                                break
                            old_i = c[i]
                            old_j = c[j]
                            for new_i in [-1, 0, 1]:
                                if new_i == old_i:
                                    continue
                                for new_j in [-1, 0, 1]:
                                    if new_j == old_j:
                                        continue
                                    delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                                    new_sum = current_sum + delta
                                    if new_sum != 0 and abs(new_sum) < abs(current_sum):
                                        c[i] = new_i
                                        c[j] = new_j
                                        current_sum = new_sum
                                        try_update(c, current_sum)
                                        found = True
                                        break
                            if found:
                                break
                    if not found:
                        # Local minimum, try random perturbation
                        if random.random() < 0.3:
                            # Perturb 1-2 random positions
                            num_perturb = random.randint(1, 2)
                            for _ in range(num_perturb):
                                idx = random.randint(0, 29)
                                c[idx] = random.choice([-1, 0, 1])
                            current_sum = get_sum(c)
                            if current_sum != 0:
                                try_update(c, current_sum)
                        else:
                            break

    # Run restarts with better initial configurations
    num_restarts = 0
    while time.time() - start_time < limit:
        num_restarts += 1
        
        # Create a more informed initial configuration
        if num_restarts == 1:
            c = best_c[:]
        else:
            # Mix: start with some structure
            c = [0] * 30
            num_nonzero = random.randint(2, 8)
            positions = random.sample(range(30), num_nonzero)
            for pos in positions:
                c[pos] = random.choice([-1, 1])
        
        local_search(c)
        if best_val == 1:
            break
    
    # Final polish: exhaustive local search around best_c
    if time.time() - start_time < limit - 5:
        c = best_c[:]
        current_sum = get_sum(c)
        for _ in range(100):
            if time.time() - start_time >= limit - 3:
                break
            improved = False
            best_move = None
            best_new_sum = current_sum
            for idx in range(30):
                old_val = c[idx]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * A[idx+1]
                    new_sum = current_sum + delta
                    if new_sum != 0 and abs(new_sum) < abs(best_new_sum):
                        best_new_sum = new_sum
                        best_move = (idx, new_val)
            if best_move:
                idx, new_val = best_move
                c[idx] = new_val
                current_sum = best_new_sum
                try_update(c, current_sum)
                improved = True
            else:
                break
    
    return best_c
