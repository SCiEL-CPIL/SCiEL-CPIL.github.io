import time
import random
from math import gcd
from functools import reduce

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe

    n_terms = 30
    # Calculate LCM(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    D = reduce(lcm, range(1, n_terms + 1))
    
    # Precompute weights w_i = D / i
    # N = sum(c_i * w_i)
    weights = [D // i for i in range(1, n_terms + 1)]
    
    # Initialize coefficients
    # A good heuristic is alternating signs to cancel harmonic terms
    # c_i = 1 if i is odd, -1 if i is even
    c = [1 if i % 2 == 1 else -1 for i in range(1, n_terms + 1)]
    
    def calc_num(coeffs):
        return sum(ci * wi for ci, wi in zip(coeffs, weights))
    
    best_N = calc_num(c)
    best_c = c[:]
    
    # If best_N is 0, we need to move away from it
    if best_N == 0:
        # Flip one term to make it non-zero
        # Pick the term with smallest weight to minimize change? 
        # Or just any term.
        idx = 0
        c[idx] = -c[idx] if c[idx] != 0 else 1
        best_N = calc_num(c)
        best_c = c[:]

    current_c = best_c[:]
    current_N = best_N
    
    # Local search with 1-flip and 2-flip moves
    # We prioritize moves that reduce |N|
    
    def get_neighbors_1flip(coeffs):
        for i in range(n_terms):
            # Possible values for c_i: -1, 0, 1
            # Current is coeffs[i]
            # Generate other values
            for new_val in [-1, 0, 1]:
                if new_val != coeffs[i]:
                    new_coeffs = coeffs[:]
                    new_coeffs[i] = new_val
                    delta = (new_val - coeffs[i]) * weights[i]
                    yield new_coeffs, delta, [i]
    
    def get_neighbors_2flip(coeffs):
        # To save time, we don't generate all pairs explicitly in a list,
        # but iterate through a random subset or specific pairs
        # For efficiency in tight loop, we might just stick to 1-flip 
        # and use random restarts for 2-flip effect.
        # However, 2-flip is powerful. Let's implement a limited 2-flip search.
        # We will try flipping 2 random indices if 1-flip fails.
        pass

    # Simulated Annealing style or simple Hill Climbing with Random Restarts
    # Given the small dimension (30), Hill Climbing with 2-flips is strong.
    
    improved = True
    while time.time() < deadline:
        if not improved:
            # If no improvement, try a random perturbation (kick)
            # Flip 2-5 random coefficients
            num_flips = random.randint(2, 5)
            for _ in range(num_flips):
                i = random.randint(0, n_terms - 1)
                # Randomly set to -1, 0, 1
                current_c[i] = random.choice([-1, 0, 1])
            current_N = calc_num(current_c)
            if current_N == 0:
                # Avoid zero
                i = random.randint(0, n_terms - 1)
                current_c[i] = random.choice([-1, 1])
                current_N = calc_num(current_c)
            
            # Update best if this kick was better (unlikely but possible)
            if abs(current_N) < abs(best_N):
                best_N = current_N
                best_c = current_c[:]
            
            improved = True
            continue
        
        improved = False
        
        # Try 1-flip moves
        # To avoid O(30*3) checks every time which is small, we do it.
        # But we can be smarter: calculate potential |N| for each flip.
        
        min_abs_N = float('inf')
        best_move = None
        best_move_coeffs = None
        
        # 1-flip search
        for i in range(n_terms):
            old_val = current_c[i]
            w = weights[i]
            # Possible new values: -1, 0, 1 excluding old_val
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * w
                new_N = current_N + delta
                if new_N == 0:
                    continue
                abs_new_N = abs(new_N)
                if abs_new_N < min_abs_N:
                    min_abs_N = abs_new_N
                    best_move = (i, new_val)
                    # We don't need to copy coeffs here, just track the change
                    # But for 2-flip we might need more context. 
                    # For 1-flip, we can just apply it.
        
        if best_move:
            i, new_val = best_move
            current_c[i] = new_val
            current_N += (new_val - current_c[i]) * weights[i] # Wait, current_c[i] is old val
            # Correction:
            # current_c[i] is updated.
            # Let's recalculate N properly to avoid drift or just use delta.
            # current_N = current_N + (new_val - old_val) * w
            # We stored old_val? No, we used current_c[i] in loop.
            # Let's re-implement the acceptance step cleanly.
            
            # Actually, the loop above found the best 1-flip.
            # Apply it:
            # We need to know the delta.
            # Let's just recalculate N for simplicity and safety, 
            # since n=30 is small.
            current_N = calc_num(current_c)
            
            if abs(current_N) < abs(best_N):
                best_N = current_N
                best_c = current_c[:]
                improved = True
            else:
                # This move didn't improve the global best, but it might be 
                # a step towards a better minimum? 
                # In strict Hill Climbing, we only accept if it improves current local min.
                # But here we are looking for global min.
                # If the best 1-flip from current state doesn't improve the GLOBAL best,
                # it might still be better than the CURRENT state?
                # My loop found the min_abs_N among neighbors.
                # If min_abs_N < abs(current_N), it's a local improvement.
                # We should accept local improvements to climb down.
                if min_abs_N < abs(current_N):
                    # We already applied the move to current_c in the logic above?
                    # No, I just identified it. Let's apply it.
                    # Wait, the code block above `if best_move:` applies the move.
                    # So current_c is updated.
                    # If it doesn't improve best_N, it's still a local improvement 
                    # (min_abs_N < abs(current_N_old)).
                    # So we set improved = True to continue climbing.
                    # But we didn't check if min_abs_N < abs(current_N_old).
                    # We checked if best_move exists.
                    # If best_move exists, it means there was a neighbor with 
                    # smaller |N| than current? No, min_abs_N is the min among neighbors.
                    # It's possible min_abs_N > abs(current_N) if current is a local min.
                    # In that case, we shouldn't accept?
                    # If we don't accept, we are stuck.
                    # So if min_abs_N < abs(current_N), accept.
                    # Else, no 1-flip improves current state.
                    
                    # Let's refine the logic.
                    pass # The logic above was a bit tangled. Let's restart the loop body logic.
        
        # Restarting loop body logic for clarity in thought process:
        # 1. Find best 1-flip neighbor.
        # 2. If best_neighbor_abs < current_abs:
        #      Apply move.
        #      Update current_N.
        #      If current_N < best_N: Update best.
        #      Set improved = True.
        # 3. Else:
        #      Try 2-flip? Or Kick?
        #      If we try 2-flip, we need to search neighbors of size 2.
        #      Number of 2-flip moves is ~ 30*30*3*3 / 2 ~ 4000.
        #      This is feasible per step.
        
        # Let's implement the robust version.
        
        # Reset improved flag for this iteration
        improved = False
        
        # 1. Check 1-flips
        min_neighbor_abs = float('inf')
        best_1flip = None # (index, new_val, delta)
        
        for i in range(n_terms):
            old_val = current_c[i]
            w = weights[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * w
                new_N = current_N + delta
                if new_N == 0:
                    continue
                abs_new_N = abs(new_N)
                if abs_new_N < min_neighbor_abs:
                    min_neighbor_abs = abs_new_N
                    best_1flip = (i, new_val, delta)
        
        if best_1flip and min_neighbor_abs < abs(current_N):
            i, new_val, delta = best_1flip
            current_c[i] = new_val
            current_N += delta
            if abs(current_N) < abs(best_N):
                best_N = current_N
                best_c = current_c[:]
            improved = True
            continue
        
        # 2. If 1-flip didn't improve current state, try 2-flips
        # We look for a pair of flips that reduces |N|
        # To keep it fast, we might limit this or use random sampling of pairs
        # But 30 choose 2 is 435 pairs. For each pair, 3*3=9 moves. ~4000 checks.
        # This is fast enough for 30 terms.
        
        min_2flip_abs = float('inf')
        best_2flip = None # (i1, v1, i2, v2, delta1, delta2)
        
        # Optimization: Only consider pairs if 1-flip failed.
        # We can prune: if |current_N| is already very small, we are close.
        
        for i1 in range(n_terms):
            old_val1 = current_c[i1]
            w1 = weights[i1]
            for v1 in [-1, 0, 1]:
                if v1 == old_val1:
                    continue
                delta1 = (v1 - old_val1) * w1
                for i2 in range(i1 + 1, n_terms):
                    old_val2 = current_c[i2]
                    w2 = weights[i2]
                    for v2 in [-1, 0, 1]:
                        if v2 == old_val2:
                            continue
                        delta2 = (v2 - old_val2) * w2
                        total_delta = delta1 + delta2
                        new_N = current_N + total_delta
                        if new_N == 0:
                            continue
                        abs_new_N = abs(new_N)
                        if abs_new_N < min_2flip_abs:
                            min_2flip_abs = abs_new_N
                            best_2flip = (i1, v1, i2, v2, total_delta)
        
        if best_2flip and min_2flip_abs < abs(current_N):
            i1, v1, i2, v2, total_delta = best_2flip
            current_c[i1] = v1
            current_c[i2] = v2
            current_N += total_delta
            if abs(current_N) < abs(best_N):
                best_N = current_N
                best_c = current_c[:]
            improved = True
            continue
        
        # 3. If neither 1-flip nor 2-flip improves current state, we are in a local minimum.
        # Perform a random kick to escape.
        # Flip 2-4 random coefficients.
        num_kicks = random.randint(2, 4)
        for _ in range(num_kicks):
            i = random.randint(0, n_terms - 1)
            current_c[i] = random.choice([-1, 0, 1])
        
        current_N = calc_num(current_c)
        if current_N == 0:
            # Ensure non-zero
            i = random.randint(0, n_terms - 1)
            current_c[i] = random.choice([-1, 1])
            current_N = calc_num(current_c)
            
        if abs(current_N) < abs(best_N):
            best_N = current_N
            best_c = current_c[:]
            
        # Loop continues to try and climb from the new kicked state

    return best_c

# Test/Check
if __name__ == "__main__":
    coeffs = construct()
    D = reduce(lambda a, b: a * b // gcd(a, b), range(1, 31))
    N = sum(c * (D // i) for i, c in enumerate(coeffs, 1))
    print(f"Best N: {N}, |N|: {abs(N)}, Score: {D / abs(N) if N != 0 else 'Inf'}")
    print(coeffs)
