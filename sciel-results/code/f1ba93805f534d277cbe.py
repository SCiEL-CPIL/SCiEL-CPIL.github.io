import time
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate primes
    primes = [p for p in range(2, 1001) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    L_max = 5000 # Search up to this limit
    # cover_count[y] = number of classes covering y
    cover_count = np.zeros(L_max + 1, dtype=np.int8)
    
    selected_classes = []
    used_primes = set()
    current_sum = Fraction(0)
    
    def update_coverage(p, r, sign):
        """
        sign = 1 for add, -1 for remove
        """
        # y in [1, L_max] such that y % p == r
        # First such y >= 1:
        # r is in [0, p-1]
        # If r == 0, y = p, 2p, ...
        # If r > 0, y = r, r+p, ...
        # Note: y must be >= 1.
        # If r == 0, first y is p.
        # If r > 0, first y is r.
        
        start_y = r if r > 0 else p
        if start_y > L_max:
            return
        
        # Iterate y
        for y in range(start_y, L_max + 1, p):
            cover_count[y] += sign

    def get_L():
        # Find first y >= 1 where cover_count[y] == 0
        # We can use numpy to find this fast
        # But cover_count[0] is 0, we start from 1
        zeros = np.where(cover_count[1:L_max+1] == 0)[0]
        if len(zeros) == 0:
            return L_max
        return zeros[0] + 1

    # 1. Greedy Initialization
    # We want to cover 1, 2, ...
    # Efficiency: gain * p
    # We iterate x from 1 to L_max
    # If x is covered, skip.
    # Else, pick best p.
    
    # To make it faster, we can just iterate x
    for x in range(1, L_max + 1):
        if time.time() > deadline:
            break
            
        if cover_count[x] > 0:
            continue
            
        # Find best prime
        best_p = None
        best_gain = -1
        best_cost = None
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if current_sum + cost > 1:
                continue
            
            r = x % p
            
            # Calculate gain: how many y in [1, L_max] are covered by [p, r]
            # that are currently uncovered?
            # Actually, we just want to maximize the number of NEWLY covered numbers.
            # Since we are covering x, and we haven't added this class yet,
            # the gain is the number of y such that y % p == r and cover_count[y] == 0.
            
            # Approximate gain by counting all y % p == r in [1, L_max]
            # This is an upper bound, but good enough for greedy.
            # Better: count only uncovered.
            
            # Efficient count:
            # y = r, r+p, ...
            # We can just estimate L_max // p.
            # Let's use exact count of uncovered for better accuracy, but it might be slow.
            # Let's try L_max // p first.
            
            gain = L_max // p
            
            # Tie-breaker: smaller p (denser) or larger p (cheaper)?
            # We want to maximize gain per cost: gain / (1/p) = gain * p.
            # So maximize gain * p.
            score = gain * p
            
            if score > best_gain:
                best_gain = score
                best_p = p
                best_cost = cost
        
        if best_p is None:
            break
            
        used_primes.add(best_p)
        current_sum += best_cost
        r = x % best_p
        selected_classes.append([best_p, r])
        update_coverage(best_p, r, 1)
        
    L_current = get_L()
    
    # 2. Local Search
    # We will try to improve L by changing residues or swapping primes.
    # We maintain a list of classes.
    # classes is a list of [p, r]
    
    # To speed up, we can limit the number of iterations.
    # Each iteration:
    # 1. Try changing residue of each class to cover current L.
    # 2. Try swapping each class with each available prime to cover current L.
    
    available_primes = [p for p in primes if p not in used_primes]
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        L_current = get_L()
        if L_current >= L_max:
            break
            
        x = L_current # The gap we want to cover
        
        best_move = None
        best_new_L = L_current
        
        # --- Move Type 1: Change Residue ---
        for i, (p, r) in enumerate(selected_classes):
            if x % p == r:
                continue # Already covers x
            
            new_r = x % p
            
            # Check if removing [p, r] creates a gap < x
            # We need to check y < x such that y % p == r and cover_count[y] == 1
            # If any such y exists, it becomes 0.
            
            # Find y < x, y % p == r
            # y = r, r+p, ...
            # If r == 0, y = p, 2p...
            
            start_y = r if r > 0 else p
            if start_y > x:
                # No y < x affected
                is_safe = True
            else:
                is_safe = True
                # Iterate y
                # Optimization: only check up to x
                for y in range(start_y, x, p):
                    if cover_count[y] == 1:
                        is_safe = False
                        break
            
            if not is_safe:
                continue
                
            # If safe, calculate new L
            # We remove [p, r] and add [p, new_r]
            # Update cover_count
            update_coverage(p, r, -1)
            update_coverage(p, new_r, 1)
            
            new_L = get_L()
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = ('change', i, new_r)
                
            # Revert
            update_coverage(p, new_r, -1)
            update_coverage(p, r, 1)
            
        # --- Move Type 2: Swap Prime ---
        # For each class i, and each available prime p'
        # Check budget
        # Choose r' = x % p'
        # Check if removing [p, r] is safe
        # Check if adding [p', r'] helps
        
        # To save time, only try a subset of swaps?
        # Or just all?
        # Number of classes ~ 160. Available primes ~ 160.
        # 25600 swaps.
        # Each swap check involves iterating y < x for the removed prime.
        # x ~ 2000. p ~ 13. 2000/13 ~ 150 iterations.
        # 25600 * 150 = 3.8M operations.
        # Plus get_L() which is O(L_max) ~ 5000.
        # 25600 * 5000 = 128M operations.
        # This might be too slow for Python in 110s.
        # We should optimize get_L or limit swaps.
        
        # Let's limit swaps to those where p' is "good".
        # Or just run it and see if it times out.
        # We can break out of the swap loop if we find a move that increases L significantly.
        
        for i, (p, r) in enumerate(selected_classes):
            cost_p = Fraction(1, p)
            
            for p_prime in available_primes:
                cost_pp = Fraction(1, p_prime)
                
                if current_sum - cost_p + cost_pp > 1:
                    continue
                
                new_r = x % p_prime
                
                # Check safety of removing [p, r]
                is_safe = True
                start_y = r if r > 0 else p
                if start_y < x:
                    for y in range(start_y, x, p):
                        if cover_count[y] == 1:
                            is_safe = False
                            break
                
                if not is_safe:
                    continue
                
                # Apply swap
                update_coverage(p, r, -1)
                update_coverage(p_prime, new_r, 1)
                
                new_L = get_L()
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_move = ('swap', i, p_prime, new_r)
                
                # Revert
                update_coverage(p_prime, new_r, -1)
                update_coverage(p, r, 1)
                
                # Early break if we found a perfect move?
                # No, we want the best.
        
        if best_move is not None:
            # Apply the best move
            if best_move[0] == 'change':
                _, i, new_r = best_move
                p, r = selected_classes[i]
                update_coverage(p, r, -1)
                update_coverage(p, new_r, 1)
                selected_classes[i][1] = new_r
            elif best_move[0] == 'swap':
                _, i, p_prime, new_r = best_move
                p, r = selected_classes[i]
                update_coverage(p, r, -1)
                update_coverage(p_prime, new_r, 1)
                selected_classes[i] = [p_prime, new_r]
                used_primes.remove(p)
                used_primes.add(p_prime)
                available_primes.remove(p_prime)
                available_primes.append(p)
                current_sum += Fraction(1, p_prime) - Fraction(1, p)
        
        if best_move is None:
            # No improvement found in this pass
            # We can stop or randomize?
            # Let's stop to avoid infinite loop.
            break
            
    return selected_classes
