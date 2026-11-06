import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 256
    
    # Precomputation
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def is_valid_add(s, x):
        """Check if adding x to s creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s: return False
            elif x == b:
                if a in s and c in s and d in s: return False
            elif x == c:
                if a in s and b in s and d in s: return False
            else: # x == d
                if a in s and b in s and c in s: return False
        return True

    def delta_add(s, x):
        """Change in 3-AP count if we add x to s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s: delta += 1
            elif x == b:
                if a in s and c in s: delta += 1
            else: # x == c
                if a in s and b in s: delta += 1
        return delta

    def delta_swap(s, x_out, x_in):
        """
        Calculate change in 3-AP count if x_out is removed and x_in is added.
        Assumes x_out in s and x_in not in s.
        """
        # Loss from removing x_out
        loss = 0
        for ap_idx in elem_to_3aps[x_out]:
            a, b, c = three_aps[ap_idx]
            others = []
            if x_out == a: others = [b, c]
            elif x_out == b: others = [a, c]
            else: others = [a, b]
            
            # The other two must be in s (and not x_out, which is true)
            # But wait, if one of the others is x_in? x_in is not in s yet.
            # So we just check if they are in s.
            if all(y in s for y in others):
                loss += 1
                
        # Gain from adding x_in
        # The set is effectively s - {x_out} + {x_in}
        # But x_in is not in s.
        # So we check if the other two are in s.
        # If one of the other two is x_out, it is removed, so it's not in the new set.
        # So we must check: other two in s AND other two != x_out.
        gain = 0
        for ap_idx in elem_to_3aps[x_in]:
            a, b, c = three_aps[ap_idx]
            others = []
            if x_in == a: others = [b, c]
            elif x_in == b: others = [a, c]
            else: others = [a, b]
            
            valid = True
            for y in others:
                if y not in s:
                    valid = False
                    break
                if y == x_out:
                    valid = False
                    break
            if valid:
                gain += 1
                
        return gain - loss

    def is_valid_swap(s, x_out, x_in):
        """Check if S - {x_out} + {x_in} is valid (no 4-AP)."""
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a: others = [b, c, d]
            elif x_in == b: others = [a, c, d]
            elif x_in == c: others = [a, b, d]
            else: others = [a, b, c]
            
            valid = True
            for y in others:
                if y not in s:
                    valid = False
                    break
                if y == x_out:
                    valid = False
                    break
            if valid:
                return False
        return True

    # Greedy Initialization
    s = set()
    for x in range(1, N + 1):
        if is_valid_add(s, x):
            s.add(x)
            
    best_s = set(s)
    best_score = count_3aps(best_s)
    current_score = best_score

    # Local Search: Best-Improvement Hill Climbing
    # With Random Kicks to escape local optima
    
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        
        # Find best swap
        best_delta = 0
        best_x_out = None
        best_x_in = None
        
        x_out_list = list(s)
        x_in_list = [x for x in range(1, N + 1) if x not in s]
        
        if not x_in_list:
            break
            
        for x_out in x_out_list:
            for x_in in x_in_list:
                if is_valid_swap(s, x_out, x_in):
                    d = delta_swap(s, x_out, x_in)
                    if d > best_delta:
                        best_delta = d
                        best_x_out = x_out
                        best_x_in = x_in
                        
        if best_delta > 0:
            s.remove(best_x_out)
            s.add(best_x_in)
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_s = set(s)
        else:
            # Local optimum reached. Perform a random kick to escape.
            # Try a few random swaps, accepting even if score drops, 
            # but ensure validity.
            kicked = False
            for _ in range(10): # Try up to 10 random swaps
                if not s or not x_in_list:
                    break
                x_out = random.choice(x_out_list)
                x_in = random.choice(x_in_list)
                
                if is_valid_swap(s, x_out, x_in):
                    d = delta_swap(s, x_out, x_in)
                    # Accept if it improves, or with some probability if it worsens
                    # For simplicity, accept any valid swap to change state
                    # Or use a temperature-like decay? 
                    # Let's just accept if d > -5 (small penalty)
                    if d > -5:
                        s.remove(x_out)
                        s.add(x_in)
                        current_score += d
                        kicked = True
                        # Update lists
                        x_out_list = list(s)
                        x_in_list = [x for x in range(1, N + 1) if x not in s]
                        break
                        
            if not kicked:
                # If we can't even kick, maybe try removing an element?
                # Or just break?
                # Let's try removing a random element to allow different additions
                if s:
                    x_out = random.choice(list(s))
                    s.remove(x_out)
                    current_score -= delta_add(s, x_out) # Wait, delta_add is for adding. 
                    # We need delta_remove. 
                    # delta_remove(s, x) = - (gain if added to s\x) 
                    # Actually, loss = number of 3-APs containing x and 2 others in s.
                    # This is the same as delta_add(s\x, x).
                    # So loss = delta_add(s - {x}, x).
                    # But we already removed x. 
                    # Let's just recalculate score for simplicity in kick, or 
                    # calculate delta_remove properly.
                    
                    # For now, if we remove, we lose 3-APs.
                    # Let's just break if we can't kick.
                pass

    return sorted(best_s)
