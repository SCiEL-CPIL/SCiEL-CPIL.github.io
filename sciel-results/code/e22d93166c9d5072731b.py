import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    
    # Initial Sequence: Primes + Greedy Polish
    current_seq = primes[:]
    current_prods = set()
    # Build prods for initial
    n = len(current_seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= current_seq[j]
            current_prods.add(p)
            
    best_seq = current_seq[:]
    best_size = len(current_seq)
    
    # Greedy Polish to get a better start
    remaining = [x for x in all_numbers if x not in set(current_seq)]
    random.seed(42)
    # Shuffle remaining for variety in insertion attempts
    check_order = remaining[:]
    random.shuffle(check_order)
    
    for c in check_order:
        if time.time() > deadline - 10:
            break
        # Find pos
        pos = 0
        while pos < len(current_seq) and current_seq[pos] < c:
            pos += 1
        if pos < len(current_seq) and current_seq[pos] == c:
            continue
            
        # Check validity of insertion
        # New products involving c
        new_prods = []
        # Left part: current_seq[i] ... c
        left_prod = 1
        for i in range(pos - 1, -1, -1):
            left_prod *= current_seq[i]
            new_prods.append(left_prod * c)
        # Right part: c ... current_seq[j]
        right_prod = 1
        for j in range(pos, len(current_seq)):
            right_prod *= current_seq[j]
            new_prods.append(c * right_prod)
        # Middle part: current_seq[i] ... c ... current_seq[j]
        # This is (left_prod_i * c) * right_prod_j? No.
        # Product is s_i ... s_{pos-1} * c * s_pos ... s_j
        # We can compute this by iterating.
        # But note: new_prods above covers i=pos (c*right) and j=pos (left*c).
        # The spanning ones are i < pos and j >= pos.
        # Actually, the set of all products involving c is { P(i, pos-1) * c * P(pos, j) } for i<=pos-1, j>=pos.
        # Plus c itself (i=pos, j=pos).
        
        # Efficient check:
        # Generate all products involving c.
        # Check if any in current_prods.
        # Check if any duplicate among themselves.
        
        valid = True
        seen_new = set()
        
        # Products ending at c (from left)
        p_left = 1
        for i in range(pos - 1, -1, -1):
            p_left *= current_seq[i]
            prod = p_left * c
            if prod in current_prods or prod in seen_new:
                valid = False
                break
            seen_new.add(prod)
        if not valid: continue
        
        # Products starting at c (to right) and spanning
        # We need to check c * (s_pos ... s_j) and (s_i ... s_{pos-1}) * c * (s_pos ... s_j)
        # Let's just generate all and check.
        
        # Re-do cleanly:
        seen_new = set()
        # 1. c
        if c in current_prods:
            valid = False
        else:
            seen_new.add(c)
            
        if valid:
            # 2. s_i ... c
            p = c
            for i in range(pos - 1, -1, -1):
                p *= current_seq[i]
                if p in current_prods or p in seen_new:
                    valid = False
                    break
                seen_new.add(p)
                
        if valid:
            # 3. c ... s_j
            p = c
            for j in range(pos, len(current_seq)):
                p *= current_seq[j]
                if p in current_prods or p in seen_new:
                    valid = False
                    break
                seen_new.add(p)
                
        if valid:
            # 4. s_i ... c ... s_j (i < pos, j >= pos)
            # We can iterate i from pos-1 down to 0
            # And for each i, iterate j from pos to end
            # This is O(N^2)
            # But we can break early.
            for i in range(pos - 1, -1, -1):
                p_left = 1
                for k in range(i, pos):
                    p_left *= current_seq[k]
                p_base = p_left * c
                # Now extend to the right
                p = p_base
                for j in range(pos, len(current_seq)):
                    p *= current_seq[j]
                    if p in current_prods or p in seen_new:
                        valid = False
                        break
                    seen_new.add(p)
                if not valid:
                    break
                    
        if valid:
            # Accept Add
            current_seq.insert(pos, c)
            current_prods.update(seen_new)
            if len(current_seq) > best_size:
                best_size = len(current_seq)
                best_seq = current_seq[:]
                
    # Simulated Annealing Phase
    # State: current_seq, current_prods
    # We will try Swaps and Adds.
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    best_score = len(current_seq)
    
    while time.time() < deadline - 2:
        if T < T_min:
            T = T_min
            
        # Choose a move
        # 70% Add, 30% Swap
        move_type = random.random()
        
        old_size = len(current_seq)
        old_score = old_size # Assuming valid
        
        if move_type < 0.7:
            # Try Add
            candidates = [x for x in all_numbers if x not in set(current_seq)]
            if not candidates:
                continue
            x = random.choice(candidates)
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < x:
                pos += 1
            if pos < len(current_seq) and current_seq[pos] == x:
                continue
                
            # Check validity
            valid, new_prods_to_add = check_insertion(current_seq, current_prods, x, pos)
            
            if valid:
                # Accept
                current_seq.insert(pos, x)
                current_prods.update(new_prods_to_add)
                new_size = len(current_seq)
                if new_size > best_size:
                    best_size = new_size
                    best_seq = current_seq[:]
            else:
                # Reject or accept with penalty?
                # For simplicity, reject invalid adds in SA phase to keep state valid?
                # Or allow invalid states?
                # Allowing invalid states allows escaping local optima.
                # But checking penalty is expensive.
                # Let's stick to Valid Moves for now, but use Swaps to explore.
                pass
        else:
            # Try Swap: Remove x, Add y
            if len(current_seq) == 0:
                continue
            x = random.choice(current_seq)
            idx_x = current_seq.index(x)
            
            # Remove x
            current_seq.pop(idx_x)
            # Remove products involving x
            # This is tricky. We need to identify which products involved x.
            # Easier: Recalculate prods for the new smaller sequence?
            # O(N^2) to recalc. N=60 -> 3600 ops.
            # This is acceptable for 30% of moves.
            
            # Recalculate prods
            temp_prods = set()
            n = len(current_seq)
            for i in range(n):
                p = 1
                for j in range(i, n):
                    p *= current_seq[j]
                    temp_prods.add(p)
                    
            # Now try to add y
            candidates = [y for y in all_numbers if y not in set(current_seq)]
            if not candidates:
                # Put x back
                current_seq.insert(idx_x, x)
                current_prods = temp_prods # Wait, temp_prods is for seq without x.
                # We need prods for seq WITH x.
                # Let's just restore from best or re-calc.
                # This is getting messy.
                
                # Alternative: Keep a "stack" of changes or just re-calc from scratch if swap fails?
                # Given N is small, re-calc is fast.
                
                # Let's re-calc prods for current_seq (without x)
                # Then pick y.
                # If y adds successfully, keep.
                # If not, put x back and re-calc prods for original.
                
                pass # Logic below
            
            # Pick y
            y = random.choice(candidates)
            pos_y = 0
            while pos_y < len(current_seq) and current_seq[pos_y] < y:
                pos_y += 1
            if pos_y < len(current_seq) and current_seq[pos_y] == y:
                # Conflict, put x back
                current_seq.insert(idx_x, x)
                # Re-calc prods for original
                n = len(current_seq)
                current_prods = set()
                for i in range(n):
                    p = 1
                    for j in range(i, n):
                        p *= current_seq[j]
                        current_prods.add(p)
                continue
                
            # Check if y can be added to current_seq (without x)
            valid, new_prods_to_add = check_insertion(current_seq, temp_prods, y, pos_y)
            
            if valid:
                # Accept Swap
                current_seq.insert(pos_y, y)
                current_prods = temp_prods.union(new_prods_to_add)
                # Note: current_prods was temp_prods. We add new ones.
                # But we need to make sure temp_prods doesn't have conflicts with new_prods_to_add.
                # check_insertion ensures new_prods_to_add are disjoint from temp_prods.
                # And new_prods_to_add are internally distinct.
                # So union is valid.
                
                new_size = len(current_seq)
                # Size is same as old_size (removed 1, added 1)
                # Score is same.
                # Accept to explore.
            else:
                # Reject Swap, restore x
                current_seq.insert(idx_x, x)
                # Re-calc prods for original
                n = len(current_seq)
                current_prods = set()
                for i in range(n):
                    p = 1
                    for j in range(i, n):
                        p *= current_seq[j]
                        current_prods.add(p)
                
        T *= cooling
        
    return best_seq

def check_insertion(seq, prods, x, pos):
    """
    Checks if inserting x at pos in seq results in a valid sequence.
    Returns (valid, new_prods_set)
    """
    n = len(seq)
    seen_new = set()
    
    # 1. x
    if x in prods:
        return False, set()
    seen_new.add(x)
    
    # 2. s_i ... x
    p = x
    for i in range(pos - 1, -1, -1):
        p *= seq[i]
        if p in prods or p in seen_new:
            return False, set()
        seen_new.add(p)
        
    # 3. x ... s_j
    p = x
    for j in range(pos, n):
        p *= seq[j]
        if p in prods or p in seen_new:
            return False, set()
        seen_new.add(p)
        
    # 4. s_i ... x ... s_j
    for i in range(pos - 1, -1, -1):
        p_left = 1
        for k in range(i, pos):
            p_left *= seq[k]
        p_base = p_left * x
        p = p_base
        for j in range(pos, n):
            p *= seq[j]
            if p in prods or p in seen_new:
                return False, set()
            seen_new.add(p)
            
    return True, seen_new
