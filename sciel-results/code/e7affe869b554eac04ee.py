import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute: for each pair (a,b) with a < b, store the set of c values
    # that would create a forbidden triple if a, b, c, a+b, a+c, b+c are all in the set.
    # Actually, we need: for a given mask, check if any forbidden triple exists.
    # A triple (a,b,c) with a<b<c is forbidden if a+b, a+c, b+c are all in mask.
    
    # Strategy: For each pair (a,b) with a<b and a+b <= N, precompute the bitmask of
    # all c > b such that a+c <= N and b+c <= N. Then for a given mask, if (a,b,a+b) are
    # all in mask, we check if mask has any c from that precomputed set.
    
    # Precompute forbidden_c_masks[a][b] = bitmask of valid c values for pair (a,b)
    # where a < b, a+b <= N, and c > b, a+c <= N, b+c <= N
    forbidden_c_masks = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            # Find all c > b such that a+c <= N and b+c <= N
            c_mask = 0
            max_c = min(N - a, N - b)
            for c in range(b + 1, max_c + 1):
                c_mask |= (1 << c)
            if c_mask != 0:
                forbidden_c_masks[(a, b)] = c_mask
    
    def is_valid_bitset(mask):
        """Check if no three distinct a,b,c in mask have all pair sums in mask."""
        for (a, b), c_mask in forbidden_c_masks.items():
            # Check if a, b, a+b are all in mask
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> (a + b)) & 1):
                # Check if any c in c_mask is also in mask
                if (mask & c_mask) != 0:
                    return False
        return True

    def is_valid_set(s_set):
        """Wrapper for set input."""
        mask = 0
        for x in s_set:
            mask |= (1 << x)
        return is_valid_bitset(mask)

    # Initial candidate: upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    # Local search function using bitsets for speed
    def local_search(current_set, max_steps=20000):
        current_mask = 0
        for x in current_set:
            current_mask |= (1 << x)
        
        steps = 0
        no_improve_count = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            # Try adding elements
            candidates = []
            for x in range(1, N + 1):
                if not (current_mask >> x) & 1:
                    candidates.append(x)
            
            if not candidates:
                break  # All elements are in the set
            
            random.shuffle(candidates)
            
            for elem in candidates:
                trial_mask = current_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    current_mask = trial_mask
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
            
            no_improve_count += 1
            
            # If no single addition works, try swaps (remove 1, add 1)
            if current_mask == 0:
                break
            
            # Pick a random element to remove
            current_elements = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
            if not current_elements:
                break
            to_remove = random.choice(current_elements)
            
            base_mask = current_mask & ~(1 << to_remove)
            
            # Try to add 1 element to base_mask (swap)
            candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
            random.shuffle(candidates)
            
            swap_found = False
            for elem in candidates:
                trial_mask = base_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    current_mask = trial_mask
                    swap_found = True
                    break
            
            if not swap_found:
                # Try 2-for-1 swap? Remove 1, add 2.
                # This increases size by 1.
                candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
                random.shuffle(candidates)
                
                new_mask = base_mask
                added = 0
                for elem in candidates:
                    if added == 2:
                        break
                    trial_mask = new_mask | (1 << elem)
                    if is_valid_bitset(trial_mask):
                        new_mask = trial_mask
                        added += 1
                
                if added == 2:
                    current_mask = new_mask
                    improved = True
                    no_improve_count = 0
                elif added == 1:
                    # Accept the single addition as a swap (size stays same)
                    current_mask = new_mask
                    # Don't count as improvement for no_improve_count reset
                else:
                    # No improvement, check if we should stop
                    if no_improve_count > 200:
                        break

        # Convert back to set
        result_set = set()
        for x in range(1, N + 1):
            if (current_mask >> x) & 1:
                result_set.add(x)
        return result_set

    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        # Different initial sets
        if restart == 0:
            init = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            # Try a mix: upper half plus some lower
            init = set(range(N // 2 + 1, N + 1))
            # Add random lower elements if valid
            lower_candidates = list(range(1, N // 2 + 1))
            random.shuffle(lower_candidates)
            for x in lower_candidates:
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        elif restart == 2:
            # Try: all numbers > 64, plus some strategic lower numbers
            init = set(range(65, N + 1))
            # Add numbers that won't create triples: focus on small numbers
            # If we add a small number a, then for any b,c in upper half,
            # a+b and a+c will be in upper half if a+b > 64 and a+c > 64
            # But b+c > 128 always, so b+c is never in the set.
            # So adding small numbers is safe as long as a+b and a+c are not both in set.
            # Actually, if b,c are both > 64, then a+b > a+64 and a+c > a+64.
            # If a is small (say < 10), then a+b and a+c are in range 65-137, 
            # but only those <= 128 are in the set.
            # The issue is: for triple (a,b,c) with a small, b,c upper:
            # We need a+b, a+c, b+c all in set. b+c > 128 always, so this can't happen.
            # So ANY small number can be added to the upper half!
            # Wait, but what about triples where two are small?
            # If we add multiple small numbers, say a1, a2, and b from upper:
            # Triple (a1, a2, b): need a1+a2, a1+b, a2+b all in set.
            # a1+a2 is small (if a1,a2 < 10, then a1+a2 < 20, not in upper half).
            # So a1+a2 is NOT in the set, so this triple is safe.
            # What about (a1, b1, b2) where a1 is small, b1,b2 are upper?
            # Need a1+b1, a1+b2, b1+b2 all in set. b1+b2 > 128, so safe.
            # So indeed, we can add ALL numbers 1-64 to the upper half?
            # Let me verify: take a=1, b=65, c=66.
            # a+b=66 (in set), a+c=67 (in set), b+c=131 (not in set, >128). Safe.
            # Take a=1, b=2, c=65.
            # a+b=3 (not in set if we only have 1-64... wait, 3 IS in 1-64).
            # a+c=66 (in set), b+c=67 (in set).
            # So (1,2,65) with sums 3,66,67 all in set! That's a forbidden triple!
            # So we CAN'T add all lower numbers.
            
            # The constraint is: if a,b are both in lower half and a+b is also in lower half,
            # and c is in upper half, then a+c and b+c might be in upper half.
            # Specifically, if a+b <= 64, and a+c <= 128 and b+c <= 128,
            # and a+c >= 65 and b+c >= 65, then we have a problem.
            
            # So we need to be careful about which lower numbers to add.
            # Let's try adding lower numbers greedily.
            lower_candidates = list(range(1, N // 2 + 1))
            random.shuffle(lower_candidates)
            for x in lower_candidates:
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        else:
            # Random subset of upper half + some lower
            init = set()
            for x in range(N // 2 + 1, N + 1):
                if random.random() < 0.95: # Higher probability for upper
                    init.add(x)
            for x in range(1, N // 2 + 1):
                if random.random() < 0.5:
                    init.add(x)
            
            # Ensure validity by removing random elements
            while not is_valid_set(init):
                if len(init) == 0:
                    break
                to_remove = random.choice(list(init))
                init.remove(to_remove)
        
        # Run local search
        improved_set = local_search(init)
        
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final polish: try to add any remaining element
    remaining = set(range(1, N + 1)) - best_set
    for elem in remaining:
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid_set(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    result = [int(x) for x in result]
    return result
