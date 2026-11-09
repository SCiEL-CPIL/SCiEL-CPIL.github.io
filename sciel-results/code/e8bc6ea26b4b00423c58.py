import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Bitset representation
    # We use Python integers as bitsets. Bit i corresponds to element i.
    # Elements are 1..256, so we use bits 1..256.
    
    # Helper to get list of set bits from a bitset
    def bits_to_list(bits):
        return [i for i in range(1, n + 1) if (bits >> i) & 1]
    
    # Helper to get set bits
    def bits_set(bits):
        res = []
        temp = bits
        while temp:
            lsb = temp & -temp
            index = lsb.bit_length()
            res.append(index)
            temp ^= lsb
        return res

    # Pre-compute potential conflicts? 
    # Since n is small (256), we can check on the fly with bit operations.
    
    def update_masks_add(bits, sums, diffs, x):
        """Returns new sums, diffs if x can be added to set represented by bits, else None"""
        # Check sums
        # New sums: 2*x and x+s for all s in bits
        # We can compute the mask of new sums
        new_sums_mask = 1 << (2 * x)
        temp = bits
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length()
            new_sums_mask |= 1 << (x + s)
            temp ^= lsb
            
        if new_sums_mask & sums:
            return None
            
        # Check diffs
        # New diffs: |x-s| for all s in bits
        new_diffs_mask = 0
        temp = bits
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length()
            d = abs(x - s)
            if d > 0:
                new_diffs_mask |= 1 << d
            temp ^= lsb
            
        if new_diffs_mask & diffs:
            return None
            
        # If we reach here, it's valid
        # Update masks
        sums |= new_sums_mask
        diffs |= new_diffs_mask
        return sums, diffs

    def update_masks_remove(bits, sums, diffs, x):
        """Removes x from set represented by bits, updates sums and diffs"""
        # This is tricky. We need to recompute or carefully subtract.
        # Given the small size, recomputation from scratch for the set might be faster than complex removal logic.
        # Or we can just rebuild the masks for the set.
        pass # We will use a simpler approach: full recompute on remove/swap if needed, or just try add.

    # Given the complexity of removal, let's stick to a strategy that favors addition but allows "undo" by keeping a stack of states.
    # Or, simpler: Local Search by "Adding" but if stuck, randomly remove an element and try again.
    
    def validate_set(bits):
        """Check if a set represented by bits is a valid Sidon set"""
        elems = bits_to_list(bits)
        if not elems:
            return True
        sums = set()
        diffs = set()
        valid = True
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            for j in range(i + 1, len(elems)):
                d = abs(elems[i] - elems[j])
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        return valid

    # Main Loop: Simulated Annealing / Local Search
    # State: A_bits, B_bits, sums_A, diffs_A, diffs_B
    # We will maintain the masks incrementally for additions.
    # For removals, we will recompute the masks for that set.
    
    def recompute_masks(bits):
        """Compute sums and diffs masks for a given bitset"""
        elems = bits_to_list(bits)
        sums_mask = 0
        diffs_mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                sums_mask |= 1 << (elems[i] + elems[j])
            for j in range(i + 1, len(elems)):
                diffs_mask |= 1 << abs(elems[i] - elems[j])
        return sums_mask, diffs_mask

    def try_add(x, side, state):
        """
        state: dict with keys 'A', 'B', 'sums_A', 'diffs_A', 'diffs_B'
        side: 0 for A, 1 for B
        Returns new state or None
        """
        A, B = state['A'], state['B']
        sums_A, diffs_A = state['sums_A'], state['diffs_A']
        sums_B, diffs_B = state['sums_B'], state['diffs_B']
        
        if side == 0:
            if (A >> x) & 1: return None
            new_sums, new_diffs = update_masks_add(A, sums_A, diffs_A, x)
            if new_sums is None: return None
            if new_diffs & diffs_B: return None # Check cross diff
            # Note: update_masks_add checks internal diff collision. 
            # We also need to check if new_diffs collides with diffs_B.
            # And we need to ensure the new element doesn't create a diff that collides with diffs_A? 
            # Wait, update_masks_add checks `new_diffs & diffs`. 
            # So `new_diffs & diffs_A` is checked.
            # We just need to add the cross check.
            return {
                'A': A | (1 << x),
                'B': B,
                'sums_A': new_sums,
                'diffs_A': new_diffs,
                'sums_B': sums_B,
                'diffs_B': diffs_B
            }
        else:
            if (B >> x) & 1: return None
            new_sums, new_diffs = update_masks_add(B, sums_B, diffs_B, x)
            if new_sums is None: return None
            if new_diffs & diffs_A: return None
            return {
                'A': A,
                'B': B | (1 << x),
                'sums_A': sums_A,
                'diffs_A': diffs_A,
                'sums_B': new_sums,
                'diffs_B': new_diffs
            }

    def get_score(state):
        return bin(state['A']).count('1') + bin(state['B']).count('1')

    # Initialize best state
    best_state = None
    best_score = 0
    
    # Random restarts with Local Search
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Start with empty sets
        current = {
            'A': 0, 'B': 0,
            'sums_A': 0, 'diffs_A': 0,
            'sums_B': 0, 'diffs_B': 0
        }
        
        # Greedy initialization
        available = list(range(1, n + 1))
        random.shuffle(available)
        
        # Try to build up greedily first
        for x in available:
            if time.time() > deadline: break
            next_state_A = try_add(x, 0, current)
            next_state_B = try_add(x, 1, current)
            
            if next_state_A and next_state_B:
                # Prefer the one that leaves more room? Or random.
                if random.random() < 0.5:
                    current = next_state_A
                else:
                    current = next_state_B
            elif next_state_A:
                current = next_state_A
            elif next_state_B:
                current = next_state_B
        
        # Local Search Phase
        no_improvement_count = 0
        while time.time() < deadline:
            # Try to add
            improved = False
            # Pick a random few candidates
            candidates = [x for x in range(1, n + 1) if not ((current['A'] | current['B']) >> x) & 1]
            random.shuffle(candidates)
            
            for x in candidates[:10]: # Try only a few to keep it fast
                next_A = try_add(x, 0, current)
                if next_A:
                    current = next_A
                    improved = True
                    break
                next_B = try_add(x, 1, current)
                if next_B:
                    current = next_B
                    improved = True
                    break
            
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # If stuck, try a swap or removal
                # Simple removal: remove a random element from A or B
                # Then try to add something else.
                if no_improvement_count > 10:
                    # Remove random element from A or B
                    all_elements = bits_to_list(current['A'] | current['B'])
                    if all_elements:
                        x_to_remove = random.choice(all_elements)
                        if (current['A'] >> x_to_remove) & 1:
                            # Remove from A
                            new_A = current['A'] & ~(1 << x_to_remove)
                            sums_A, diffs_A = recompute_masks(new_A)
                            current['A'] = new_A
                            current['sums_A'] = sums_A
                            current['diffs_A'] = diffs_A
                        else:
                            # Remove from B
                            new_B = current['B'] & ~(1 << x_to_remove)
                            sums_B, diffs_B = recompute_masks(new_B)
                            current['B'] = new_B
                            current['sums_B'] = sums_B
                            current['diffs_B'] = diffs_B
                        no_improvement_count = 0 # Reset counter after perturbation
                    else:
                        break # Empty sets, nothing to do
                
                if no_improvement_count > 100:
                    break # Stuck hard, break to next restart
        
        score = get_score(current)
        if score > best_score:
            best_score = score
            best_state = current.copy()
            # Optional: Verify best_state
            # print(f"New best: {best_score}")

    if best_state is None:
        return [[1], [2]] # Fallback
    
    A_list = bits_to_list(best_state['A'])
    B_list = bits_to_list(best_state['B'])
    
    # Ensure non-empty
    if not A_list: A_list = [1]
    if not B_list: B_list = [2]
    
    return [A_list, B_list]

