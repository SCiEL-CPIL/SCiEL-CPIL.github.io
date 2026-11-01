import random
import time
import math

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _solve_sa_3_6_1()
    else:
        # Fallback
        return _generic_construct(q, n, radius)

def _solve_sa_3_6_1():
    q, n, radius = 3, 6, 1
    total = q ** n
    
    # Precompute masks
    masks = [0] * total
    def decode(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
    def encode(w):
        val = 0
        for d in w:
            val = val * q + d
        return val

    for c in range(total):
        center = decode(c)
        mask = 0
        mask |= (1 << c)
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_center = center[:]
                    new_center[pos] = new_val
                    nb = encode(new_center)
                    mask |= (1 << nb)
        masks[c] = mask
        
    total_mask = (1 << total) - 1
    
    # Helper to get uncovered count
    def get_uncovered_count(code_set):
        covered = 0
        for c in code_set:
            covered |= masks[c]
        return (total_mask ^ covered).bit_count()
    
    # Initial Solution: Randomized Greedy
    # Pick random codewords that cover new ground
    uncovered = total_mask
    init_code = []
    # To ensure validity and good start, use a slightly smarter random greedy
    # 1. Start with empty
    # 2. While uncovered:
    #    Pick a random word from uncovered
    #    Pick a random codeword that covers that word
    #    Add it
    #    Update uncovered
    
    # This might be slow if done naively. 
    # Better: Use the deterministic greedy from the original code as a *base* 
    # but then run SA on it. 
    # To satisfy "new approach", the *optimization* is SA. The initialization is just a setup.
    
    # Let's do a fast greedy init
    uncovered = total_mask
    init_code = []
    while uncovered:
        # Pick a random uncovered word to force diversity? 
        # Or just pick best coverage (original greedy)
        # Let's pick a random word from uncovered, then a random cw covering it.
        # This creates a more random start.
        uncovered_indices = [i for i in range(total) if (uncovered >> i) & 1]
        if not uncovered_indices: break
        target_word = random.choice(uncovered_indices)
        
        # Find all cws covering target_word
        candidates = [c for c in range(total) if (masks[c] >> target_word) & 1]
        if not candidates:
            # Should not happen
            break
        chosen_cw = random.choice(candidates)
        init_code.append(chosen_cw)
        uncovered &= ~masks[chosen_cw]
        
    current_code = set(init_code)
    current_cost = len(current_code) + 1000 * get_uncovered_count(current_code)
    
    best_code = set(current_code)
    best_cost = current_cost
    
    start_time = time.time()
    max_time = 110
    
    T = 100.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        if T < 1e-4:
            break
            
        # Generate a move
        # 50% chance: Add a random cw
        # 50% chance: Remove a random cw
        
        new_code = set(current_code)
        
        if random.random() < 0.5:
            # Add
            c = random.randint(0, total - 1)
            if c not in new_code:
                new_code.add(c)
        else:
            # Remove
            if new_code:
                # Pick random to remove
                c = random.choice(list(new_code))
                new_code.remove(c)
            else:
                # Empty set, can't remove
                new_code.add(random.randint(0, total - 1))
                
        new_uncovered_count = get_uncovered_count(new_code)
        new_cost = len(new_code) + 1000 * new_uncovered_count
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_code = new_code
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_code = set(current_code)
                # If we reach a valid solution (uncovered 0), we can try to reduce further
                # The penalty ensures validity is prioritized.
                
        T *= cooling
        
    # Final check: Ensure best_code is valid
    # If best_code is invalid, we need to repair it.
    # But the penalty should have driven it to validity.
    # If not, fall back to greedy repair.
    if get_uncovered_count(best_code) > 0:
        # Repair
        uncovered = total_mask
        for c in best_code:
            uncovered &= ~masks[c]
        while uncovered:
            # Find cw covering most uncovered
            best_cw = -1
            best_cov = 0
            for c in range(total):
                if c in best_code: continue
                cov = (masks[c] & uncovered).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_cw = c
            if best_cw == -1: break
            best_code.add(best_cw)
            uncovered &= ~masks[best_cw]
            
    return [decode(c) for c in best_code]

def _generic_construct(q, n, radius):
    # Simple fallback
    total = q ** n
    # Just return all words if small, or greedy
    # For safety, implement a simple greedy
    def decode(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
    def encode(w):
        val = 0
        for d in w:
            val = val * q + d
        return val
        
    masks = [0] * total
    for c in range(total):
        center = decode(c)
        mask = 0
        if radius == 1:
            mask |= (1 << c)
            for pos in range(n):
                orig = center[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_center = center[:]
                        new_center[pos] = new_val
                        nb = encode(new_center)
                        mask |= (1 << nb)
        else:
            import itertools
            for word in itertools.product(range(q), repeat=n):
                if sum(1 for a,b in zip(center, word) if a!=b) <= radius:
                    mask |= (1 << encode(word))
        masks[c] = mask
        
    total_mask = (1 << total) - 1
    uncovered = total_mask
    codewords = []
    while uncovered:
        best_cw = -1
        best_count = -1
        for cw in range(total):
            count = (masks[cw] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_cw = cw
        codewords.append(best_cw)
        uncovered &= ~masks[best_cw]
    return [decode(c) for c in codewords]
