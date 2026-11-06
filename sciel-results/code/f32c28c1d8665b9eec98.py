"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    # All odds are indices 0,2,4,...,254 (numbers 1,3,5,...,255)
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def get_conflict_count(idx, current_mask):
        """Count conflicts between idx and elements in current_mask using bit operations."""
        # conflict_mask[idx] & current_mask gives a bitmask of conflicting elements
        # Count bits
        cm = conflict_mask[idx] & current_mask
        count = 0
        while cm:
            cm &= cm - 1
            count += 1
        return count
    
    def greedy_add_evens(start_mask, even_order, deadline):
        """Greedily add evens from even_order that don't conflict with current set."""
        current_mask = start_mask
        for e in even_order:
            if time.time() > deadline:
                break
            if (current_mask >> e) & 1:
                continue
            # Check if e conflicts with any element in current
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
        return current_mask
    
    def local_search(current_mask, deadline, max_iters=50000):
        """Local search using bitmask operations."""
        best_mask = current_mask
        best_size = bin(current_mask).count('1')
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding a random element not in current
            if random.random() < 0.8:
                # Find elements not in current
                not_in = ((1 << N) - 1) & ~current_mask
                if not_in == 0:
                    break
                # Pick a random bit from not_in
                # Use a simple method: pick random index
                e = random.randint(0, N - 1)
                if (current_mask >> e) & 1:
                    continue
                # Check conflicts
                if (conflict_mask[e] & current_mask) == 0:
                    new_mask = current_mask | (1 << e)
                    current_mask = new_mask
                    sz = bin(current_mask).count('1')
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
                else:
                    # Try swapping: remove one conflicting element, add e
                    conflicts = conflict_mask[e] & current_mask
                    # Pick a random conflicting element
                    # Extract a random bit from conflicts
                    c = random.randint(0, N - 1)
                    # Find a bit in conflicts
                    while conflicts:
                        # Pick lowest set bit
                        lsb = conflicts & (-conflicts)
                        c = lsb.bit_length() - 1
                        if random.random() < 0.3:
                            break
                        conflicts ^= lsb
                    new_mask = (current_mask & ~(1 << c)) | (1 << e)
                    # Check validity: for each element in new_mask, check no conflicts
                    # This is expensive, so only do it if size doesn't decrease
                    if bin(new_mask).count('1') >= bin(current_mask).count('1'):
                        # Quick check: e should not conflict with new_mask minus c
                        if (conflict_mask[e] & (new_mask & ~(1 << e))) == 0:
                            current_mask = new_mask
                            sz = bin(current_mask).count('1')
                            if sz > best_size:
                                best_mask = current_mask
                                best_size = sz
            else:
                # Try 2-opt: remove one, add one or two
                if current_mask == 0:
                    break
                # Pick random element to remove
                to_remove = random.randint(0, N - 1)
                if not ((current_mask >> to_remove) & 1):
                    continue
                trial = current_mask & ~(1 << to_remove)
                # Try to add up to 2 elements
                not_in = ((1 << N) - 1) & ~trial
                added = 0
                for _ in range(100):  # try up to 100 random candidates
                    if added >= 2:
                        break
                    e = random.randint(0, N - 1)
                    if (trial >> e) & 1:
                        continue
                    if (conflict_mask[e] & trial) == 0:
                        trial |= (1 << e)
                        added += 1
                if bin(trial).count('1') > bin(current_mask).count('1'):
                    current_mask = trial
                    sz = bin(current_mask).count('1')
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
        
        return best_mask
    
    # Start with all odds
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)
    
    best_overall = base_mask
    best_size = bin(base_mask).count('1')
    
    # Strategy 1: Greedy add evens in natural order
    current = greedy_add_evens(base_mask, evens_indices, DEADLINE)
    if bin(current).count('1') > best_size:
        best_overall = current
        best_size = bin(current).count('1')
    
    # Strategy 2: Greedy add evens in reverse order
    current = greedy_add_evens(base_mask, list(reversed(evens_indices)), DEADLINE)
    if bin(current).count('1') > best_size:
        best_overall = current
        best_size = bin(current).count('1')
    
    # Strategy 3: Multiple random orderings
    for attempt in range(50):
        if time.time() > DEADLINE:
            break
        random_order = evens_indices[:]
        random.shuffle(random_order)
        current = greedy_add_evens(base_mask, random_order, DEADLINE)
        sz = bin(current).count('1')
        if sz > best_size:
            best_overall = current
            best_size = sz
    
    # Local search from best
    if time.time() < DEADLINE - 15:
        improved = local_search(best_overall, DEADLINE, max_iters=100000)
        sz = bin(improved).count('1')
        if sz > best_size:
            best_overall = improved
            best_size = sz
    
    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])
    
    return result
# EVOLVE-BLOCK-END
