import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    def validate_and_update(candidate_set):
        """Validate a set and update best if valid and larger."""
        nonlocal best_n, best_set
        n = len(candidate_set)
        if n <= best_n:
            return False
        
        # Bitset DP: bits[k] is a bitmask where bit 'sum' is 1 if 'sum' is achievable with exactly k elements
        bits = [0] * (n + 1)
        bits[0] = 1  # Sum 0 is achievable with 0 elements
        
        for x in candidate_set:
            # Update bits in reverse order to avoid using the same element multiple times
            for k in range(n - 1, -1, -1):
                if bits[k]:
                    bits[k + 1] |= bits[k] << x
        
        # Check for disjointness: if any sum is achievable with two different cardinalities, it's invalid
        seen_bits = 0
        for k in range(n + 1):
            if bits[k] & seen_bits:
                return False
            seen_bits |= bits[k]
        
        best_n = n
        best_set = candidate_set[:]
        return True
    
    # --- Strategy 1: Greedy construction with multiple candidate orderings ---
    # For each ordering, greedily add elements if they don't cause a collision.
    # We maintain the bitset DP incrementally for efficiency.
    
    candidate_orderings = []
    
    # Natural order
    candidate_orderings.append(list(range(1, 257)))
    # Reverse order
    candidate_orderings.append(list(range(256, 0, -1)))
    # Even first, then odd
    candidate_orderings.append(list(range(2, 257, 2)) + list(range(1, 257, 2)))
    # Odd first, then even
    candidate_orderings.append(list(range(1, 257, 2)) + list(range(2, 257, 2)))
    
    # Multiples-based orderings
    for base in [2, 3, 4, 5, 6, 8]:
        fam = []
        for mult in range(1, 257 // base + 1):
            val = base * mult
            if 1 <= val <= 256:
                fam.append(val)
        for x in range(1, 257):
            if x not in fam:
                fam.append(x)
        candidate_orderings.append(fam)
    
    # Random shuffles
    for _ in range(30):
        fam = list(range(1, 257))
        random.shuffle(fam)
        candidate_orderings.append(fam)
    
    # Greedy construction for each ordering
    for ordering in candidate_orderings:
        if time.time() > deadline:
            break
        
        current_set = []
        # bits[k] for current set
        # We'll use a list that can grow up to size 60
        max_possible = 60
        bits = [0] * (max_possible + 1)
        bits[0] = 1
        used = set()
        
        for x in ordering:
            if time.time() > deadline:
                break
            if x in used:
                continue
            
            # Check if adding x causes a collision
            # For each k, new sums at cardinality k+1 are (bits[k] << x)
            # These must not overlap with bits[j] for any j != k+1
            valid = True
            current_size = len(current_set)
            
            for k in range(current_size, -1, -1):
                if bits[k] == 0:
                    continue
                new_sums = bits[k] << x
                # Check collision with all other cardinalities
                for j in range(current_size + 2):
                    if j == k + 1:
                        continue
                    if j < len(bits) and bits[j] & new_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update bits in reverse order
                for k in range(current_size, -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
                current_set.append(x)
                used.add(x)
                
                if len(current_set) > best_n:
                    best_n = len(current_set)
                    best_set = current_set[:]
                
                if best_n >= 60:
                    return [int(x) for x in best_set]
    
    # --- Strategy 2: Random restarts with local improvement ---
    # Try to find larger sets by random sampling and then greedily extending
    
    if best_n < 60 and time.time() < deadline - 5:
        attempts = 0
        while time.time() < deadline - 2 and attempts < 2000:
            attempts += 1
            target_n = best_n + 1
            if target_n > 60:
                break
            
            # Generate a random set of size target_n
            candidates = random.sample(range(1, 257), target_n)
            
            # Validate
            n = len(candidates)
            bits = [0] * (n + 1)
            bits[0] = 1
            for x in candidates:
                for k in range(n - 1, -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
            
            seen_bits = 0
            valid = True
            for k in range(n + 1):
                if bits[k] & seen_bits:
                    valid = False
                    break
                seen_bits |= bits[k]
            
            if valid:
                best_n = target_n
                best_set = candidates[:]
    
    # --- Strategy 3: Structured gap-based construction ---
    # Try to build a set where min sum of (k+1) elements > max sum of k elements
    # This guarantees disjointness.
    
    if best_n < 60 and time.time() < deadline - 2:
        # Try different base structures
        for start_val in range(1, 20):
            if time.time() > deadline:
                break
            current_set = []
            current_max = 0
            for _ in range(60):
                # Next element must be > current_max to maintain gap property
                # Actually, the condition is more complex. Let's just try powers of 2 and similar.
                pass
        
        # Try superincreasing-like sequences
        for base in range(2, 6):
            seq = []
            val = 1
            while val <= 256 and len(seq) < 60:
                seq.append(val)
                val = val * base
            if len(seq) > best_n:
                # Validate
                n = len(seq)
                bits = [0] * (n + 1)
                bits[0] = 1
                for x in seq:
                    for k in range(n - 1, -1, -1):
                        if bits[k]:
                            bits[k + 1] |= bits[k] << x
                seen_bits = 0
                valid = True
                for k in range(n + 1):
                    if bits[k] & seen_bits:
                        valid = False
                        break
                    seen_bits |= bits[k]
                if valid:
                    best_n = n
                    best_set = seq[:]
    
    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}

# EVOLVE-BLOCK-END
