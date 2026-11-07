import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    # Try multiple structured candidate families
    # Each family is a list of candidates to try in order
    families = []
    
    # Family 1: All integers 1..256 in natural order
    families.append(list(range(1, 257)))
    
    # Family 2: All integers 1..256 in reverse order
    families.append(list(range(256, 0, -1)))
    
    # Family 3: Even numbers first, then odd
    families.append(list(range(2, 257, 2)) + list(range(1, 257, 2)))
    
    # Family 4: Multiples of 2, then 3, etc.
    for base in range(2, 10):
        fam = []
        for mult in range(1, 257 // base + 1):
            val = base * mult
            if 1 <= val <= 256:
                fam.append(val)
        # Add remaining numbers not in this family
        for x in range(1, 257):
            if x not in fam:
                fam.append(x)
        families.append(fam)
    
    # Family 5: Random shuffles (multiple)
    for _ in range(20):
        fam = list(range(1, 257))
        random.shuffle(fam)
        families.append(fam)
    
    # For each family, try greedy construction
    for fam in families:
        if time.time() > deadline:
            break
        
        current_set = []
        # bits[k] is a bitmask where bit 'sum' is 1 if 'sum' is achievable with exactly k elements
        bits = [0] * 61  # Max size 60, so cardinality 0..60
        bits[0] = 1  # Sum 0 is achievable with 0 elements
        
        used = set()
        
        for x in fam:
            if time.time() > deadline:
                break
            if x in used:
                continue
            
            # Check if adding x would cause a collision
            # New bits: for each k, new_sums = bits[k] << x
            # Collision if (bits[k] << x) & bits[j] for any j != k+1
            # Actually, we need to check: for each k, the new sums at cardinality k+1
            # must not overlap with any existing sums at cardinality k+1 (from other subsets)
            # Wait, no: the condition is that sums from DIFFERENT cardinalities must be distinct.
            # So when we add x, for each k, the new sums are (bits[k] << x) at cardinality k+1.
            # These new sums must not overlap with ANY existing sums at cardinality != k+1.
            # But also, the new sums at cardinality k+1 can overlap with existing sums at cardinality k+1 (that's allowed).
            
            # So: for each k from 0 to len(current_set)-1:
            #   new_sums = bits[k] << x
            #   These are sums achievable with k+1 elements (using x)
            #   They must not overlap with bits[j] for any j != k+1
            
            valid = True
            for k in range(len(current_set), -1, -1):
                if bits[k] == 0:
                    continue
                new_sums = bits[k] << x
                # Check collision with all other cardinalities
                for j in range(0, len(current_set) + 2):
                    if j == k + 1:
                        continue
                    if j < len(bits) and bits[j] & new_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x to the set
                # Update bits in reverse order
                for k in range(len(current_set), -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
                current_set.append(x)
                used.add(x)
                
                if len(current_set) > best_n:
                    best_n = len(current_set)
                    best_set = current_set[:]
                
                if best_n >= 60:
                    break
        else:
            continue
        if best_n >= 60:
            break
    
    # If we haven't found a good set, try random restarts with the best structure
    if best_n < 60 and time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Try a random set of size best_n + 1
            target_n = best_n + 1
            if target_n > 60:
                break
            
            candidates = random.sample(range(1, 257), target_n)
            # Validate
            bits = [0] * (target_n + 1)
            bits[0] = 1
            for x in candidates:
                for k in range(target_n - 1, -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
            
            # Check disjointness
            seen_bits = 0
            valid = True
            for k in range(target_n + 1):
                if bits[k] & seen_bits:
                    valid = False
                    break
                seen_bits |= bits[k]
            
            if valid:
                best_n = target_n
                best_set = candidates
    
    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}

# The construct function is defined above.
# We need to make sure it's accessible as construct().
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had:
# def construct():
#    return ...
# So we just expose construct.

# EVOLVE-BLOCK-END
