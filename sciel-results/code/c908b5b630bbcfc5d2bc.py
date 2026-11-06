"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _set_to_bits(s):
    """Convert a set to a bitset integer where bit i represents presence of value i."""
    bits = 0
    for x in s:
        bits |= (1 << x)
    return bits

def _bits_to_set(bits):
    """Convert a bitset integer to a set."""
    s = set()
    x = 1
    while bits:
        if bits & 1:
            s.add(x)
        bits >>= 1
        x += 1
    return s

def _can_add_bitset(bits, x):
    """
    Check if adding x to the set represented by bits would create a 4-AP.
    Uses bit operations for speed.
    Returns True if x can be added safely.
    """
    # For each possible position of x in a 4-AP, check if the other 3 elements exist
    # Case 1: x is the 4th term (largest): x-3d, x-2d, x-d, x
    # Case 2: x is the 3rd term: x-2d, x-d, x, x+d
    # Case 3: x is the 2nd term: x-d, x, x+d, x+2d
    # Case 4: x is the 1st term: x, x+d, x+2d, x+3d
    
    # We'll check each case by iterating over possible d values
    
    # Case 1: x is 4th term
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x - d)) & 1 and (bits >> (x - 2*d)) & 1 and (bits >> (x - 3*d)) & 1:
            return False
    
    # Case 2: x is 3rd term
    if x + 1 <= N:
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (bits >> (x + d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x - 2*d)) & 1:
                return False
    
    # Case 3: x is 2nd term
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1:
            return False
    
    # Case 4: x is 1st term
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1 and (bits >> (x + 3*d)) & 1:
            return False
    
    return True

def _is_4ap_free_bitset(bits):
    """
    Check if the bitset contains no 4-term arithmetic progression.
    """
    # Extract the set for iteration
    s = _bits_to_set(bits)
    for x in s:
        if not _can_add_bitset(bits ^ (1 << x), x):
            return False
    return True

def _get_conflicts_bitset(bits, x):
    """
    Return a list of 4-APs in the set represented by bits U {x} that contain x.
    """
    conflicts = []
    
    # Case 1: x is the 4th term (largest)
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x - d)) & 1 and (bits >> (x - 2*d)) & 1 and (bits >> (x - 3*d)) & 1:
            conflicts.append((x - 3*d, x - 2*d, x - d, x))
            
    # Case 2: x is the 3rd term
    if x + 1 <= N:
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (bits >> (x + d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x - 2*d)) & 1:
                conflicts.append((x - 2*d, x - d, x, x + d))
                
    # Case 3: x is the 2nd term
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1:
            conflicts.append((x - d, x, x + d, x + 2*d))
            
    # Case 4: x is the 1st term (smallest)
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1 and (bits >> (x + 3*d)) & 1:
            conflicts.append((x, x + d, x + 2*d, x + 3*d))
            
    return conflicts

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Greedy Initialization using bitset
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(bits, x):
            bits |= (1 << x)
    
    # 2. Local Search (Hill Climbing) using bitset
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any element
        candidates = [x for x in range(1, N + 1) if not (bits >> x) & 1]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check if adding x is valid
            if _can_add_bitset(bits, x):
                bits |= (1 << x)
                improved = True
                break  # Restart loop with new set
                
            # If not valid, find conflicts
            conflicts = _get_conflicts_bitset(bits, x)
            if not conflicts:
                continue
                
            # Try to resolve by removing one element from one of the conflicts
            ap = random.choice(conflicts)
            others = [y for y in ap if y != x]
            random.shuffle(others)
            
            for y in others:
                # Tentatively remove y and add x
                new_bits = (bits ^ (1 << y)) | (1 << x)
                # Check if new set is valid
                if _is_4ap_free_bitset(new_bits):
                    bits = new_bits
                    improved = True
                    break  # Restart loop
                    
            if improved:
                break
                
        # If no simple add/swap improved the set, try a "Remove 1, Add 2" move
        if not improved and time.time() < deadline:
            # Pick a random element to remove
            s = _bits_to_set(bits)
            if s:
                y = random.choice(list(s))
                temp_bits = bits ^ (1 << y)
                
                # Try to add as many elements as possible to temp_set
                added = 0
                new_bits = temp_bits
                candidates = [x for x in range(1, N + 1) if not (new_bits >> x) & 1]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add_bitset(new_bits, x):
                        new_bits |= (1 << x)
                        added += 1
                        if added >= 2:
                            # We have a better set
                            bits = new_bits
                            improved = True
                            break

    return sorted(_bits_to_set(bits))
# EVOLVE-BLOCK-END
