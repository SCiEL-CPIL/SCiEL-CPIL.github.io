"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512
MAX_VAL = N + 1  # We use 1-indexed, so values are 1..512

def _is_4ap_free_bitset(bits):
    """
    Check if the bitset (an integer where bit i represents presence of value i)
    contains no 4-term arithmetic progression.
    Uses bit operations for speed.
    """
    # For each possible 4th term x and difference d, check if x, x-d, x-2d, x-3d are all present
    # We'll iterate through all possible (x, d) pairs
    for x in range(4, N + 1):
        # x is the 4th term: need x-3d, x-2d, x-d, x all present
        # d can range from 1 to (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            # Check if all four bits are set
            if (bits >> (x-1)) & 1 and (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1 and (bits >> (x-3*d-1)) & 1:
                return False
    return True

def _can_add_bitset(bits, x):
    """
    Check if adding x to the set represented by bits would create a 4-AP.
    Only checks 4-APs that involve x.
    """
    # Case 1: x is the 4th term (largest)
    # x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1 and (bits >> (x-3*d-1)) & 1:
            return False
            
    # Case 2: x is the 3rd term
    # x-2d, x-d, x, x+d
    # Need x+d in set, and x-2d, x-d in set
    if x + 1 <= N:
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (bits >> (x+d-1)) & 1 and (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1:
                return False
                
    # Case 3: x is the 2nd term
    # x-d, x, x+d, x+2d
    # Need x-d, x+d, x+2d in set
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (bits >> (x-d-1)) & 1 and (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1:
            return False
            
    # Case 4: x is the 1st term (smallest)
    # x, x+d, x+2d, x+3d
    # Need x+d, x+2d, x+3d in set
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1 and (bits >> (x+3*d-1)) & 1:
            return False
            
    return True

def _get_conflicts_bitset(bits, x):
    """
    Return a list of 4-APs in set(bits) U {x} that contain x.
    Each AP is a tuple of 4 elements.
    """
    conflicts = []
    
    # Case 1: x is the 4th term (largest)
    # x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1 and (bits >> (x-3*d-1)) & 1:
            conflicts.append((x - 3*d, x - 2*d, x - d, x))
            
    # Case 2: x is the 3rd term
    # x-2d, x-d, x, x+d
    if x + 1 <= N:
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (bits >> (x+d-1)) & 1 and (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1:
                conflicts.append((x - 2*d, x - d, x, x + d))
                
    # Case 3: x is the 2nd term
    # x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (bits >> (x-d-1)) & 1 and (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1:
            conflicts.append((x - d, x, x + d, x + 2*d))
            
    # Case 4: x is the 1st term (smallest)
    # x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1 and (bits >> (x+3*d-1)) & 1:
            conflicts.append((x, x + d, x + 2*d, x + 3*d))
            
    return conflicts

def _bits_to_set(bits):
    """Convert bitset to a set of integers."""
    return {i+1 for i in range(N) if (bits >> i) & 1}

def _set_to_bits(s):
    """Convert a set of integers to a bitset."""
    bits = 0
    for x in s:
        bits |= (1 << (x-1))
    return bits

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Greedy Initialization using bitset
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(bits, x):
            bits |= (1 << (x-1))
    
    best_bits = bits
    best_size = bits.bit_count()
    
    # 2. Local Search with bitset operations
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        # Try to add any element
        candidates = [x for x in range(1, N + 1) if not ((bits >> (x-1)) & 1)]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Check if adding x is valid
            if _can_add_bitset(bits, x):
                bits |= (1 << (x-1))
                improved = True
                if bits.bit_count() > best_size:
                    best_bits = bits
                    best_size = bits.bit_count()
                break # Restart loop with new set
                
            # If not valid, find conflicts
            conflicts = _get_conflicts_bitset(bits, x)
            if not conflicts:
                continue
                
            # Try to resolve by removing one element from one of the conflicts
            # We pick a random conflict
            ap = random.choice(conflicts)
            # The element x is in the AP. We want to remove one of the OTHER elements.
            others = [y for y in ap if y != x]
            random.shuffle(others)
            
            for y in others:
                # Tentatively remove y and add x
                new_bits = (bits & ~(1 << (y-1))) | (1 << (x-1))
                if _is_4ap_free_bitset(new_bits):
                    bits = new_bits
                    improved = True
                    if bits.bit_count() > best_size:
                        best_bits = bits
                        best_size = bits.bit_count()
                    break # Restart loop
                    
            if improved:
                break
                
        # If no simple add/swap improved the set, try a "Remove 1, Add 2" move
        if not improved and time.time() < deadline:
            # Pick a random element to remove
            if best_size > 0:
                # Find all elements in the set
                elements = _bits_to_set(bits)
                if elements:
                    y = random.choice(list(elements))
                    temp_bits = bits & ~(1 << (y-1))
                    
                    # Try to add as many elements as possible to temp_bits
                    added = 0
                    new_bits = temp_bits
                    candidates = [x for x in range(1, N + 1) if not ((new_bits >> (x-1)) & 1)]
                    random.shuffle(candidates)
                    
                    for x in candidates:
                        if time.time() > deadline:
                            break
                        if _can_add_bitset(new_bits, x):
                            new_bits |= (1 << (x-1))
                            added += 1
                            if added >= 2:
                                # We have a better set
                                bits = new_bits
                                improved = True
                                if bits.bit_count() > best_size:
                                    best_bits = bits
                                    best_size = bits.bit_count()
                                break

    # Return the best set found
    return sorted(_bits_to_set(best_bits))
# EVOLVE-BLOCK-END
