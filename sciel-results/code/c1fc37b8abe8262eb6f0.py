import time
import random
from collections import defaultdict

N = 512

def _precompute_conflicts():
    """
    Precompute for each x, the set of y values that would form a 4-AP with x.
    conflict[x] = set of y such that {x, y} is part of some 4-AP.
    Also store the specific APs for each pair (x, y) to check if removing y resolves conflicts.
    """
    # For each x, find all y such that x and y can be part of the same 4-AP
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # For a given x, it can appear in positions 1, 2, 3, or 4 of the AP
    
    conflicts = defaultdict(set)  # conflicts[x] = set of y that conflict with x
    aps_for_pair = defaultdict(list)  # aps_for_pair[(x,y)] = list of 4-APs containing both x and y
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap = [a, a + d, a + 2*d, a + 3*d]
            # All pairs in this AP conflict with each other
            for i in range(4):
                for j in range(i + 1, 4):
                    x, y = ap[i], ap[j]
                    conflicts[x].add(y)
                    conflicts[y].add(x)
                    aps_for_pair[(x, y)].append(ap)
    
    return conflicts, aps_for_pair

# Precompute at module level
_CONFLICTS, _APS_FOR_PAIR = _precompute_conflicts()

def _can_add_bitset(bits, x):
    """
    Check if adding x to the set represented by bits (integer bitset)
    creates a 4-term arithmetic progression.
    bits has bit i set if value i is in the set (1-indexed).
    """
    # For each y in conflicts[x], check if all other members of any AP containing (x,y) are present
    # But we need to be smarter: we need to check if there exists a 4-AP containing x and three other elements all in the set
    
    # Direct approach: for each possible 4-AP containing x, check if the other 3 are in the set
    # x as 4th: a, a+d, a+2d, x => a = x-3d, need a, a+d, a+2d in set
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3*d
        if (bits >> a) & 1 and (bits >> (a + d)) & 1 and (bits >> (a + 2*d)) & 1:
            return False
    
    # x as 3rd: a, a+d, x, x+d => a = x-2d, need a, a+d, x+d in set
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        a = x - 2*d
        if (bits >> a) & 1 and (bits >> (a + d)) & 1 and (bits >> (x + d)) & 1:
            return False
    
    # x as 2nd: a, x, x+d, x+2d => a = x-d, need a, x+d, x+2d in set
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        a = x - d
        if (bits >> a) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1:
            return False
    
    # x as 1st: x, x+d, x+2d, x+3d => need x+d, x+2d, x+3d in set
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1 and (bits >> (x + 3*d)) & 1:
            return False
    
    return True

def _get_blocking_elements(bits, x):
    """
    Return the set of elements in the current set that are part of a 4-AP with x.
    These are the elements that block x from being added.
    """
    blockers = set()
    
    # x as 4th: a, a+d, a+2d, x
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3*d
        if (bits >> a) & 1 and (bits >> (a + d)) & 1 and (bits >> (a + 2*d)) & 1:
            blockers.update([a, a + d, a + 2*d])
    
    # x as 3rd: a, a+d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        a = x - 2*d
        if (bits >> a) & 1 and (bits >> (a + d)) & 1 and (bits >> (x + d)) & 1:
            blockers.update([a, a + d, x + d])
    
    # x as 2nd: a, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        a = x - d
        if (bits >> a) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1:
            blockers.update([a, x + d, x + 2*d])
    
    # x as 1st: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (bits >> (x + d)) & 1 and (bits >> (x + 2*d)) & 1 and (bits >> (x + 3*d)) & 1:
            blockers.update([x + d, x + 2*d, x + 3*d])
    
    return blockers

def _bits_to_set(bits):
    """Convert bitset to sorted list of elements."""
    return [i for i in range(1, N + 1) if (bits >> i) & 1]

def construct():
    deadline = time.time() + 55.0  # Safety margin
    best_size = 0
    best_bits = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Greedy Initialization with random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        bits = 0
        current_size = 0
        
        for x in order:
            if _can_add_bitset(bits, x):
                bits |= (1 << x)
                current_size += 1
        
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            candidates = [x for x in range(1, N + 1) if not (bits >> x) & 1]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if _can_add_bitset(bits, x):
                    bits |= (1 << x)
                    current_size += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # If no simple add, try targeted swaps
            if not candidates:
                break
                
            # Pick a random candidate x
            x = random.choice(candidates)
            
            # Find blocking elements
            blockers = _get_blocking_elements(bits, x)
            
            # Try removing one blocker to see if x can be added
            # Sort blockers by how "central" they are (more conflicts = more central)
            blocker_list = list(blockers)
            random.shuffle(blocker_list)
            
            swap_found = False
            for y in blocker_list:
                if time.time() > deadline:
                    break
                # Tentatively remove y
                bits_without_y = bits & ~(1 << y)
                
                # Check if x can now be added
                if _can_add_bitset(bits_without_y, x):
                    # Also check that removing y doesn't create new issues (it shouldn't, since we're removing)
                    bits = bits_without_y | (1 << x)
                    # current_size stays the same (removed y, added x)
                    improved = True
                    swap_found = True
                    break
            
            if not swap_found and current_size > best_size:
                best_size = current_size
                best_bits = bits
        
        if current_size > best_size:
            best_size = current_size
            best_bits = bits
            
    return _bits_to_set(best_bits)
