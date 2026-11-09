"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256

# Precompute: for each x in [1, N], which other elements would form a 3-AP with it
# conflict[x] = set of elements y such that x and y are part of some 3-AP with a third element
# More precisely: if we have set S and want to add x, conflict if exists a,b in S with {a,b,x} containing a 3-AP
# For efficiency, precompute for each x, the set of "partners" that would conflict

def precompute_conflicts():
    """For each x, compute which pairs (a,b) in [1,N] would form a 3-AP with x.
    Actually, for adding x to a set S, we need: no a,b in S such that {a,b,x} has a 3-AP.
    This means: for each a in S, b = 2x-a must not be in S (if b != x and b in [1,N])
    Also: for each a in S, if (a+x)/2 is integer and in S, conflict.
    Also: for each a in S, if 2a-x is in S and 2a-x != x, conflict.
    
    Better: precompute for each x, the set of elements that CANNOT coexist with x in a 3-AP-free set.
    Two elements x,y cannot both be in a 3-AP-free set if there exists z such that {x,y,z} is a 3-AP.
    But that's not quite right either - x and y CAN coexist if no z completes the AP.
    
    For the greedy/add check: when adding x to set S, check if any a in S has:
    - 2x - a in S (and 2x-a != x)  => x is the middle
    - (x+a)/2 in S (integer, != x, != a) => x is an end
    - 2a - x in S (and 2a-x != x, 2a-x != a) => a is the middle
    
    Let's just precompute for each x, the list of a values such that adding x conflicts if a is in S.
    """
    # For each x, compute the set of a values where {a, x, ?} or {?, a, x} forms AP
    # Actually simpler: for each x, and each potential a in [1,N], determine if {a, x} "conflicts"
    # meaning there exists z in [1,N] such that {a, x, z} is a 3-AP.
    # But we need z to also be in S.
    
    # For the add-check: adding x to S is invalid iff exists a in S such that:
    # 1) z = 2*x - a is in S and z != a and z != x  (x is middle)
    # 2) z = (x + a) / 2 is integer, in S, z != a, z != x  (x is end, a is other end)
    # 3) z = 2*a - x is in S and z != a and z != x  (a is middle)
    
    # Precompute: for each x, for each a, what z values would complete a 3-AP with x and a
    # Then when checking if x can be added to S, for each a in S, check if any such z is in S.
    
    # To make this fast, let's precompute for each x, a list of (a, z) pairs
    # where {a, x, z} is a 3-AP. When adding x, for each a in S, check if z is in S.
    
    ap_pairs = {}  # ap_pairs[x] = list of (a, z) such that {a,x,z} is a 3-AP
    for x in range(1, N+1):
        pairs = []
        for a in range(1, N+1):
            if a == x:
                continue
            # Case 1: x is middle: a, x, z => z = 2x - a
            z = 2*x - a
            if 1 <= z <= N and z != a and z != x:
                pairs.append((a, z))
            # Case 2: x is end (smaller): x, a, z => z = 2a - x, need x < a < z
            # Actually x, a, z with a-x = z-a => z = 2a - x, need z > a > x
            z = 2*a - x
            if 1 <= z <= N and z != a and z != x and z > a > x:
                pairs.append((a, z))
            # Case 3: x is end (larger): a, z, x => z = (a+x)/2, need a < z < x
            if (a + x) % 2 == 0:
                z = (a + x) // 2
                if 1 <= z <= N and z != a and z != x and a < z < x:
                    pairs.append((a, z))
        ap_pairs[x] = pairs
    return ap_pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    ap_pairs = precompute_conflicts()
    
    # Greedy construction: process numbers in order of decreasing 1/x (smallest first)
    # Add x if it doesn't create a 3-AP
    selected_set = set()
    
    for x in range(1, N + 1):
        # Check if adding x creates any 3-AP
        can_add = True
        for a, z in ap_pairs[x]:
            if a in selected_set and z in selected_set:
                can_add = False
                break
        if can_add:
            selected_set.add(x)
    
    if time.time() > deadline:
        return list(selected_set)
    
    best_set = selected_set.copy()
    best_sum = sum(Fraction(1, x) for x in best_set)
    
    # Local search: try swaps to improve reciprocal sum
    # Precompute reciprocal values as floats for quick comparison, use Fraction only for final
    recip = {x: Fraction(1, x) for x in range(1, N+1)}
    recip_float = {x: 1.0 / x for x in range(1, N+1)}
    
    improved = True
    max_iterations = 50000  # safety cap
    iteration = 0
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        unselected = [x for x in range(1, N+1) if x not in best_set]
        
        for rem in list(best_set):
            if time.time() > deadline:
                break
            
            # Try removing rem and adding each unselected element
            trial_set = best_set - {rem}
            
            for add in unselected:
                # Check if add can be added to trial_set
                can_add = True
                for a, z in ap_pairs[add]:
                    if a in trial_set and z in trial_set:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                
                # Compute new sum
                new_sum = best_sum - recip[rem] + recip[add]
                if new_sum > best_sum:
                    best_set = trial_set | {add}
                    best_sum = new_sum
                    improved = True
                    break
            
            if improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
