import time
import random
import math

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    # Disjoint if (r2 - r1) is NOT divisible by g
    return (r2 - r1) % g != 0

def _greedy_assign(moduli, time_limit):
    """
    Attempts to assign residues to the given moduli such that all pairs are disjoint.
    Returns list of [residue, modulus] if successful, None otherwise.
    """
    n = len(moduli)
    # Sort moduli? Let's try keeping them as is or randomizing.
    # Randomizing order can help escape local minima in greedy.
    indices = list(range(n))
    random.shuffle(indices)
    
    assigned = {} # m -> r
    # We need to check against all previously assigned.
    
    # To speed up, we can pre-compute gcds?
    # n is 64. n^2 is 4096. Not an issue.
    
    for idx in indices:
        m = moduli[idx]
        found = False
        # Try residues in random order? Or 0..m-1?
        # Random order might be better to avoid systematic conflicts.
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            ok = True
            for m2, r2 in assigned.items():
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                assigned[m] = r
                found = True
                break
        
        if not found:
            return None
            
    # Convert to list of pairs
    result = []
    for m in moduli:
        result.append([assigned[m], m])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # The set of all even numbers from 2 to 128
    # This forms a clique in the "non-coprime" graph
    moduli = list(range(2, MAX_MOD + 1, 2))
    
    # Try to find a valid assignment for all 64 moduli
    # Use multiple restarts
    best_result = None
    attempts = 0
    max_attempts = 10000 # Safety cap
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        res = _greedy_assign(moduli, deadline)
        if res is not None:
            best_result = res
            break
        
        # If we fail, we might try a smaller set?
        # But we want to maximize.
        # If 64 is impossible (unlikely), we could drop one.
        # But let's assume 64 is possible.
        
    if best_result is None:
        # Fallback: try a smaller set, e.g., all multiples of 4?
        # Or just return a valid small set to ensure non-zero score.
        # The seed returns 7. We should try to beat that.
        # Let's try a greedy on a subset if 64 fails.
        # But 64 is very likely to work with random restarts.
        
        # Emergency fallback: return the seed solution or a simple valid one
        # Let's construct a valid one for sure:
        # Moduli: 2, 4, 8, 16, 32, 64, 128
        # Residues: 1, 1, 1, 1, 1, 1, 1? No, conflicts.
        # Let's just return the seed's solution if we fail completely.
        # But the seed is [[(1<<k)-1,1<<(k+1)] for k in range(7)]
        # Which is [[1,2], [3,4], [7,8], [15,16], [31,32], [63,64], [127,128]]
        # This is valid.
        best_result = [[(1<<k)-1,1<<(k+1)] for k in range(7)]
        
    return best_result
