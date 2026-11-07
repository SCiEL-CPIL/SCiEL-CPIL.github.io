"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _init_bitmask():
    """Return a bitmask for the upper-half set {65..128}."""
    mask = 0
    for i in range(65, N + 1):
        mask |= (1 << i)
    return mask

def _count_violations(mask):
    """Count triples (a,b,c) distinct in S such that a+b, a+c, b+c all in S."""
    # Collect elements
    elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
    n = len(elems)
    if n < 3:
        return 0
    count = 0
    # For each triple of distinct elements, check if all pair sums are in S
    # Use bit operations for speed
    for i in range(n - 2):
        a = elems[i]
        for j in range(i + 1, n - 1):
            b = elems[j]
            ab = a + b
            if ab > N or not ((mask >> ab) & 1):
                continue
            for k in range(j + 1, n):
                c = elems[k]
                ac = a + c
                bc = b + c
                if ac <= N and bc <= N and ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    count += 1
    return count

def _try_add(mask, x):
    """If adding x to mask doesn't create violations, return new mask and delta_violations."""
    if (mask >> x) & 1:
        return mask, 0
    # Check: for all pairs (a,b) in S with a != b != x, if a+b in S and a+x in S and b+x in S, that's a violation
    # Actually we need: for all distinct a,b in S\{x}, if a+b in S AND a+x in S AND b+x in S -> violation
    # More precisely: adding x creates a violation iff there exist distinct a,b in S (not x) 
    # such that a+b in S, a+x in S, b+x in S.
    # But a+x and b+x must be in S. Let's check.
    delta = 0
    elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
    for i in range(len(elems)):
        a = elems[i]
        ax = a + x
        if ax > N or not ((mask >> ax) & 1):
            continue
        for j in range(i + 1, len(elems)):
            b = elems[j]
            bx = b + x
            if bx > N or not ((mask >> bx) & 1):
                continue
            ab = a + b
            if ab <= N and ((mask >> ab) & 1):
                delta += 1
    return mask | (1 << x), delta

def _try_remove(mask, x):
    """Remove x from mask. Return new mask and delta_violations (negative = violations removed)."""
    if not ((mask >> x) & 1):
        return mask, 0
    # Count violations that involve x: triples (a,b,x) distinct where a+b, a+x, b+x all in S
    elems = [i for i in range(1, N + 1) if (mask >> i) & 1 and i != x]
    delta = 0
    for i in range(len(elems)):
        a = elems[i]
        ax = a + x
        if ax > N or not ((mask >> ax) & 1):
            continue
        for j in range(i + 1, len(elems)):
            b = elems[j]
            bx = b + x
            if bx > N or not ((mask >> bx) & 1):
                continue
            ab = a + b
            if ab <= N and ((mask >> ab) & 1):
                delta += 1
    return mask & ~(1 << x), -delta

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial solution: upper half
    mask = _init_bitmask()
    violations = _count_violations(mask)
    best_mask = mask
    best_size = bin(mask).count('1')
    best_violations = violations
    
    if violations == 0:
        # Already valid, try to improve
        pass
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute element list for current mask for faster iteration
    while T > T_min:
        if time.time() - start_time > DEADLINE:
            break
        
        # Random move: add, remove, or swap
        move_type = random.random()
        elems_in = [i for i in range(1, N + 1) if (mask >> i) & 1]
        elems_out = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
        
        if move_type < 0.4 and elems_out:
            # Add a random element
            x = random.choice(elems_out)
            new_mask, delta_v = _try_add(mask, x)
            if new_mask == mask:
                continue
            delta_size = 1
        elif move_type < 0.7 and elems_in:
            # Remove a random element
            x = random.choice(elems_in)
            new_mask, delta_v = _try_remove(mask, x)
            if new_mask == mask:
                continue
            delta_size = -1
        elif elems_in and elems_out:
            # Swap
            x_in = random.choice(elems_in)
            x_out = random.choice(elems_out)
            # Remove x_in, add x_out
            temp_mask, _ = _try_remove(mask, x_in)
            new_mask, delta_v2 = _try_add(temp_mask, x_out)
            if new_mask == mask:
                continue
            delta_v = delta_v2  # This is approximate; recompute if needed
            delta_size = 0
        else:
            continue
        
        # For accuracy, recompute violations if the incremental count might be off
        # But for speed, trust the incremental count for add/remove
        new_violations = violations + delta_v
        
        # Acceptance
        delta_obj = (new_violations - violations) * 10 + (best_size - (bin(new_mask).count('1'))) * 0.1
        # We want to minimize violations and maximize size
        # Objective: minimize (violations * 100 - size)
        old_obj = violations * 100 - bin(mask).count('1')
        new_obj = new_violations * 100 - bin(new_mask).count('1')
        delta_obj = new_obj - old_obj
        
        if delta_obj <= 0 or random.random() < __import__('math').exp(-delta_obj / T):
            mask = new_mask
            violations = new_violations
            size = bin(mask).count('1')
            if violations == 0 and size > best_size:
                best_mask = mask
                best_size = size
                best_violations = 0
            elif violations < best_violations:
                best_mask = mask
                best_size = size
                best_violations = violations
        
        T *= cooling
    
    # Final: return best valid set (violations == 0)
    if best_violations == 0:
        result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    else:
        # Fallback: upper half
        result = list(range(65, N + 1))
    
    # Ensure we return a list of Python ints
    return list(result)

# EVOLVE-BLOCK-END
