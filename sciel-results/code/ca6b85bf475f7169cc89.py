import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    best_val = float('inf')
    best_assign = None
    
    # Try all windows [L, L+W] for small W
    # L can range from 1 to MAX_V
    # For each window, check if each divisor has a multiple in it
    # and assign distinct multiples
    
    # First, find minimum possible W
    # For a window to work, each divisor d must have at least one multiple in [L, L+W]
    # The "tightest" divisors are the largest ones
    
    # Let's do a smarter search: try W from 1 upward
    for W in range(1, 200):
        if time.time() > deadline:
            break
        # For each possible L, check feasibility
        # L ranges from 1 to MAX_V - W
        # But we can be smarter: L mod lcm matters
        # Actually, let's just try a range of L values
        
        found = False
        # Try L from 1 to some limit, stepping
        for L in range(1, min(MAX_V - W + 1, 50000)):
            if time.time() > deadline:
                break
            # For each divisor, find if there's a multiple in [L, L+W]
            assign = []
            valid = True
            used = set()
            for d in divisors:
                # Find multiples of d in [L, L+W]
                # First multiple >= L: ceil(L/d)*d
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                candidates = [start_k * d + i * d for i in range(end_k - start_k + 1)]
                # Filter out used
                candidates = [c for c in candidates if c not in used]
                if not candidates:
                    valid = False
                    break
                # Pick the first available (greedy)
                chosen = candidates[0]
                assign.append(chosen)
                used.add(chosen)
            if valid:
                # Check all distinct (already ensured by used set)
                val = max(assign) - min(assign)
                if val < best_val:
                    best_val = val
                    best_assign = assign[:]
                found = True
                break  # Found one for this W, move to next W
        if found:
            break
    
    if best_assign is None:
        # Fallback to seed
        return divisors[:]
    
    return [int(x) for x in best_assign]
