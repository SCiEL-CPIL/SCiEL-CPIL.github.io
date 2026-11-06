import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Identify self-inverse elements
    # x * x % p == 1
    # Solutions to x^2 = 1 mod p are x=1, x=p-1
    self_inv = {1, p - 1}
    
    # Build pairs
    # We want pairs {a, b} with a < b such that inv[a] == b and a, b not in self_inv
    # We will create a list of (a, b)
    pairs = []
    visited = set()
    for x in range(2, p - 1):
        if x in visited:
            continue
        y = inv[x]
        if y == x:
            # Should not happen for x in 2..p-2 if p is prime > 2
            # But just in case
            continue
        a = min(x, y)
        b = max(x, y)
        if a not in visited and b not in visited:
            pairs.append((a, b))
            visited.add(a)
            visited.add(b)
            
    # Compute M(S) = min { b_i | a_i >= S }
    # min_b_from[i] stores min b for pairs with a >= i
    INF = float('inf')
    min_b_from = [INF] * (p + 2)
    min_b_from[p + 1] = INF
    
    # Initialize min_b_from with INF
    # We need to associate each a with the min b for that a?
    # Actually, M(S) = min( M(S+1), min{b_i | a_i == S} )
    
    # First, find min b for each a
    min_b_for_a = [INF] * (p + 1)
    for a, b in pairs:
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
            
    # Fill min_b_from from right to left
    for i in range(p, 0, -1):
        min_b_from[i] = min(min_b_from[i + 1], min_b_for_a[i])
        
    best_start = 2
    best_len = 0
    
    # Iterate over possible starts
    # S cannot be 1 or 1008
    for S in range(2, p - 1):
        if time.time() > deadline:
            break
            
        # Max E is min(M(S) - 1, 1007)
        # 1007 is p-2, ensuring 1008 (p-1) is not included
        # Also 1 is not included since S >= 2
        max_E = min(min_b_from[S] - 1, p - 2)
        
        if max_E >= S:
            length = max_E - S + 1
            if length > best_len:
                best_len = length
                best_start = S
                
    # Fallback
    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
