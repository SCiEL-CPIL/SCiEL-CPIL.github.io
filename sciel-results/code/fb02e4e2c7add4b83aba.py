def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = 85.0
    
    denominators = list(range(2, 49))
    n = 47
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    target = L
    
    mid = n // 2  # 23
    weights_A = weights[:mid]  # 23 items
    weights_B = weights[mid:]  # 24 items
    
    # Enumerate (sum, mask) for A
    entries_A = [(0, 0)]
    for i, w in enumerate(weights_A):
        bit = 1 << i
        entries_A += [(s + w, m | bit) for s, m in entries_A]
    
    entries_A.sort(key=lambda x: x[0])
    
    # Enumerate (sum, mask) for B
    entries_B = [(0, 0)]
    for i, w in enumerate(weights_B):
        bit = 1 << i
        entries_B += [(s + w, m | bit) for s, m in entries_B]
    
    entries_B.sort(key=lambda x: x[0])
    
    # Two-pointer scan
    pA = len(entries_A) - 1
    best_sum = -1
    best_mask_A = 0
    best_mask_B = 0
    
    for sb, mb in entries_B:
        if time.time() > deadline - 5:
            break
        limit = target - sb
        if limit <= 0:
            break
        while pA >= 0 and entries_A[pA][0] >= limit:
            pA -= 1
        if pA >= 0:
            sa, ma = entries_A[pA]
            total = sa + sb
            if total > best_sum:
                best_sum = total
                best_mask_A = ma
                best_mask_B = mb
    
    # Reconstruct indices
    result = []
    for i in range(mid):
        if best_mask_A & (1 << i):
            result.append(i)
    for i in range(len(weights_B)):
        if best_mask_B & (1 << i):
            result.append(mid + i)
    
    if not result:
        # Fallback: pick the single largest weight < target
        for i in range(n):
            if weights[i] < target:
                result.append(i)
                break
    
    assert len(result) > 0
    assert sum(weights[i] for i in result) < target
    
    return result
