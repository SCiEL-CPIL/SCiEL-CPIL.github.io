import time

def _square_free_part(n):
    """Compute the square-free part of n."""
    result = 1
    d = 2
    x = n
    while d * d <= x:
        count = 0
        while x % d == 0:
            count += 1
            x //= d
        if count % 2 == 1:
            result *= d
        d += 1
    if x > 1:
        result *= x
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = 210
    
    # First pass: find smallest M
    achievable = {0}
    M = 211
    while True:
        if time.time() - start_time > deadline:
            break
        sf = _square_free_part(M)
        new_achievable = set(achievable)
        for a in achievable:
            new_achievable.add(a ^ sf)
        achievable = new_achievable
        if target in achievable:
            break
        M += 1
    
    # Second pass: find subset via parent tracking
    achievable = {0}
    parent = {}
    for num in range(211, M + 1):
        sf = _square_free_part(num)
        new_vals = set()
        for a in achievable:
            nv = a ^ sf
            if nv not in achievable:
                new_vals.add(nv)
                parent[nv] = (num, a)
        achievable |= new_vals
        if target in achievable:
            break
    
    # Backtrack
    subset = []
    cur = target
    while cur != 0:
        num, prev = parent[cur]
        subset.append(num)
        cur = prev
    
    subset.reverse()
    result = [int(x) for x in subset]
    return result
