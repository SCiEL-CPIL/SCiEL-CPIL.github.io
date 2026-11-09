import time

def _is_valid(a):
    n = len(a)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += a[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _greedy_construct(n=12):
    a = [1]
    for i in range(1, n):
        val = a[-1]
        while True:
            val += 1
            a.append(val)
            if _is_valid(a):
                break
            a.pop()
    return a

def _reduce_element(a, idx):
    old = a[idx]
    for v in range(1, old):
        if v in a:
            continue
        a[idx] = v
        if _is_valid(a):
            return True
    a[idx] = old
    return False

def construct():
    deadline = time.time() + 85.0
    a = _greedy_construct(12)
    
    # Try to reduce each element
    for _ in range(100):  # Multiple passes
        if time.time() > deadline - 5.0:
            break
        any_reduced = False
        for i in range(12):
            if time.time() > deadline - 5.0:
                break
            if _reduce_element(a, i):
                any_reduced = True
                a.sort()
        if not any_reduced:
            break
    
    if not _is_valid(a):
        a = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return a
