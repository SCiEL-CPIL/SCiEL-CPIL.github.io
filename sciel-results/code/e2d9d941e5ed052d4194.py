def construct():
    n = 1008
    p = 1009
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p-2, p)
    
    best_start = 1
    best_len = 0
    
    for a in range(1, n+1):
        if a == 1 or a == n:
            continue
        b = a
        valid = True
        while b < n:
            b += 1
            if b == n:  # 1008
                valid = False
                break
            if a <= inv[b] <= b-1:
                valid = False
                break
        if valid:
            length = b - a + 1
        else:
            # The last valid b was b-1
            if b > a:
                length = b - a
            else:
                continue  # a itself is invalid, but we already skipped 1 and 1008
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
