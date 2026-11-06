import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    forbidden = {1, MOD - 1}

    best_len = 0
    best_start = 1

    # Forward pass
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        if e in forbidden:
            s = max(s, e + 1)
            continue
            
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Backward pass (cross-check)
    e = N
    for s in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        if s in forbidden:
            e = min(e, s - 1)
            continue
            
        ie = inv[s]
        # If inv[s] is in [s, e], we must shrink e
        if ie > s and ie <= e:
            e = ie - 1
            
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    if best_len == 0:
        return [2, 1]
        
    # Final verification
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        for x in range(start, end + 1):
            if (x * x) % MOD == 1:
                return False
            ix = inv[x]
            if ix != x and start <= ix <= end:
                return False
        return True

    if not is_valid(best_start, best_len):
        # Fallback to exhaustive search if verification fails (should not happen)
        for s2 in range(1, N + 1):
            if time.time() > deadline:
                break
            if s2 in forbidden:
                continue
            for e2 in range(s2, N + 1):
                if e2 in forbidden:
                    break
                if not is_valid(s2, e2 - s2 + 1):
                    break
                if e2 - s2 + 1 > best_len:
                    best_len = e2 - s2 + 1
                    best_start = s2

    return [best_start, best_len]
