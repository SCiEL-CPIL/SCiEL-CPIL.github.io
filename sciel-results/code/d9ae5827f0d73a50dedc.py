import time

def construct():
    start_time = time.time()
    MOD = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, -1, MOD)
    
    best_len = 0
    best_start = 1
    s = 1
    
    # Sliding window: for each end e, maintain the smallest valid start s
    # such that [s, e] has no inverse pair
    for e in range(1, MAX_VAL + 1):
        partner = inv[e]
        
        # If e is self-inverse, it can't be in any valid interval
        if partner == e:
            s = e + 1
            continue
        
        # If partner is in the current window [s, e-1], move s past partner
        if s <= partner < e:
            s = partner + 1
        
        # Now [s, e] should be valid
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    # Verify the result
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break
    
    if not valid:
        # Fallback: brute force search for the best valid interval
        for start in range(1, MAX_VAL + 1):
            for end in range(MAX_VAL, start - 1, -1):
                ok = True
                for x in range(start, end + 1):
                    if start <= inv[x] <= end:
                        ok = False
                        break
                if ok:
                    if end - start + 1 > best_len:
                        best_len = end - start + 1
                        best_start = start
                    break
            # Early exit if remaining can't beat best
            if (MAX_VAL - start + 1) <= best_len:
                break
    
    return [best_start, best_len]
