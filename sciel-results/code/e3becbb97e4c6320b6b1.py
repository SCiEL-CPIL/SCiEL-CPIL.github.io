def backtrack(idx, current_sum, last_d, path, N):
    if time.time() > deadline:
        return False
    
    remaining = N - idx
    
    if remaining == 0:
        return current_sum == 1
        
    if remaining == 1:
        rem_val = 1 - current_sum
        if rem_val <= 0:
            return False
        # Check if rem_val is 1/d
        d = 1 / rem_val
        if d == int(d) and int(d) > last_d and int(d) <= 10000000:
            path.append(int(d))
            return True
        return False
        
    if remaining == 2:
        rem_val = 1 - current_sum
        if rem_val <= 0:
            return False
        # rem_val = 1/a + 1/b
        # 1/a < rem_val => a > 1/rem_val
        # 1/b = rem_val - 1/a
        # We iterate a
        min_a = last_d + 1
        max_a = int(1 / rem_val) + 1 # 1/a < rem_val
        for a in range(min_a, max_a + 1):
            if time.time() > deadline:
                return False
            rem_b = rem_val - Fraction(1, a)
            if rem_b <= 0:
                continue
            b = 1 / rem_b
            if b == int(b) and int(b) > a and int(b) <= 10000000:
                path.append(a)
                path.append(int(b))
                return True
        return False

    # General case
    # Pruning max_sum
    max_sum = current_sum
    d = last_d + 1
    for i in range(remaining):
        max_sum += Fraction(1, d)
        d += 1
    if max_sum < 1:
        return False
        
    if current_sum > 1:
        return False
        
    for d in range(last_d + 1, 10000000):
        if time.time() > deadline:
            return False
        new_sum = current_sum + Fraction(1, d)
        if new_sum > 1:
            break
        path.append(d)
        if backtrack(idx + 1, new_sum, d, path, N):
            return True
        path.pop()
    return False
