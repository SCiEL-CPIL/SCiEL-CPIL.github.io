import time

def construct():
    n = 256
    start_time = time.time()
    timeout = 80.0
    
    def time_up():
        return time.time() - start_time > timeout

    # Start with the upper half
    S = set(range(128, n + 1))
    
    # Repair loop
    max_iterations = 2000
    while max_iterations > 0 and not time_up():
        valid = True
        viol = None
        # Find a violation
        for a in S:
            min_sum = 2 * (a + 1)
            max_sum = 2 * n
            M = a * 2
            while M <= max_sum:
                if M >= min_sum:
                    # Check for b, c
                    # b in S, b > a, b <= M//2
                    # c = M - b in S, c >= b
                    for b in S:
                        if b <= a:
                            continue
                        if b > M // 2:
                            continue
                        c = M - b
                        if c in S and c >= b:
                            valid = False
                            viol = (a, b, c)
                            break
                if not valid:
                    break
                M += a
            if not valid:
                break
        
        if not valid:
            a, b, c = viol
            # Remove c
            S.remove(c)
            max_iterations -= 1
        else:
            break
            
    # Expansion
    if not time_up():
        candidates = list(range(1, 128))
        for x in reversed(candidates):
            if time_up():
                break
            if x in S:
                continue
            
            safe = True
            
            # 1. x as 'a'
            min_sum = 2 * (x + 1)
            max_sum = 2 * n
            M = x * 2
            while M <= max_sum:
                if M >= min_sum:
                    for b in S:
                        if b <= x:
                            continue
                        if b > M // 2:
                            continue
                        c = M - b
                        if c in S and c >= b:
                            safe = False
                            break
                if not safe:
                    break
                M += x
                
            if not safe:
                continue
                
            # 2. x as 'b' or 'c'
            for a in S:
                if a >= x:
                    continue
                for c in S:
                    if c < x:
                        continue
                    if (x + c) % a == 0:
                        safe = False
                        break
                if not safe:
                    break
                    
            if safe:
                S.add(x)
                
    return list(S)
