def construct():
    n = 10000
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n+1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    max_gap = n  # 10000
    
    dp = [{} for _ in range(num_primes)]
    prefix_max = [None] * num_primes  # prefix_max[j][g] = max length for gap <= g at prime j
    
    for i in range(num_primes):
        dp[i][0] = (1, -1, 0)
        # Build prefix max for i
        arr = [0] * (max_gap + 1)
        for g, (length, pi, pg) in dp[i].items():
            if g <= max_gap:
                arr[g] = max(arr[g], length)
        # Prefix max
        for g in range(1, max_gap + 1):
            arr[g] = max(arr[g], arr[g-1])
        prefix_max[i] = arr
    
    for i in range(1, num_primes):
        p = primes[i]
        for j in range(i):
            b = primes[j]
            gap = p - b
            if gap > max_gap:
                continue
            best_length = prefix_max[j][gap]
            if best_length > 0:
                # Find which gap in dp[j] gives this best_length and is <= gap
                # We need to find the actual gap for reconstruction
                best_prev_gap = 0
                for g, (length, pi, pg) in dp[j].items():
                    if g <= gap and length == best_length:
                        best_prev_gap = g
                        break
                new_len = best_length + 1
                if gap not in dp[i] or new_len > dp[i][gap][0]:
                    dp[i][gap] = (new_len, j, best_prev_gap)
        
        # Rebuild prefix max for i
        arr = [0] * (max_gap + 1)
        for g, (length, pi, pg) in dp[i].items():
            if g <= max_gap:
                arr[g] = max(arr[g], length)
        for g in range(1, max_gap + 1):
            arr[g] = max(arr[g], arr[g-1])
        prefix_max[i] = arr
    
    # Find best
    max_len = 0
    best_i = -1
    best_gap = -1
    for i in range(num_primes):
        for g, (length, pi, pg) in dp[i].items():
            if length > max_len:
                max_len = length
                best_i = i
                best_gap = g
    
    # Reconstruct
    seq = []
    cur_i = best_i
    cur_gap = best_gap
    while cur_i != -1:
        seq.append(primes[cur_i])
        if cur_gap == 0:
            break
        _, prev_i, prev_gap = dp[cur_i][cur_gap]
        cur_i = prev_i
        cur_gap = prev_gap
    seq.reverse()
    return seq
