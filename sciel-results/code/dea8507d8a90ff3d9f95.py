def construct():
    t0 = time.time()
    deadline = t0 + 110
    
    best_result = None
    best_count = 0
    
    # Try increasing limits
    # L=10^9 is fast. L=10^10 is slower.
    for L in [10**9, 10**10]:
        if time.time() > deadline - 20:
            break
            
        powerful = _generate_powerful(L)
        powerful_list = sorted(powerful)
        n_pow = len(powerful_list)
        
        # Estimate time
        # If n_pow is too large, skip or use sampling?
        # For L=10^10, n_pow ~ 15000. Pairs ~ 112M.
        # 112M iterations in pure Python is risky (might take 10-20s).
        # We have 110s.
        
        sum_counts = {}
        # Use local vars for speed
        pc = sum_counts.get
        for i in range(n_pow):
            a = powerful_list[i]
            for j in range(i, n_pow):
                b = powerful_list[j]
                s = a + b
                sum_counts[s] = pc(s, 0) + 1
        
        if sum_counts:
            N_best = max(sum_counts, key=sum_counts.get)
            count_best = sum_counts[N_best]
            
            if count_best > best_count:
                best_count = count_best
                powerful_set = powerful
                pairs = []
                for a in powerful_list:
                    if 2 * a > N_best:
                        break
                    b = N_best - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': N_best, 'pairs': pairs}
                
    if best_result is None:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return best_result
