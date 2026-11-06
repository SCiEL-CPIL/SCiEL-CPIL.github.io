"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math

    start_time = time.time()
    TIME_LIMIT = 110

    # Use Hash Map approach to allow larger N_MAX
    # K ~ sqrt(N_MAX). K^2 ~ N_MAX.
    # N_MAX = 3 * 10^8 -> K ~ 17000 -> K^2/2 ~ 1.5 * 10^8 ops.
    # Feasible in Python within 110s.
    N_MAX = 300_000_000

    # Generate powerful numbers up to N_MAX
    pw_nums = []
    
    max_b = int(N_MAX ** (1/3)) + 2
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_square_free = True
        d = 2
        temp_b = b
        while d * d <= temp_b:
            if temp_b % (d * d) == 0:
                is_square_free = False
                break
            d += 1
        if is_square_free:
            b3 = b * b * b
            max_a = math.isqrt(N_MAX // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= N_MAX:
                    pw_nums.append(n)
        
        if time.time() - start_time > TIME_LIMIT - 60:
            break

    # Sort to ensure a <= b in pairs and optimize iteration
    pw_nums.sort()
    
    # Count sums using a dictionary
    counts = {}
    pw_len = len(pw_nums)
    
    # Local variables for speed
    c_dict = counts
    p_list = pw_nums
    
    for i in range(pw_len):
        a = p_list[i]
        for j in range(i, pw_len):
            b = p_list[j]
            s = a + b
            c_dict[s] = c_dict.get(s, 0) + 1
        
        if time.time() - start_time > TIME_LIMIT - 30:
            break

    # Find N with maximum count
    best_N = 0
    max_count = 0
    for s, cnt in counts.items():
        if cnt > max_count:
            max_count = cnt
            best_N = s
    
    # Retrieve pairs for best_N
    pairs = []
    pw_set = set(pw_nums)
    for a in pw_nums:
        if a > best_N - a:
            break
        b_val = best_N - a
        if b_val in pw_set:
            pairs.append([a, b_val])
            
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
