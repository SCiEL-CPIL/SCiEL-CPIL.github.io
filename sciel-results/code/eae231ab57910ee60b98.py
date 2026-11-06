def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit

    target = 123457
    max_n = target - 1

    # Sieve for largest prime factor
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            for j in range(i, max_n + 1, i):
                lpf[j] = i

    half_target = target // 2
    best_max = float('inf')
    best_pair = (1, target - 1)

    # Full exhaustive search - N is small enough (~61k iterations)
    for a in range(1, half_target + 1):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_pair = (a, b)
            if best_max <= 2:
                break
        # Time guard with periodic check
        if (a & 0x3FFF) == 0 and time.time() > deadline:
            break

    return [best_pair[0], best_pair[1]]
