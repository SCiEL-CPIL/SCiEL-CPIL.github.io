import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Compute rough part: i with all small prime factors removed
    rough = list(range(MAX_VAL + 1))
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            while rough[i] % p == 0:
                rough[i] //= p

    # smooth_value[i] = i // rough[i] = the 23-smooth part of i
    smooth_value = [0] * (MAX_VAL + 1)
    for i in range(1, MAX_VAL + 1):
        smooth_value[i] = i // rough[i]

    # Sliding window search for n in [1000, MAX_N]
    # Window for n: smooth_value[n+1], ..., smooth_value[n+24]
    best_n = 1000
    best_count = 100

    # Initialize window for n=1000: indices 1001..1024
    freq = {}
    for i in range(1001, 1025):
        v = smooth_value[i]
        freq[v] = freq.get(v, 0) + 1
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000

    for n in range(1001, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        # Transition from n-1 to n: remove smooth_value[n], add smooth_value[n+24]
        out_val = smooth_value[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            distinct -= 1

        in_val = smooth_value[n + 24]
        if in_val in freq:
            freq[in_val] += 1
        else:
            freq[in_val] = 1
            distinct += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n

    return best_n
