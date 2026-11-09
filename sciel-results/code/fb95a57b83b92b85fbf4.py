"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 123457
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Step 1: Compute largest prime factor (LPF) for all integers 1..N
    # Using a modified sieve
    lpf = list(range(N + 1))
    lpf[0] = 0
    lpf[1] = 1

    for i in range(2, int(N**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                lpf[j] = i

    # For numbers that are still equal to themselves, they are prime
    # (or 1, but we already set lpf[1]=1)
    # Actually the above loop only sets lpf[j] for composite j.
    # For primes, lpf[prime] remains prime itself, which is correct.
    # But wait - the loop starts at i*i and goes up by i.
    # This correctly sets lpf[j] to the smallest prime factor for each composite j.
    # But we want the LARGEST prime factor. Let me reconsider.

    # The above actually gives the SMALLEST prime factor, not the largest.
    # We need to compute the largest prime factor.

    # Better approach: factor each number and track its largest prime factor.
    # Or use a different sieve technique.

    # Let's use a proper approach for largest prime factor:
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # keep updating to the largest prime factor

    # Now lpf[x] is the largest prime factor of x for x >= 2, and lpf[1] = 1.

    # Step 2: Find pair (a, N-a) that minimizes max(lpf[a], lpf[N-a])
    best_score = N  # worst case is N itself (prime)
    best_pair = (1, N - 1)

    for a in range(1, N // 2 + 1):
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_pair = (a, b)

        # Check time
        if a % 10000 == 0 and time.time() - start_time > DEADLINE:
            break

    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
