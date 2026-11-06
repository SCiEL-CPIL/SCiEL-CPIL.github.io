"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp

    def generate_smooth(max_prime, limit):
        """Generate all numbers <= limit with all prime factors <= max_prime,
        along with their largest prime factor."""
        primes = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
            if p <= max_prime:
                primes.append(p)
            else:
                break

        # Use recursive generation
        smooth = {}  # number -> largest prime factor

        def gen(idx, current, lpf):
            if current > limit:
                return
            if current not in smooth or smooth[current] < lpf:
                smooth[current] = lpf
            if idx >= len(primes):
                return
            p = primes[idx]
            gen(idx + 1, current, lpf)
            val = current
            while val * p <= limit:
                val *= p
                gen(idx, val, p)

        gen(0, 1, 1)
        return smooth

    best_result = [1, TARGET - 1]
    best_score = largest_prime_factor(TARGET - 1)  # P(123456) = 281

    # Try increasing smoothness bounds
    for max_p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break

        smooth = generate_smooth(max_p, TARGET)
        if not smooth:
            continue

        # For each smooth number a, check if TARGET - a is also smooth
        found_at_this_level = False
        for a, pa in smooth.items():
            if time.time() > deadline:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth:
                pb = smooth[b]
                score = max(pa, pb)
                if score < best_score:
                    best_score = score
                    best_result = [a, b]
                    found_at_this_level = True

        if found_at_this_level and best_score <= max_p:
            # We found a solution where both are max_p-smooth
            # Can we do better? Only if we find something with smaller max prime
            # But since we're going in order, if we found at this level, 
            # the next level might have a better solution
            # Actually no - a solution at level max_p has max(P(a),P(b)) <= max_p
            # A solution at level max_p+1 might have max(P(a),P(b)) = max_p+1 which is worse
            # So once we find a solution at level max_p, we should check if there's
            # a better one at the same level or lower
            # Actually, we should continue to see if a lower max_p was missed
            # But we're going in order, so we've already checked lower levels
            # The best we can find at this level is <= max_p
            # If best_score == max_p, we can't improve at this level
            # But we should stop if best_score is very small
            if best_score <= 5:
                break

    return best_result
# EVOLVE-BLOCK-END
