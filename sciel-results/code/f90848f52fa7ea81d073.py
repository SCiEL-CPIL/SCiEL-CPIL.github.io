"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

TARGET = 123457
DEADLINE = 85.0  # safety margin before 90s kill

def _largest_prime_factor(n):
    """Return the largest prime factor of n. P(1)=1."""
    if n <= 1:
        return 1
    largest = 1
    # Factor out 2s
    while n % 2 == 0:
        largest = 2
        n //= 2
    # Factor out odd primes
    f = 3
    while f * f <= n:
        while n % f == 0:
            largest = f
            n //= f
        f += 2
    if n > 1:
        largest = n
    return largest

def _generate_smooth_numbers(primes, limit):
    """Generate all numbers <= limit composed only of the given primes."""
    smooth = {1}
    for p in primes:
        new_smooth = set()
        for s in smooth:
            val = s
            while val * p <= limit:
                val *= p
                new_smooth.add(val)
        smooth.update(new_smooth)
    return smooth

def construct():
    start_time = time.time()
    
    # The primes in increasing order
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    best_pair = (1, TARGET - 1)  # fallback: [1, 123456]
    best_max_pf = _largest_prime_factor(TARGET - 1)  # 643
    
    # Try increasing sets of primes
    for num_primes in range(1, len(all_primes) + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        primes = all_primes[:num_primes]
        max_p = primes[-1]
        
        # If the max prime already exceeds our best, we might still find a better pair
        # but we can't improve beyond max_p for this set
        # Actually, we want to find the smallest max_p, so we stop when we find a solution
        # But we should keep going to find even smaller ones... wait, we're going in increasing order
        # So the first solution we find with a given num_primes has max_p = primes[-1]
        # But there might be a solution with fewer primes that we missed? No, we check all.
        
        # Generate smooth numbers
        smooth = _generate_smooth_numbers(primes, TARGET)
        
        # Check if any pair sums to TARGET
        found = False
        for a in smooth:
            b = TARGET - a
            if 1 <= b <= TARGET and b in smooth:
                # Both a and b are smooth with these primes
                # The actual largest prime factors might be smaller than max_p
                pa = _largest_prime_factor(a)
                pb = _largest_prime_factor(b)
                max_pf = max(pa, pb)
                if max_pf < best_max_pf:
                    best_max_pf = max_pf
                    best_pair = (a, b)
                    found = True
                # Once we find a solution, we can break since we're going in order
                # But we want the minimum max_pf, so we should check all
                # Actually, for a fixed set of primes, the max possible is primes[-1]
                # So if we found one, the best we can do with this set is at most primes[-1]
                # But we should continue to find if there's one with smaller actual max_pf
        
        # If we found a solution and the max prime in our set is >= best_max_pf,
        # we might still find better with more primes, but the max_pf won't decrease below
        # what we already have if we use more primes... actually it could, because
        # a number composed of {2,3,5} might have P=5, but a number composed of {2,3,5,7}
        # might also have P=5. So we should keep searching.
        
        # Optimization: if max_p >= best_max_pf and we've already found a solution,
        # adding more primes won't help because any new number will have P >= some prime
        # in the extended set, but it could still have P <= best_max_pf if it doesn't use
        # the new primes. So we can't easily prune.
        
        # However, if max_p < best_max_pf and we found NO solution, we must continue.
        # If max_p >= best_max_pf and we found a solution, we can potentially stop
        # only if we're confident no better solution exists. But it's safer to continue.
        
        # Let's add a heuristic: if we've found a solution with max_pf = m, and our current
        # max_p >= m, we can stop because any future solution will have max_pf >= some prime
        # >= current max_p >= m. Wait, that's not right. A number using primes up to 11
        # could have P=7 (if it doesn't use 11). So we need to be careful.
        
        # Actually, the key insight: if we're using primes up to p_k, any smooth number
        # has P <= p_k. So if we find a solution with actual max_pf = m <= p_k, and
        # m < best_max_pf, we update. We can stop when p_k >= best_max_pf AND we've
        # checked all smooth numbers for this p_k. But we can't stop early within the loop
        # easily.
        
        # For now, let's just continue until deadline or until we're confident.
        
        # Heuristic stop: if best_max_pf <= max_p and we found at least one solution,
        # and max_p is already small (like <= 11), we can probably stop.
        # But to be safe, let's continue a bit more.
        
        # Actually, let's think about it differently. The optimal answer is the minimum
        # over all valid pairs of max(P(a),P(b)). We're enumerating by increasing prime set.
        # The first time we find ANY valid pair, its max_pf gives an upper bound. As we
        # add more primes, we can find pairs with potentially smaller max_pf (if the
        # numbers don't use the new large primes). But we can also find pairs with larger
        # max_pf. So the optimal is the minimum over all pairs found across all prime sets.
        
        # Since the number of smooth numbers grows rapidly, let's limit our search.
        
    # After the loop, return the best pair found
    a, b = best_pair
    return [a, b]

# EVOLVE-BLOCK-END
