"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import product

def _is_powerful(n):
    """Check if n is a powerful number using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp == 1:
                return False
        p += 1 if p == 2 else 2
    # If temp > 1, it's a prime factor with exponent 1
    if temp > 1:
        return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    # A powerful number is of the form a^2 * b^3 where b is square-free
    # But more efficiently, we can generate by iterating over possible square-free parts
    # and square parts.
    
    # Method: iterate over b (square-free part) up to limit^(1/3)
    # and a such that a^2 * b^3 <= limit
    # Actually, let's use a different approach:
    # For each square-free b, for each a such that a^2 * b^3 <= limit
    # But b must be square-free.
    
    # Better: generate all numbers of form a^2 * b^3 where b is square-free
    # and a^2 * b^3 <= limit
    
    # First, find all square-free numbers up to limit^(1/3)
    # Actually, b can be up to limit^(1/3) since b^3 <= limit
    max_b = int(limit ** (1.0/3.0)) + 2
    
    # Generate square-free numbers up to max_b
    square_free = [True] * (max_b + 1)
    square_free[0] = False
    square_free[1] = True
    for p in range(2, max_b + 1):
        if square_free[p]:
            for k in range(p*p, max_b + 1, p*p):
                square_free[k] = False
    
    sq_free_list = [i for i in range(2, max_b + 1) if square_free[i]]
    
    # Also include b=1
    all_b = [1] + sq_free_list
    
    for b in all_b:
        b_cubed = b ** 3
        if b_cubed > limit:
            continue
        max_a = int(math.isqrt(limit // b_cubed))
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful.add(val)
    
    powerful.add(1)
    return sorted(powerful)

def construct():
    """
    Construct N with many representations as sum of two powerful numbers.
    Strategy: Use N = Q * M where Q is a highly composite square-free integer.
    Then a = x^2 * Q1, b = y^2 * Q2 where Q1*Q2 divides Q appropriately.
    Actually, simpler: if a and b are both powerful and a+b=N,
    we can try N = k * Q where Q is square-free, and look for a = x^2 * Q, b = y^2 * Q
    which gives x^2 + y^2 = k. But this requires Q to divide both a and b,
    meaning Q divides N and the square-free parts of a and b both contain Q.
    
    Alternative: generate powerful numbers up to some bound, then for each candidate N
    count representations. But N can be up to 10^18.
    
    Better approach: 
    1. Generate powerful numbers up to a reasonable bound (say 10^6 or 10^7).
    2. For each pair (a, b) of powerful numbers with a <= b, compute N = a + b.
    3. Count how many pairs give each N.
    4. Pick the N with the most pairs.
    
    But N must be <= 10^18, and we need to maximize the count.
    
    Let's generate powerful numbers up to B, then for all pairs with a+b <= 10^18,
    count the representations.
    
    To be efficient, we can:
    - Generate powerful numbers up to B where B is chosen so that B^2 <= 10^18, i.e., B <= 10^9.
    - But generating all powerful numbers up to 10^9 might be too many.
    
    Let's estimate: the number of powerful numbers up to X is about O(sqrt(X)).
    For X = 10^9, that's about 31622 numbers. Pairs would be ~5e8, which is too many.
    
    So we need a smarter approach.
    
    Insight: If we fix the square-free part, we can reduce the problem.
    Let a = s1 * u^2, b = s2 * v^2 where s1, s2 are square-free.
    Then N = s1*u^2 + s2*v^2.
    
    If we choose s1 = s2 = s (same square-free part), then N = s*(u^2 + v^2).
    So for a fixed s, the number of representations is the number of ways to write
    N/s as a sum of two squares, with u <= v.
    
    The number of representations of m as a sum of two squares is related to the
    prime factorization of m. Specifically, if m = 2^a * prod p_i^a_i * prod q_j^b_j
    where p_i ≡ 1 mod 4 and q_j ≡ 3 mod 4, then the number of representations is
    4 * prod (a_i + 1) if all b_j are even, else 0.
    
    So to maximize the number of representations, we want N/s to have many prime factors
    ≡ 1 mod 4 with high exponents, and all primes ≡ 3 mod 4 with even exponents.
    
    Let's try: pick s = 1 (so a and b are perfect squares). Then N = u^2 + v^2.
    We want N to have many representations as a sum of two squares.
    
    The number of representations r_2(n) = 4 * (d_1(n) - d_3(n)) where d_1, d_3 are
    divisors ≡ 1, 3 mod 4.
    
    To maximize this, we want n to be a product of many primes ≡ 1 mod 4.
    
    Let's construct N as a product of the first k primes ≡ 1 mod 4, each to some power.
    Then check that N itself is not necessarily powerful, but we need a and b to be powerful.
    
    Wait, a = u^2 and b = v^2 are always powerful (since they're perfect squares).
    So if we pick N = u^2 + v^2 where N has many representations as a sum of two squares,
    then all those representations (u', v') give powerful a = u'^2, b = v'^2.
    
    So the problem reduces to: find N <= 10^18 that maximizes r_2(N)/4 (number of unordered
    pairs u <= v with u^2 + v^2 = N).
    
    Let's construct N as a product of primes ≡ 1 mod 4.
    Primes ≡ 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, ...
    
    Let's try N = 5^a * 13^b * 17^c * ... and see how many representations we get.
    
    The number of representations (counting order and signs) is 4 * prod (e_i + 1)
    where e_i are exponents of primes ≡ 1 mod 4.
    
    The number of unordered pairs (u, v) with u >= 1, v >= 1, u <= v is:
    (r_2(N) - 4) / 8 if N is not a square, or (r_2(N) - 4) / 8 + 1 if N is a square... 
    Actually, let's just compute it directly.
    
    For a given N, the number of representations as u^2 + v^2 with u >= 0, v >= 0,
    u <= v, is:
    - If N = 0: 1
    - If N is a square: (r_2(N) + 4) / 8
    - Otherwise: r_2(N) / 8
    
    Where r_2(N) is the number of integer solutions (x, y) to x^2 + y^2 = N.
    
    Let's just generate candidate N values and count representations by actually
    finding all (u, v) pairs.
    
    Given the 120s time limit, let's try a practical approach:
    1. Generate powerful numbers up to B = 10^6.
    2. For each pair, count N = a + b.
    3. Find the N with the most pairs.
    
    Number of powerful numbers up to 10^6: let's estimate.
    """
    B = 10**6
    powerful = _generate_powerful_numbers(B)
    powerful_set = set(powerful)
    
    # Count representations for each N
    from collections import defaultdict
    counts = defaultdict(int)
    pairs_by_n = defaultdict(list)
    
    n_powerful = len(powerful)
    for i in range(n_powerful):
        a = powerful[i]
        for j in range(i, n_powerful):
            b = powerful[j]
            n = a + b
            if n <= 10**18:
                counts[n] += 1
                if len(pairs_by_n[n]) < 1000:  # Store up to 1000 pairs
                    pairs_by_n[n].append([a, b])
    
    # Find the N with the most representations
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_n = max(counts.keys(), key=lambda n: counts[n])
    best_pairs = pairs_by_n[best_n]
    
    # Verify all pairs are valid
    verified_pairs = []
    for a, b in best_pairs:
        if a <= b and a + b == best_n and _is_powerful(a) and _is_powerful(b):
            verified_pairs.append([a, b])
    
    # If we didn't store enough, regenerate
    if len(verified_pairs) < counts[best_n]:
        verified_pairs = []
        for a in powerful:
            b = best_n - a
            if b < a:
                break
            if b in powerful_set and _is_powerful(b):
                verified_pairs.append([a, b])
    
    return {'N': best_n, 'pairs': verified_pairs}
# EVOLVE-BLOCK-END
