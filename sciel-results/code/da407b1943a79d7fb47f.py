"""Rewritten program: frequency-based search for N with many powerful-number pair representations."""

def construct():
    import time
    import math
    from collections import Counter

    deadline = time.time() + 140.0  # safety margin before 150s kill
    LIMIT = 10**18

    # --- Generate powerful numbers up to LIMIT ---
    # A powerful number is 1 or a product of prime powers with exponent >= 2.
    # Generate by: for each set of distinct primes, multiply prime powers.
    # Simpler: recursive generation over primes.

    def gen_powerful(limit):
        # Generate all powerful numbers <= limit.
        # Use the structure: powerful numbers are of the form a^2 * b^3 where a,b are integers
        # (since any powerful number can be written as a^2 * b^3 with gcd... actually not exactly,
        #  but the standard generation is via prime powers).
        # Better: generate via recursion over primes.
        # First get primes up to limit^(1/2) = 10^9 -- too many.
        # Instead, generate prime powers first, then combine.
        
        # Approach: generate all prime powers p^k (k>=2) up to limit.
        # Then a powerful number is a product of distinct such prime powers (one per prime).
        # So we need to combine prime powers for distinct primes.
        
        # Step 1: for each prime p, the possible prime-power factors are p^2, p^3, p^4, ... <= limit.
        # Step 2: a powerful number = product over distinct primes of (p^{e_p}) where e_p>=2.
        # This is like a subset product where each prime contributes one power.
        
        # To keep it manageable, limit the number of distinct prime factors (small).
        # Generate primes up to limit^(1/2) = 1e9 is too many. 
        # But prime powers p^2 <= 1e18 means p <= 1e9. That's ~5e7 primes, too many to enumerate all.
        
        # Alternative: generate powerful numbers directly by a known enumeration.
        # The count of powerful numbers up to 1e18 is about 1.05e6.
        # We can generate them by: for b in range(1, isqrt(limit)+1): # b^2 factor
        #   for a in ... : a^3 * b^2 ... this double counts.
        
        # Standard method: powerful numbers are exactly the numbers of the form a^2 * b^3
        # where a and b are integers? No, that's not unique. 
        # Actually every powerful number n can be written as a^2 * b^3 uniquely? 
        # Let's think: n = prod p_i^{e_i}, e_i>=2. Write e_i = 2*a_i + 3*b_i? Not unique.
        # But n = a^2 * b^3 where a = prod p_i^{floor((e_i-2)/2)}... hmm.
        # Actually: let e_i = 2*u_i + 3*v_i. We can always represent. The representation 
        # a^2 b^3 with a,b integers: choose v_i such that... 
        # The cleanest: n is powerful iff n = a^2 b^3 for some integers a,b (with a,b chosen appropriately).
        # Yes! A number is powerful iff it is of the form a^2 * b^3 for positive integers a,b.
        # Proof: if n = prod p^{e_p}, e_p>=2, set b = prod p^{floor(e_p/3)}... 
        # Let e_p = 3*q_p + r_p, r_p in {0,1,2}. If r_p=0: p^{e_p} = (p^{q_p})^3. 
        # If r_p=1: e_p>=2 so e_p>=4, p^{e_p} = p^3 * p^{e_p-3} = (p^{q_p})^3 * (p^{(e_p-3)/2})^2 since e_p-3 is odd>=1... 
        # e_p-3 = 3q_p -3 + ... let me redo. e_p = 3q_p + r_p.
        # r_p=0: p^{e_p} = (p^{q_p})^3 * 1^2.
        # r_p=1: e_p = 3q_p+1 >= 2 => q_p>=1 (since if q_p=0, e_p=1 not powerful). So e_p>=4.
        #   p^{e_p} = p^{3q_p+1} = (p^{q_p})^3 * p^1. But p^1 is not a square. 
        #   Hmm, need p^{e_p} = a^2 b^3. Let b_p = p^{q_p}, then b_p^3 = p^{3q_p}. 
        #   Remaining p^1 = a_p^2? No. So this form fails for r_p=1?
        # Let me check: is 16 = 2^4 powerful? Yes. 16 = a^2 b^3? 16 = 4^2 * 1 = 16*1. Yes a=4,b=1. 
        #   Or 16 = 1^2 * ... no b^3=16 not a cube. So a=4,b=1 works: 16=16*1. Good.
        #   In general r_p=1, e_p=3q_p+1, q_p>=1: p^{e_p} = p^{3q_p+1} = (p^{q_p})^3 * p^1.
        #   To write as a^2 b^3: set b_p = p^{q_p}, a_p such that a_p^2 = p^1? No.
        #   Instead set b_p = p^{q_p - 1} (q_p>=1 so this is >=1), then b_p^3 = p^{3q_p-3}.
        #   Remaining: p^{3q_p+1 - (3q_p-3)} = p^4 = (p^2)^2. So a_p = p^2.
        #   So p^{e_p} = (p^2)^2 * (p^{q_p-1})^3. Works!
        # r_p=2: e_p=3q_p+2>=2, q_p>=0. p^{e_p} = p^{3q_p+2} = (p^1)^2 * (p^{q_p})^3. a_p=p, b_p=p^{q_p}. Works.
        # So yes, every powerful number = a^2 b^3. And conversely a^2 b^3 is powerful.
        # Great, so generate all a^2 b^3 <= limit.
        
        # But a^2 b^3 with a,b up to 1e9 each is 1e18 combinations, too many.
        # We need a smarter enumeration. 
        # Note b^3 <= limit => b <= limit^(1/3) = 1e6.
        # a^2 <= limit / b^3 => a <= sqrt(limit/b^3).
        # So iterate b from 1 to 1e6, and for each b, a from 1 to sqrt(limit/b^3).
        # Total iterations = sum_{b=1}^{1e6} sqrt(limit/b^3) = sqrt(limit) * sum b^{-3/2} ~ 1e9 * 2.6 = 2.6e9. Too many.
        
        # Hmm, that's too slow in Python. Need a better way.
        # Alternative: the number of powerful numbers up to 1e18 is ~1e6, so we should generate 
        # exactly those, not iterate all a,b pairs.
        # Better generation: use the prime-power product method with limited primes.
        
        # Let's generate prime powers and combine.
        # Get primes up to 1e9? Too many. 
        # But for a powerful number, each prime factor p must have p^2 <= limit, so p <= 1e9.
        # However, the product of several such p^2's must be <= 1e18.
        # Number of distinct prime factors is small (since (2*3*5*7*11*13*17*19*23)^2 is huge).
        # (product of first 9 primes) = 223092870, squared ~ 5e16 < 1e18. First 10: *29=6.47e9, squared ~4e19 >1e18.
        # So at most 9 distinct primes.
        
        # This is getting complex. Let me use a different, practical approach:
        # Generate powerful numbers up to a smaller limit (say 1e12 or 1e14) where generation is fast,
        # then use pairwise sums to find N. But N must be <= 1e18, and a,b <= N.
        # If we only have powerful numbers up to M, then a,b <= M, so N <= 2M.
        # To get N up to 1e18 we'd need M up to 5e17. 
        # But the high-representation N values are likely not at the very top; 
        # the frequency approach finds N with many small powerful pairs.
        
        # Practical plan: generate powerful numbers up to M = 2e12 (fast), 
        # compute pairwise sum frequencies, find top N candidates, verify.
        # But N from two numbers <= 2e12 is <= 4e12, well within 1e18.
        # The representation count for such N is what we maximize.
        
        # How many powerful numbers up to 2e12? About (2e12)^0.5 * const ~ 1.4e6 * ~0.65 ~ maybe 1e5-1e6.
        # Let's generate them via the a^2 b^3 method but bounded:
        # b^3 <= 2e12 => b <= 12599. a^2 <= 2e12/b^3 => a <= sqrt(2e12/b^3).
        # Total = sum_{b=1}^{12599} sqrt(2e12/b^3) ~ sqrt(2e12)*sum b^{-3/2} ~ 1.4e6 * 2.6 ~ 3.7e6. Feasible!
        
        pass

    # Generate powerful numbers up to M using a^2 b^3 enumeration
    M = 2 * 10**12
    b_max = int(round(M ** (1/3))) + 1
    powerful_set = set()
    powerful_set.add(1)
    for b in range(1, b_max + 1):
        b3 = b * b * b
        if b3 > M:
            break
        a_max = int(math.isqrt(M // b3))
        for a in range(1, a_max + 1):
            val = a * a * b3
            if val <= M:
                powerful_set.add(val)
    powerful = sorted(powerful_set)
    powerful_set = set(powerful)
    n_pow = len(powerful)
    print(f"Generated {n_pow} powerful numbers up to {M}")

    # --- Frequency-based: count pairwise sums ---
    # For each pair (i,j) with i<=j, sum = powerful[i]+powerful[j]. Count frequencies.
    # This is O(n_pow^2) which for n_pow ~ 1e6 is 1e12 -- too slow.
    # Need to limit. Use a subset or limit the pair count.
    
    # Instead: sample a large subset of powerful numbers and do pairwise sums on the subset.
    # Or: only consider pairs where both are <= some threshold to keep sum <= 2*threshold.
    # To find N with many representations, we want many pairs (a,b) with a+b=N.
    # Using the full set is ideal but O(n^2). 
    # Compromise: use a subset of size K where K^2 is feasible (K ~ 20000 => 4e8 pairs, borderline in Python).
    # Let's use K ~ 10000 => 1e8 pairs, feasible in ~tens of seconds.
    
    # Pick a subset: the smaller powerful numbers give more overlap in sums.
    # Use the first K powerful numbers (smallest).
    K = 12000
    subset = powerful[:K]
    
    sum_count = Counter()
    # Pairwise sums within subset
    for i in range(len(subset)):
        ai = subset[i]
        for j in range(i, len(subset)):
            s = ai + subset[j]
            if s <= LIMIT:
                sum_count[s] += 1
        if time.time() > deadline - 30:
            break
    
    # Get top candidates
    top_candidates = sum_count.most_common(50)
    print(f"Top candidate: {top_candidates[0][0]} with {top_candidates[0][1]} reps (in subset)")
    
    # --- Verify candidates with exact factorization ---
    def is_powerful(n):
        """Check if n is powerful via exact factorization."""
        if n <= 0:
            return False
        if n == 1:
            return True
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt < 2:
                    return False
            p += 1 if p == 2 else 2
        if temp > 1:
            # temp is a prime factor with exponent 1
            return False
        return True
    
    best_N = None
    best_pairs = []
    best_count = 0
    
    for N, approx_count in top_candidates:
        if time.time() > deadline - 10:
            break
        # Find all pairs (a,b) with a<=b, a+b=N, a,b powerful (from our generated set, 
        # but also need to ensure a,b <= M since we only generated up to M)
        pairs = []
        for a in powerful:
            if 2 * a > N:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
            print(f"New best: N={N}, count={best_count}")
    
    if best_N is None:
        # Fallback: use the top candidate from frequency
        best_N = top_candidates[0][0]
        best_pairs = []
        for a in powerful:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set:
                best_pairs.append([a, b])
        best_count = len(best_pairs)
    
    return {'N': best_N, 'pairs': best_pairs}
