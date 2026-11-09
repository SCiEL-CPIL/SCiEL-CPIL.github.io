import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
                
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    def count_pairs_for_N(N, P_set, P_list):
        """Count pairs (a,b) with a<=b, a+b=N, both powerful."""
        count = 0
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                count += 1
        return count

    # Generate powerful numbers up to a large limit
    # We'll use multiple strategies:
    # 1. Generate P up to M
    # 2. Find good candidate N values by sampling
    # 3. Verify and count pairs for the best candidates

    best_result = None
    best_score = 0

    # Strategy: Use a larger M but smarter N selection
    # Instead of O(P^2), we'll:
    # - Generate P up to M
    # - Sample random pairs to find promising N values
    # - Also check N values that are sums of small powerful numbers
    # - Verify top candidates with full O(P) count

    M = 10**8
    P = generate_powerful(M)
    P_set = set(P)
    L = len(P)

    if time.time() > deadline - 30:
        # Fallback to smaller M
        M = 10**6
        P = generate_powerful(M)
        P_set = set(P)
        L = len(P)

    # Candidate N selection strategy:
    # 1. Check sums of pairs of small powerful numbers (likely to have many reps)
    # 2. Sample random pairs
    # 3. Check N values near multiples of small powerful numbers

    candidate_Ns = set()

    # Add sums of pairs from the first K powerful numbers
    K = min(500, L)
    small_P = P[:K]
    for i in range(K):
        for j in range(i, K):
            s = small_P[i] + small_P[j]
            if s <= 2 * M:
                candidate_Ns.add(s)

    # Add some random samples
    import random
    random.seed(42)
    for _ in range(500):
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        s = P[i] + P[j]
        if s <= 2 * M:
            candidate_Ns.add(s)

    # Also add N values that are "popular" based on frequency of small components
    # Count how often each powerful number appears in small sums
    freq = defaultdict(int)
    for i in range(K):
        for j in range(i, K):
            freq[small_P[i]] += 1
            if small_P[j] != small_P[i]:
                freq[small_P[j]] += 1

    # Add N = a + b where a and b are both high-frequency
    high_freq = sorted(freq.keys(), key=lambda x: -freq[x])[:50]
    for a in high_freq:
        for b in high_freq:
            if a <= b:
                s = a + b
                if s <= 2 * M:
                    candidate_Ns.add(s)

    # Sort candidates by heuristic: prefer N that can be written as sum of many small powerful numbers
    # We'll just evaluate the top candidates

    # Limit number of candidates to evaluate
    candidate_list = list(candidate_Ns)
    # Prioritize: N values that are sums of many pairs from small_P
    # Compute a quick score for each candidate
    scored_candidates = []
    for N in candidate_list:
        # Quick score: count how many a in small_P have N-a also in small_P
        quick_count = 0
        for a in small_P:
            if a > N // 2:
                break
            if (N - a) in P_set:
                quick_count += 1
        scored_candidates.append((quick_count, N))

    scored_candidates.sort(key=lambda x: -x[0])

    # Evaluate top candidates with full count
    evaluated = 0
    for quick_count, N in scored_candidates:
        if time.time() > deadline - 15:
            break
        if N < 2 or N > 10**18:
            continue

        # Full count
        count = count_pairs_for_N(N, P_set, P)

        if count > best_score:
            best_score = count
            # Recover pairs
            pairs = []
            for a in P:
                if a > N // 2:
                    break
                b = N - a
                if b in P_set:
                    pairs.append([a, b])
            best_result = {'N': N, 'pairs': pairs}

        evaluated += 1
        if evaluated >= 200:
            break

    if best_result is not None:
        return best_result

    # Fallback: try a known good approach
    # N = 2 * (some powerful number) often works well
    # Or try N that is sum of many small powerful numbers

    # Try N values from the first few hundred powerful numbers doubled
    for a in P[:100]:
        if time.time() > deadline - 10:
            break
        N = 2 * a
        if N < 2 or N > 10**18:
            continue
        count = count_pairs_for_N(N, P_set, P)
        if count > best_score:
            best_score = count
            pairs = []
            for x in P:
                if x > N // 2:
                    break
                y = N - x
                if y in P_set:
                    pairs.append([x, y])
            best_result = {'N': N, 'pairs': pairs}

    if best_result is not None:
        return best_result

    # Final fallback
    return {'N': 2, 'pairs': [[1, 1]]}
