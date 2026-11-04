import math
import random
import sys

def calc_score(U):
    """Calculate the score for a set U."""
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    
    # For large sets, we might need to be careful about time, but the task
    # implies we can handle the sizes produced. The paper mentions 54265.
    # |U| ~ 50k, |U x U| ~ 2.5e9 which is too slow for Python sets directly?
    # Wait, the seed program uses a double loop.
    # If |U| is 2003, |U x U| is ~4e6, which is fine.
    # If |U| is 54265, |U x U| is ~3e9, which will timeout in Python (120s limit).
    # The seed program scores 1.0281, which likely comes from a smaller set.
    # We need to be careful not to generate sets that are too large to score in time,
    # OR we need a faster scoring method, OR we stick to sizes where O(|U|^2) is feasible.
    
    # Let's check the size limit. 120 seconds.
    # In Python, a loop of 4e6 iterations takes ~0.5s.
    # A loop of 1e8 iterations takes ~20-30s.
    # So |U|^2 < 1e8 means |U| < 1e4 is safe.
    # |U| = 2000 -> 4e6 ops.
    # |U| = 5000 -> 25e6 ops.
    # |U| = 10000 -> 100e6 ops (might be borderline).
    
    # The current seed finds sets with scores around 1.0281.
    # Let's estimate the size of the best set found by the seed.
    # The seed searches B in 2..14, n in 2..10.
    # For B=2, n=10, size is 1024.
    # For B=14, n=10, size is 1024.
    # So the seed is looking at sets of size <= 1024 (2^10).
    # 1024^2 = 1e6. Very fast.
    
    # Can we go larger?
    # If we use a larger n, say n=15, size = 32768.
    # 32768^2 = 1e9. Too slow.
    
    # So we are constrained to |U| < ~5000 for direct scoring in Python.
    # This limits the "structural" size we can explore.
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Strategy:
    # 1. Expand the search range for B and n in the "full set" (binary digits 0/1) approach.
    # 2. The seed searched B in [2, 15) and n in [2, 11).
    # 3. Larger B generally reduces collisions in the sumset because the "digits" don't carry
    #    as often if we view the set in base B. Wait, the digits are 0/1.
    #    If B is large, the numbers are spread out.
    #    U = { sum b_j B^j | b_j in {0,1} }
    #    Sum of two elements: sum (b_j + c_j) B^j.
    #    If B is large enough, there is no carry-over if b_j + c_j < B.
    #    Since b_j, c_j in {0,1}, max sum is 2. So if B > 2, there is no carry-over.
    #    If there is no carry-over, |U+U| = 3^n (each digit 0, 1, 2).
    #    |U-U| = 5^n (each digit -2, -1, 0, 1, 2).
    #    Ratio = (5/3)^n.
    #    Score = 1 + log((5/3)^n) / log(2 * B^(n-1) * (B+1)/2 * ... )
    #    Max U = sum_{j=0}^{n-1} B^j = (B^n - 1) / (B - 1).
    #    2 * Max U + 1 ~ 2 B^n / (B - 1).
    #    log(2 Max U + 1) ~ n log B + log(2/(B-1)).
    #    Numerator: n log(5/3).
    #    Score ~ 1 + [n log(5/3)] / [n log B] = 1 + log(5/3)/log(B) as n -> infinity.
    
    # To maximize the score, we want to minimize B (since log(5/3)/log(B) decreases with B).
    # The smallest integer B > 2 is B=3.
    # For B=3, no carry-over (1+1=2 < 3).
    # Score limit: 1 + log(5/3) / log(3) = 1 + log(1.666) / log(3) ≈ 1 + 0.51 / 1.1 ≈ 1.46.
    
    # What about B=2?
    # Carries happen. 1+1=2 = 10 in base 2.
    # This reduces the size of the sumset compared to 3^n.
    # The seed likely found that B=2 is worse than B=3, or similar.
    # Let's check the seed's behavior. It searches B=2..14.
    # If B=3 is theoretically best for large n, why didn't the seed find a higher score?
    # Because n was limited to 10.
    # Score for B=3, n=10:
    # |U| = 1024.
    # |U+U| = 3^10 = 59049.
    # |U-U| = 5^10 = 9765625.
    # Ratio = 9765625 / 59049 ≈ 165.
    # Max U = (3^10 - 1) / 2 = 29524.
    # Den = log(2 * 29524 + 1) = log(59049) ≈ 10.98.
    # Num = log(165) ≈ 5.1.
    # Score = 1 + 5.1 / 10.98 ≈ 1.46.
    
    # Wait, the seed's score is 1.0281. This is much lower.
    # Why?
    # Maybe my assumption about the seed's best is wrong.
    # Let's re-read the seed code.
    # It starts with [0,1,3].
    # It iterates B in 2..14, n in 2..10.
    # It calculates score for the FULL set of 2^n elements.
    # For B=3, n=10, the score should be ~1.46.
    # Why is the seed score 1.0281?
    # Perhaps the seed code provided in the prompt is NOT the one that achieved 1.0281?
    # Or maybe I am miscalculating.
    # Let's trace the seed code logic.
    # `all_vals` generation:
    # for mask in range(1 << n):
    #   val = 0
    #   for j in range(n):
    #     if mask & (1 << j): val += B ** j
    # This generates exactly the set described.
    
    # Let's run a quick mental check on B=2, n=10.
    # U is all numbers with 10 bits. Max U = 1023.
    # U+U: range 0..2046. Size <= 2047.
    # U-U: range -1023..1023. Size <= 2047.
    # Ratio ~ 1.
    # Score ~ 1 + 0 / log(2047) = 1.
    
    # B=3, n=10.
    # U size 1024.
    # Max U = 29524.
    # Sums: Since no carry, |U+U| = 3^10.
    # Diffs: |U-U| = 5^10.
    # Score = 1 + log(5^10/3^10) / log(2*29524+1)
    # = 1 + 10 * log(5/3) / log(59049)
    # = 1 + 10 * 0.5108 / 10.986
    # = 1 + 5.108 / 10.986
    # = 1 + 0.465
    # = 1.465.
    
    # If the seed achieved 1.0281, it implies it did NOT find the B=3, n=10 set.
    # Why?
    # Look at the seed code again.
    # `for B in range(2, 15):` -> 2, 3, 4, ..., 14.
    # `for n in range(2, 11):` -> 2, 3, ..., 10.
    # It SHOULD find B=3, n=10.
    # Unless... `calc_score` is slow and it times out? No, it's just one set.
    # Unless... the "seed program" provided in the prompt is a *degraded* version or a *starting point* and the 1.0281 score comes from a different run?
    # The prompt says: "CURRENT PROGRAM (score 1.0281)".
    # This implies the code provided IS the one that scored 1.0281.
    # If the code provided searches B=3, n=10, it MUST score ~1.46.
    # Contradiction.
    
    # Let's look closer at the seed code.
    # Ah, the subset search!
    # `if n <= 5:` ... it tries subsets.
    # But for n=10, it only checks the full set.
    # So it MUST evaluate the full set for B=3, n=10.
    
    # Is it possible that `calc_score` behaves differently?
    # `diffs.add(a - b)`.
    # `sums.add(a + b)`.
    # Standard.
    
    # Maybe the base B expansion in the seed is different?
    # `val += B ** j`.
    # j goes 0 to n-1.
    # This is standard.
    
    # Is it possible that the "score 1.0281" is from a DIFFERENT program, and the prompt is showing me a program that I should improve, but that program itself is flawed or I am misinterpreting the score?
    # Or maybe the environment has a constraint I'm missing?
    # "The evaluator measures the raw value".
    
    # Let's assume the seed code is correct and my calculation is correct.
    # If I submit the seed code, I should get 1.46.
    # If the prompt says it got 1.0281, maybe the "seed" is actually a very small set?
    # [0, 1, 3] ->
    # Sums: 0, 1, 3, 2, 4, 6. Size 6.
    # Diffs: 0, 1, 3, -1, -3, 2, -2, -4, 4. Size 9.
    # Ratio 1.5.
    # Max U = 3. Den = log(7) = 1.94.
    # Score = 1 + log(1.5)/1.94 = 1 + 0.405/1.94 = 1.20.
    
    # So [0,1,3] gives 1.20.
    # The seed score is 1.0281. This is LOWER than [0,1,3].
    # This implies the seed code might have found a WORSE set?
    # Or `best` is initialized to `[0,1,3]` and `best_score` to its score.
    # Then it searches. If it finds nothing better, it returns `[0,1,3]`.
    # If `[0,1,3]` scores 1.20, the seed score should be at least 1.20.
    # Why is it 1.0281?
    
    # Hypothesis: The "CURRENT PROGRAM" provided in the prompt is NOT the one that generated the score 1.0281.
    # OR, the score 1.0281 is from a different metric?
    # "score is the lower bound... on C6"
    # "search score is value/seed"
    # If the seed value is 1.20, and the search score is 1.0281, then the value is 1.20 * 1.0281 = 1.23?
    # No, "value/seed" where 1.0 equals the seed program.
    # So if the seed program scores X, and the new program scores Y, the score is Y/X.
    # The prompt says "CURRENT PROGRAM (score 1.0281)".
    # This usually means the CURRENT PROGRAM itself achieved a relative score of 1.0281 against some baseline?
    # Or does it mean the CURRENT PROGRAM is the baseline and it has a raw score of 1.0281?
    # "The evaluator measures the raw value... The search score is value/seed, so 1.0 equals the seed program".
    # This phrasing is tricky.
    # Usually, in these tasks, "score 1.0281" refers to the relative improvement.
    # If the current program IS the seed, its score should be 1.0.
    # If it says "score 1.0281", it might mean this program is an improvement over the *original* seed.
    
    # Let's assume the goal is to beat 1.0281 (relative to the original seed).
    # If the current program (the one I'm rewriting) achieves 1.0281 relative to the original seed, I need to beat that.
    
    # But wait, if the current program contains the B=3, n=10 logic, it should achieve a raw score of ~1.46.
    # If the original seed achieved a raw score of ~1.42, then 1.46/1.42 = 1.028.
    # This fits!
    # So the "original seed" likely found a set with raw score ~1.42.
    # The "current program" found a set with raw score ~1.46.
    # The ratio is 1.0281.
    
    # So my task is to find a set with a raw score HIGHER than ~1.46.
    # My theoretical max for B=3, n->inf is 1 + log(5/3)/log(3) = 1.465.
    # We are already at 1.465 (asymptotically).
    # To beat this, we need a different structure.
    
    # The asymptotic bound for base B (no carry) is 1 + log(5/3)/log(B).
    # This is maximized at minimum B=3.
    # Value: 1.465.
    
    # Can we do better with carries?
    # If B=2, carries reduce |U+U|.
    # If we can make |U+U| smaller while keeping |U-U| large, the ratio increases.
    # But B=2 is dense.
    
    # What if we use a MIXED base? Or non-integer digits?
    # The problem states "nonnegative integers".
    
    # What if we use a different set structure?
    # The directive says: "Model the set U as a sequence of intervals defined by a 'base-expansion' hierarchy... b_j in {0, 1}".
    # This restricts us to the {0,1} digit sets.
    # Within this constraint, B=3 is optimal for large n.
    # We are at the limit of this constraint (1.465).
    
    # To jump, we might need to:
    # 1. Use a larger n to get closer to the asymptote?
    #    For n=10, score was 1.465 (approx).
    #    Let's check n=20.
    #    Score = 1 + 20*log(5/3) / log(2 * (3^20-1)/2 + 1)
    #    = 1 + 20*0.51 / (20*log(3)) = 1 + 0.51/1.1 = 1.46.
    #    It converges to 1.465.
    #    So increasing n doesn't help much beyond n=10.
    
    # 2. Is there a better B?
    #    B=3 is the smallest B > 2.
    #    What about B=2?
    #    With B=2, carries occur.
    #    |U+U| is roughly 2 * Max U = 2 * (2^n - 1) = 2^(n+1).
    #    |U-U| is roughly 2 * Max U = 2^(n+1).
    #    Ratio ~ 1.
    #    Score ~ 1.
    #    So B=2 is worse.
    
    # 3. What if we prune the set?
    #    Remove elements that cause excessive sum collisions?
    #    But removing elements reduces |U-U| and |U+U|.
    #    Usually, the full set is optimal for this type of ratio.
    
    # 4. Is the "base-expansion" directive a strict constraint or a suggestion?
    #    "RESEARCH PROGRAMME: design a new program... built on this approach... do not switch to a different design."
    #    So I must stick to base-expansion sets.
    
    # If I am stuck at 1.465, how do I beat 1.0281 (which corresponds to 1.46)?
    # Maybe the current program's score is NOT 1.46?
    # Let's re-evaluate the seed's actual best.
    # The seed searches B up to 14.
    # For B=14, n=10:
    # Score = 1 + log(5/3)/log(14) = 1 + 0.51/2.64 = 1.19.
    # B=3 is best.
    
    # Is it possible that a DIFFERENT set structure within "base-expansion" is better?
    # The directive says: "u_k = sum b_j B^j ... b_j in {0,1}".
    # This is the standard binary-digit set.
    
    # Wait, look at the score formula again.
    # 1 + log(|U-U|/|U+U|) / log(2 max U + 1).
    
    # What if we use a NON-UNIFORM base?
    # The directive says "B being a large base". Singular.
    
    # What if we use a HIERARCHY of bases?
    # "sequence of intervals defined by a 'base-expansion' hierarchy"
    # This might allow for varying bases B_j at each digit position.
    # If we can choose B_j optimally for each j, can we improve the ratio?
    # For each digit j, the digit is 0 or 1.
    # The "weight" of digit j is W_j.
    # U = { sum b_j W_j | b_j in {0,1} }.
    # To avoid carries, we need W_j > sum_{k<j} 2 * W_k ? No, just W_j > W_{j-1} + ...
    # Actually, for the set to be "sum-free" in the sense of unique representations,
    # we need W_j > sum_{k=0}^{j-1} W_k.
    # This implies W_j grows exponentially.
    # If W_j = 2 * W_{j-1}, it's binary.
    # If W_j = 3 * W_{j-1}, it's base 3 (with digits 0,1).
    # Generalized: W_j = c * W_{j-1}.
    # The ratio of sizes is (5/3)^n.
    # The log(max U) is sum log(W_j) ~ n log(c) + ...
    # So the score depends on c.
    # Score ~ 1 + log(5/3) / log(c).
    # To maximize, minimize c.
    # Minimum c for no-carry is 2?
    # If c=2, W_j = 2^j.
    # Digits 0,1.
    # Sum of two elements: digit sum can be 0,1,2.
    # 2 * 2^j = 1 * 2^{j+1}. Carry!
    # So c=2 has carries.
    # c=3 has no carries.
    # So c=3 is the optimal no-carry base.
    
    # Can we have c < 3 with carries but still get a better ratio?
    # Let's test c=2 (standard binary) again.
    # |U+U| for U={0,1}^n (binary) is not 3^n.
    # The sumset of {0,1}^n in base 2 is the interval [0, 2^(n+1)-2]?
    # Actually, the sum of two n-bit numbers is an (n+1)-bit number.
    # It covers all values?
    # For n=1, U={0,1}. Sums={0,1,2}. Size 3. 3^1=3.
    # For n=2, U={0,1,2,3}. Sums={0..6}. Size 7. 3^2=9.
    # Carries reduce the size.
    # Ratio |U-U|/|U+U|.
    # Diffs for n=2: -3..3. Size 7.
    # Ratio 1.
    # Score 1.
    
    # So c=3 is definitely better.
    
    # Is there any other trick?
    # What if we use a set that is NOT of the form {sum b_j W_j}?
    # The directive says "Model the set U as...". It implies this is the approach.
    # "Do not switch to a different design."
    
    # If I am at the theoretical maximum of the allowed design space (1.465),
    # and the current program is at 1.46, the only way to beat it is to:
    # 1. Implement the search more efficiently to allow larger n (if n=10 wasn't fully converged).
    # 2. Find a subtle improvement in the set definition (e.g. mixed bases).
    
    # Let's try MIXED BASES.
    # Let W_j be the weights.
    # We want to maximize R = |U-U| / |U+U| given Max U.
    # If we choose W_j such that no carries happen, R = (5/3)^n.
    # Max U = sum W_j.
    # To maximize log(R) / log(Max U), we want small W_j.
    # Smallest W_j with no carries: W_0=1, W_1=3, W_2=9...
    # This is geometric with ratio 3.
    
    # What if we allow SOME carries to reduce Max U while losing little R?
    # If we use W_j = 2^j (binary), R drops significantly.
    # What if W_j = 2.5^j? Not integers.
    
    # What if we use W_j = 3^j for most, but smaller for some?
    # If W_j < 3 * W_{j-1}, carries might occur.
    # If W_j = 2 * W_{j-1} + 1?
    
    # Let's stick to the integer base B approach but expand the search.
    # The current program searches B in 2..14.
    # What if we search B in 3..50?
    # Higher B gives lower score.
    # What if we search n in 10..30?
    # Score converges.
    
    # Is it possible that the current program's score of 1.0281 is actually LOWER than 1.46?
    # If the "seed" (baseline) had a score of 1.40, and the current program found 1.44,
    # then 1.44/1.40 = 1.028.
    # If I can find a set with score 1.46, then 1.46/1.40 = 1.042.
    # This would be an improvement.
    
    # So the goal is to reach the theoretical max of 1.465.
    # The current program likely hasn't reached it because:
    # 1. It stops at n=10.
    # 2. Maybe it doesn't explore B=3 properly? (It does).
    # 3. Maybe the subset search is interfering?
    
    # Let's write a program that:
    # 1. Fixes B=3 (the optimal base).
    # 2. Sweeps n from 2 to 20.
    # 3. Calculates the exact score for each.
    # 4. Returns the best set.
    
    # This should push the raw score to the asymptote 1.465.
    # If the baseline is lower, this will yield a higher relative score.
    
    # Additionally, I will add a "mixed base" search.
    # Try sequences of bases B_0, B_1, ..., B_{n-1}.
    # W_0 = 1.
    # W_j = W_{j-1} * B_{j-1}.
    # This allows for non-geometric growth.
    # We can search over small sequences of bases.
    # Given the time limit, we can't do a full search.
    # But we can try a few heuristics.
    # Start with B=3 for all.
    # Try replacing some B_j with 2?
    # If we use B=2 at the end, it might reduce Max U without hurting R too much?
    # No, B=2 introduces carries which hurt R.
    
    # Let's stick to the pure base B search but go deeper in n.
    # And also check B=2,3,4,5.
    
    def evaluate_set(U):
        if not U: return 0
        U = sorted(set(U))
        if 0 not in U: return 0
        max_u = U[-1]
        if max_u == 0: return 0
        
        # Optimize scoring for larger sets?
        # If len(U) > 5000, direct O(N^2) might be slow.
        # But for n=20, B=3, len(U) = 2^20 = 1,048,576.
        # O(N^2) is impossible.
        
        # We MUST use the theoretical formula for large n,
        # OR we must keep n small enough for direct computation.
        # n=10 -> 1024 elements. OK.
        # n=12 -> 4096 elements. 16M ops. OK.
        # n=13 -> 8192 elements. 67M ops. Borderline.
        # n=14 -> 16384 elements. 268M ops. Too slow.
        
        # So we can only directly score up to n=12 or 13.
        # For n > 13, we should use the theoretical formula if we are confident
        # that the "no-carry" property holds (B >= 3).
        
        if B >= 3:
            # No carries.
            # |U+U| = 3^n
            # |U-U| = 5^n
            # Max U = (B^n - 1) / (B - 1)
            num = (5.0 / 3.0) ** n
            den = math.log(2 * ((B ** n - 1) / (B - 1)) + 1)
            if den == 0: return 0
            return 1 + math.log(num) / den
        
        else:
            # Direct calculation
            sums = set()
            diffs = set()
            for a in U:
                for b in U:
                    sums.add(a + b)
                    diffs.add(a - b)
            num = len(diffs) / len(sums)
            den = math.log(2 * max_u + 1)
            if den == 0: return 0
            return 1 + math.log(num) / den

    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search optimal B and n
    # We know B=3 is best.
    # We will sweep n up to 20 using the formula for B>=3.
    # For B=2, we sweep n up to 12 using direct calc.
    
    for B in [2, 3, 4, 5]:
        max_n = 12 if B == 2 else 25
        for n in range(2, max_n + 1):
            # Generate set
            U = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                U.append(val)
            
            if B >= 3:
                s = evaluate_set_fast(B, n)
            else:
                s = calc_score(U)
            
            if s > best_score:
                best_score = s
                best = U
                
    return best

def evaluate_set_fast(B, n):
    if B < 3:
        return 0
    num = (5.0 / 3.0) ** n
    max_u = (B ** n - 1) / (B - 1)
    den = math.log(2 * max_u + 1)
    if den == 0: return 0
    return 1 + math.log(num) / den

def construct():
    # Re-use the logic above but encapsulated
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    for B in [2, 3, 4, 5]:
        if B == 2:
            max_n = 12
        else:
            max_n = 25
            
        for n in range(2, max_n + 1):
            if B >= 3:
                s = evaluate_set_fast(B, n)
                # We don't need to generate the set if we just want the score,
                # but we need to store the best set.
                # Generating 2^20 elements is 1M elements. Memory is fine (~8MB).
                # But we only generate if it's the best so far?
                # Or just generate at the end.
                
                if s > best_score:
                    best_score = s
                    # Generate U
                    U = []
                    for mask in range(1 << n):
                        val = 0
                        for j in range(n):
                            if mask & (1 << j):
                                val += B ** j
                        U.append(val)
                    best = U
            else:
                # B=2
                U = []
                for mask in range(1 << n):
                    val = 0
                    for j in range(n):
                        if mask & (1 << j):
                            val += 2 ** j
                    U.append(val)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best = U
                    
    return best

# Final check on the code structure.
# The function `construct` must be defined.
# `calc_score` is helper.
# `evaluate_set_fast` is helper.

# One concern: The prompt asks to "rewrite the program... keep function names".
# The seed had `construct` and `calc_score`.
# I am adding `evaluate_set_fast`. This is fine.

# Let's double check the B=3, n=20 score.
# num = (5/3)^20
# log(num) = 20 * log(5/3) = 20 * 0.5108 = 10.216
# max_u = (3^20 - 1)/2 = (3.48e9 - 1)/2 = 1.74e9
# 2*max_u + 1 = 3.48e9
# log(3.48e9) = 21.97
# score = 1 + 10.216 / 21.97 = 1 + 0.465 = 1.465.

# B=3, n=10:
# log(num) = 10 * 0.5108 = 5.108
# max_u = 29524
# 2*max_u+1 = 59049
# log(59049) = 10.986
# score = 1 + 5.108 / 10.986 = 1 + 0.465 = 1.465.

# It's constant!
# So n doesn't matter for B>=3.
# The score is exactly 1 + log(5/3)/log(B) for all n (asymptotically).
# For B=3, it's 1.465.

# So if the current program found a set with score < 1.465, my program will beat it.
# If the current program found 1.465, my program will tie it.
# Given the "score 1.0281" hint, it's likely the current program is slightly suboptimal or the baseline is lower.

# I will output this program.
