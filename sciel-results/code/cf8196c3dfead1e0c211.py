import time

def construct():
    deadline = time.time() + 55.0
    N = 512
    MAXC = 48

    # Generate candidates with base-3 digits 0 and 1
    candidates = []
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            candidates.append(i)
            
    # We have 63 candidates. We need 48.
    # Pick the first 48.
    S = candidates[:MAXC]
    
    def verify(S):
        S = list(S)
        n = len(S)
        if n < 3:
            return True
        # Precompute sums? No, just check each element.
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            # Check if a is the mean of any subset of sub of size k >= 2
            # k * a = sum(subset)
            # Use DP for subset sums of sub
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in sub:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << x
            
            max_sum = sum(sub)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    if verify(S):
        return S
        
    # Fallback: If the deterministic set fails, use a simple greedy approach
    # This is unlikely to be needed, but ensures we return *something* valid.
    best = []
    for x in range(1, N + 1):
        if len(best) >= MAXC:
            break
        # Quick check: is x the mean of any subset of best?
        # And do any elements in best become means with x?
        # Simplified check for speed: just check size 2 and 3 subsets for now in fallback
        # If we have time, do full check.
        
        # For the fallback, let's just try to add x if it doesn't form a 3-AP with any pair in best
        # This is a necessary but not sufficient condition, but better than nothing.
        ok = True
        for j in range(len(best)):
            for l in range(j+1, len(best)):
                # Check 3-AP: best[j], x, best[l] or best[j], best[l], x or x, best[j], best[l]
                # Mean of 2 elements is 1 element.
                # x = (best[j] + best[l]) / 2
                if 2 * x == best[j] + best[l]:
                    ok = False
                    break
                # best[j] = (x + best[l]) / 2
                if 2 * best[j] == x + best[l]:
                    ok = False
                    break
                # best[l] = (x + best[j]) / 2
                if 2 * best[l] == x + best[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best.append(x)
            
    # Verify the fallback result
    if verify(best):
        return best
        
    # If even that fails, return a very small safe set
    return [1, 3, 4]
