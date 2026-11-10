import numpy as np

def can_add(x, S):
    if not S:
        return True
    
    S = list(S)
    n = len(S)
    
    # 1. Check if x is mean of subset of S
    # Subset size k >= 2, sum = k*x
    # Max sum of n elements is sum(S). Min sum of k elements is sum(k smallest).
    # We can do a single DP pass for all k?
    # dp[j][s] -> is it possible to get sum s with j elements from S?
    # Max sum is sum(S). Max j is n.
    # This might be too big if n is large (up to 48) and sum is ~ 48*512 ~ 24000.
    # 48 * 24000 booleans is ~ 1M bytes. Very fast.
    
    max_sum = sum(S)
    # dp[j] is a boolean array of size max_sum + 1
    # dp[j][s] is True if sum s is achievable with exactly j elements
    # We only care about s = k*x for k in 2..n
    # So we can restrict the sum check to specific values?
    # No, we need the full DP to know if a specific sum is reachable.
    
    # Optimization: We only need to check sums that are multiples of x?
    # No, the sum must be exactly k*x.
    # We can run the DP and check specific indices.
    
    # dp[j] = np.zeros(max_sum + 1, dtype=bool)
    # dp[0][0] = True
    # for val in S:
    #    for j in range(n, 0, -1):
    #       dp[j] |= (dp[j-1] shifted by val)
    
    # But we need to do this for EVERY x we try to add.
    # And S changes as we add elements.
    # Recalculating full DP for every candidate x is O(N * |S| * Sum(S)).
    # N=512, |S|~48, Sum(S)~24000.
    # 512 * 48 * 24000 ~ 500 million operations. NumPy can handle this in ~1s? Maybe.
    
    # Better: Incremental DP?
    # Hard to remove elements from DP easily.
    
    # Let's try the full DP check for each x.
    # If it's too slow, we fall back to checking only small k?
    # No, we must be correct.
    
    # Check 1: x is mean of subset of S
    # We need to know if there exists k>=2 such that sum of k elements in S is k*x.
    
    # Check 2: For any y in S, is y mean of subset of (S \ {y}) U {x}?
    # Subset T of (S \ {y}) U {x} contains x?
    # If T contains x, then T = {x} U T' where T' subset of S \ {y}.
    # Mean of T is y.
    # (x + sum(T')) / (1 + |T'|) = y
    # x + sum(T') = (1 + |T'|) * y
    # sum(T') = (1 + |T'|) * y - x
    # Let j = |T'|. Then sum of j elements from S \ {y} is (j+1)*y - x.
    # j can be 0 to |S|-1.
    # If j=0, sum=0, so (1)*y - x = 0 => x=y. But x not in S, so x!=y. So j>=1?
    # Wait, if T' is empty, T={x}. Mean is x. Is x=y? No.
    # So we need j >= 1.
    
    # So for each y in S, we check if there exists j in 1..|S|-1 such that
    # sum of j elements from S \ {y} is (j+1)*y - x.
    
    # This requires DP for S \ {y} for each y. That's |S| DPs.
    # Too slow.
    
    # Alternative:
    # Is there a faster way?
    # Maybe we can just check if the new set is valid by re-running a global validator?
    # No, we need incremental.
    
    # Let's optimize the NumPy DP.
    # For Check 1:
    # We can precompute the "subset sums" of S?
    # If we maintain the DP table for the current set S, we can check Check 1 in O(1) per k.
    # But we also need Check 2.
    
    # Let's assume we rebuild the DP for S every time we add an element.
    # Cost of building DP for S of size n: O(n * Sum(S)).
    # We do this once per successful addition.
    # Number of additions is ~48.
    # Total cost ~ 48 * 48 * 24000 ~ 55M ops. Very fast.
    
    # But for Check 2, we need to query "is (j+1)y - x a subset sum of S\{y} with size j?".
    # We don't have DP for S\{y}.
    # We can approximate or use a slower check for Check 2?
    # Or we can just try to add x, and if it fails, we know why?
    
    # Actually, if we maintain the DP for S, we can check Check 1 easily.
    # For Check 2, we can iterate y in S.
    # For each y, we need to know if `(j+1)y - x` is a subset sum of `S \ {y}` with size `j`.
    # This is hard without DP for `S \ {y}`.
    
    # However, note that `S \ {y}` is almost `S`.
    # If we have the DP for `S`, can we derive DP for `S \ {y}`?
    # Not easily.
    
    # Let's try a different approach for Check 2.
    # Instead of checking all y, maybe we can just check if the new set is valid using a global check on the new set?
    # No, that's expensive.
    
    # Let's go back to the Python set DP but optimized.
    # The previous code was slow because of Python loops.
    # NumPy DP for Check 1 is fast.
    # For Check 2, we can use a necessary condition:
    # If y is the mean of a subset involving x, then x must be "close" to y in some sense?
    
    # Let's try this:
    # 1. Maintain DP for current set S.
    # 2. To check if x can be added:
    #    a. Check if x is mean of subset of S (using DP of S).
    #    b. For each y in S, check if y can be mean of subset of S' = S U {x} \ {y}.
    #       This requires checking if `(j+1)y - x` is a subset sum of `S \ {y}` with size `j`.
    #       We can approximate this by checking if `(j+1)y - x` is a subset sum of `S` with size `j` or `j+1`?
    #       No, it must be exact.
    
    # Given the time limit, maybe we can just use the Python set DP but with a tighter bound?
    # Or maybe we can use a randomized check? No, must be exact.
    
    # Let's try to speed up the Python set DP by using `frozenset` or similar?
    # Or just use the NumPy DP for the whole new set?
    # If we add x, we get S_new. We can build DP for S_new.
    # Then we check if any element in S_new is the mean of a subset of others.
    # This is O(|S_new| * (DP_cost)).
    # DP_cost is O(|S_new| * Sum(S_new)).
    # Total O(|S_new|^2 * Sum).
    # 48^2 * 24000 ~ 55M.
    # We do this for each candidate x? No, only if we decide to add it?
    # But we need to decide IF we can add it.
    
    # Strategy:
    # 1. Try to add x.
    # 2. If Check 1 (x is mean of S) passes, proceed.
    # 3. For Check 2, we perform a "local" DP for each y?
    #    This is too slow.
    
    # Alternative:
    # Just build the DP for S_new and verify the whole set?
    # If it's valid, add x. If not, discard x.
    # Cost: One full validation of S_new.
    # Validation cost: For each y in S_new, check if y is mean of subset of S_new \ {y}.
    # This requires DP for S_new \ {y} for each y.
    # That's |S_new| DPs.
    # Too slow.
    
    # Let's stick to the incremental check but optimize the Python code.
    # The previous code used `list(sums_by_count[j-1])` which is slow.
    # Using `numpy` for the inner loop might help.
    
    # Let's try a hybrid:
    # Check 1: NumPy DP on S. Fast.
    # Check 2: For each y in S, use a fast heuristic or a limited DP.
    # Actually, if Check 1 passes, it's rare that Check 2 fails for "random" x.
    # We can just try to add x, and then run a full validation on the new set?
    # If the full validation is slow, we can't do it for every x.
    
    # Let's assume the greedy approach with a fast Check 1 and a slow Check 2 (only triggered if Check 1 passes) is sufficient.
    # Most x will fail Check 1?
    # No, many x will pass Check 1.
    
    # Let's use the NumPy DP for Check 1.
    # For Check 2, we will use the Python set DP but only for the specific y's that are "close" to x?
    # No, must be exact.
    
    # Okay, I will implement the NumPy DP for the *entire* validation of a candidate set.
    # I will use a greedy strategy that tries to add x, and if it looks promising (Check 1 passes), it runs a full validation of the new set.
    # To make full validation fast, I will use NumPy.
    # Full validation of set S of size n:
    # For each y in S:
    #    Compute DP for S \ {y}.
    #    Check if y is mean of any subset.
    # This is O(n^2 * Sum).
    # For n=48, this is ~55M ops.
    # If we do this for 512 candidates, it's 512 * 55M = 28B ops. Too slow.
    
    # So we must use incremental checks.
    # I will use the Python set DP but optimize it heavily.
    # Or, I will use a randomized local search with a fast approximate check, and then verify the final set exactly.
    
    # Plan:
    # 1. Greedy construction using a fast approximate check (e.g., only check k=2,3,4).
    # 2. Once a set is built, verify it exactly using NumPy.
    # 3. If valid, try to add more elements using the exact check (slow, but only a few times).
    # 4. If invalid, repair it.
    
    # This seems risky.
    
    # Let's try one more thing:
    # The constraint is symmetric.
    # I will implement the `can_add` using NumPy for Check 1.
    # For Check 2, I will iterate y in S and use a NumPy DP for `S \ {y}`?
    # No, building DP for `S \ {y}` for each y is too slow.
    
    # I will use the Python set DP but with `array` module or `bitarray`?
    # `bitarray` is not standard.
    
    # Let's just use the Python set DP from the previous program but with a much tighter time limit and better ordering.
    # The previous program scored 1.46.
    # I will add a local search step that swaps elements.
    
    # Wait, the previous program's `can_add` was the bottleneck.
    # I will replace it with a NumPy-based check for Check 1.
    # And for Check 2, I will use a simple heuristic: if `x` is far from `y`, it's unlikely to be the mean.
    # Specifically, if `|x - y| > max(S)`, it's impossible?
    # No.
    
    # Let's try this:
    # 1. Use NumPy DP for Check 1.
    # 2. For Check 2, use a fast Python check that only considers small subset sizes (k=2,3,4).
    # 3. If the set is built, do a full exact verification.
    # 4. If verification fails, remove the problematic element and continue.
    
    # This is a good compromise.

