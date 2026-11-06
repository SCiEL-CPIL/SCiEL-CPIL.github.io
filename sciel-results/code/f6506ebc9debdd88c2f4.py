"""Rewritten program: targeted local search with critical-AP pruning and deadline guard."""
import time
import random
import numpy as np

N = 256

def _build_critical_aps():
    """
    Identify a reduced set of 'critical' arithmetic progressions.
    We include:
      - All singletons (length 1) -> max abs sum is always 1, so we can skip them.
        Actually, singleton sum is just the sign, abs=1. So max over singletons is 1.
        We can ignore them if we know the max is at least 1.
      - Long APs: for each step s, include APs of length >= L_min where L_min is chosen
        so that the number of such APs is manageable.
      - Dense short APs: for small steps (1,2,3,...) include all APs of length >= 2.
    
    To keep evaluation fast, we precompute for each critical AP:
      - start index (0-based)
      - step
      - length
    and store them in a list.
    
    We aim for a set that is large enough to capture the max but small enough to evaluate quickly.
    """
    aps = []
    
    # For step 1: all contiguous subarrays. There are N*(N+1)/2 = 32896. That's a lot but manageable.
    # However, we can note that for step 1, the max abs sum over all contiguous subarrays is the
    # maximum subarray sum problem. But we need to check all of them.
    # To reduce, we can include only those with length >= 2? No, we need all to be safe.
    # But 32896 APs for step 1 alone is heavy. Let's think differently.
    
    # Alternative: For step 1, the max absolute sum over contiguous subarrays can be computed
    # in O(N) using Kadane's algorithm variant for max absolute sum. But the task requires
    # minimizing the maximum over ALL APs, so we need to consider all.
    
    # Let's include:
    # 1. All APs with step 1 and length >= 1 (all contiguous) -> too many.
    # Instead, let's be smarter: 
    #   - For step 1, the worst contiguous subarray is likely long. But we can't skip short ones.
    #   - However, for a sign sequence, the max contiguous sum is bounded by N. 
    #   - We can compute the max contiguous sum efficiently during evaluation without enumerating all.
    
    # Revised strategy:
    # Precompute a list of critical APs that are NOT step-1 contiguous. For step-1, use a fast
    # O(N) computation. For other steps, enumerate APs with length >= 2 (skip singletons).
    
    # For step s >= 2:
    #   Number of APs with step s and length L: N - s*(L-1)
    #   We'll include all APs with step s in [2, N//2] and length >= 2.
    #   But that's still a lot. Let's limit to steps s where s <= S_MAX and length >= L_MIN.
    
    # Let's choose S_MAX = 16, L_MIN = 2 for s >= 2.
    # For s=2: number of APs with length >= 2 is sum_{L=2}^{128} (256 - 2*(L-1)) = sum_{k=1}^{127} (256 - 2k) = 127*256 - 2*127*128/2 = 32512 - 16256 = 16256.
    # That's too many.
    
    # Let's be more selective. Include:
    # - For s=1: handle separately with O(N) max contiguous sum.
    # - For s=2: include APs with length >= 4 (to reduce count).
    # - For s=3..8: include APs with length >= 3.
    # - For s=9..32: include APs with length >= 2.
    # - For s>32: skip (these have very few elements, max sum is small).
    
    # Actually, let's just include a moderate set and see.
    
    # For now, let's include:
    # - s=1: special handling
    # - s=2: all APs with length >= 2
    # - s=3: all APs with length >= 2
    # - s=4..8: all APs with length >= 2
    # - s=9..16: all APs with length >= 2
    # - s=17..32: all APs with length >= 2
    # - s>32: skip
    
    # Count for s=2: 16256 (as above)
    # s=3: sum_{L=2}^{85} (256 - 3*(L-1)) = sum_{k=1}^{84} (256 - 3k) = 84*256 - 3*84*85/2 = 21504 - 10710 = 10794
    # This is getting large. Total would be ~50k APs. Evaluating 50k APs per candidate is too slow.
    
    # Better approach: Don't precompute all APs. Instead, during evaluation, compute the max
    # over a reduced set of "representative" APs.
    
    # Let's use a different strategy: 
    # 1. Compute max contiguous sum (step 1) in O(N).
    # 2. For steps 2..S_MAX, compute the max over all APs with that step in O(N^2/s) per step,
    #    but only for steps up to S_MAX=8 or so.
    
    # Actually, for a fixed step s, the APs partition the array into s residue classes.
    # For each residue r in [0, s-1], the elements at positions r, r+s, r+2s, ... form a contiguous
    # subarray in that residue class. So the max AP sum for step s is the max over all r of the
    # max contiguous subarray sum in the subsequence x[r::s].
    
    # This is O(N) per step! Because for each residue class of length ~N/s, computing max contiguous
    # sum is O(N/s), and there are s residue classes, so total O(N) per step.
    
    # So we can compute the max over all APs for a given step s in O(N).
    # For all steps s=1..S_MAX, total O(N*S_MAX).
    
    # For s > S_MAX, the APs are short (length <= N/s < N/S_MAX). The max sum for such APs is at
    # most N/s. If we choose S_MAX such that N/S_MAX is small (say <= 4), then we can skip those
    # or handle them differently.
    
    # Let's choose S_MAX = 64. Then for s > 64, max AP length is 256/65 = 3 (approx). So max sum <= 3.
    # We can compute the max for s > S_MAX by checking all APs of length 2 and 3 for those steps,
    # but that's O(N^2) which is too slow.
    
    # Alternative: For s > S_MAX, the max possible sum is floor(N/s) + 1. If this is <= current best,
    # we can skip. Otherwise, we need to check.
    
    # For now, let's compute max for s=1..S_MAX using the residue-class method, and for s > S_MAX,
    # we'll compute the max by checking all APs of length 2 (since length 3+ would have sum <= 3,
    # and if our current best is > 3, we can skip).
    
    # Actually, let's just compute for s=1..S_MAX with S_MAX=128, and for s>128, the max AP length
    # is 2, so max sum is 2. We can check all length-2 APs for s>128 in O(N^2) which is 65536,
    # but that's too slow per evaluation.
    
    # Let's simplify: compute max for s=1..64 using residue method (O(N*64) = 16384 ops).
    # For s>64, max AP length is 3 (for s=85, length=3). So max sum <= 3.
    # We'll assume that if our best so far is > 3, we can skip s>64. If best <= 3, we need to check.
    # But best is likely > 3 for random signs.
    
    # Let's implement this.
    
    pass  # We'll implement the evaluation function directly.

def _max_contiguous_sum(x):
    """Compute max absolute contiguous subarray sum."""
    # Kadane's algorithm for max subarray sum, and min subarray sum.
    max_ending = x[0]
    min_ending = x[0]
    max_so_far = x[0]
    min_so_far = x[0]
    for i in range(1, len(x)):
        max_ending = max(x[i], max_ending + x[i])
        min_ending = min(x[i], min_ending + x[i])
        max_so_far = max(max_so_far, max_ending)
        min_so_far = min(min_so_far, min_ending)
    return max(abs(max_so_far), abs(min_so_far))

def _max_ap_sum_for_step(x, s):
    """
    Compute max absolute sum over all APs with step s.
    Uses residue class decomposition.
    """
    n = len(x)
    max_sum = 0
    for r in range(s):
        # Extract subsequence x[r], x[r+s], x[r+2s], ...
        subseq = x[r::s]
        if len(subseq) == 0:
            continue
        max_c = _max_contiguous_sum(subseq)
        if max_c > max_sum:
            max_sum = max_c
    return max_sum

def _eval(x, s_max=64):
    """
    Evaluate the maximum absolute sum over all APs.
    x is a numpy array of signs.
    """
    n = len(x)
    max_sum = 0
    
    # Steps 1 to s_max
    for s in range(1, s_max + 1):
        ms = _max_ap_sum_for_step(x, s)
        if ms > max_sum:
            max_sum = ms
    
    # For steps > s_max, the max AP length is floor(n/s) which is <= floor(n/(s_max+1)).
    # For s_max=64, n=256, floor(256/65)=3. So max sum <= 3.
    # If max_sum >= 3, we can skip. Otherwise, we need to check.
    if max_sum < 3:
        # Check all APs with step > s_max
        for s in range(s_max + 1, n):
            # For this step, max length is floor((n-1)/s) + 1
            max_len = (n - 1) // s + 1
            if max_len <= 1:
                continue
            # Check all APs with this step
            for r in range(s):
                # Extract subsequence
                subseq = x[r::s]
                if len(subseq) <= 1:
                    continue
                ms = _max_contiguous_sum(subseq)
                if ms > max_sum:
                    max_sum = ms
                    if max_sum >= 3:
                        return max_sum  # Early exit
    
    return max_sum

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    best_x = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Initial random solution
    x = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    score = _eval(x)
    best_x = x.copy()
    best_score = score
    
    # Local search: flip one bit at a time
    for i in range(N):
        if time.time() > deadline:
            break
        x[i] = -x[i]
        score = _eval(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        else:
            x[i] = -x[i]  # revert
    
    # Second pass
    for i in range(N - 1, -1, -1):
        if time.time() > deadline:
            break
        x[i] = -x[i]
        score = _eval(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        else:
            x[i] = -x[i]  # revert
    
    # Restart with different random seeds
    for seed in range(1, 100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        x = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        score = _eval(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Local search
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            score = _eval(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
            else:
                x[i] = -x[i]  # revert
        
        # Second pass
        for i in range(N - 1, -1, -1):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            score = _eval(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
            else:
                x[i] = -x[i]  # revert
    
    return list(best_x)
