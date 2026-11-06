import time
import random

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 90s kill

    # Precompute base-3 candidates (numbers with no digit '2' in base-3)
    # These are 3-AP-free, a strong starting point
    base3 = []
    for i in range(1, 513):
        n = i
        ok = True
        while n > 0:
            if n % 3 == 2:
                ok = False
                break
            n //= 3
        if ok:
            base3.append(i)
    base3.sort()

    def compute_dp(S):
        """Compute subset-sum DP for set S.
        Returns dp[k] = bitset of achievable sums using exactly k elements from S."""
        m = len(S)
        max_sum = sum(S)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp, max_sum

    def check_conflicts(S):
        """Check if any element in S is the mean of a subset of >=2 OTHER elements.
        Uses global DP: for each x, check if k*x is achievable by k elements from S\{x}
        for some k >= 2.
        
        Key insight: instead of recomputing DP for S\{x} for each x,
        we compute DP for S once, then use the fact that:
        sum of k elements from S\{x} = k*x  iff  (k+1) elements from S sum to k*x + x = (k+1)*x
        ... no, that's not quite right. Let me think differently.
        
        Actually, we need: does there exist a subset T of S\{x} with |T|>=2 and sum(T) = |T|*x?
        This is equivalent to: does there exist a subset U of S with x in U, |U|>=3, 
        and sum(U) = |U|*x? No, that's not equivalent either.
        
        Let me just do it properly: for each x in S, check if x is mean of subset of S\{x}.
        We can precompute DP for the full set S, and for each x, we need DP of S\{x}.
        That's still expensive. 
        
        Alternative: compute DP for full set S. For each x, the subsets of S\{x} are 
        exactly the subsets of S that don't contain x. We can't easily extract those.
        
        Let's use a different approach: for each x in S, compute DP of S\{x} but cache it.
        Actually, with n=48, S\{x} has 47 elements. DP for 47 elements: 48 bitsets, each up to 47*512 bits.
        That's manageable. The total work is 48 * 47 * (48*512/word_size) operations.
        
        Let me just do the per-element DP but optimize it.
        """
        n = len(S)
        if n < 3:
            return True
        
        S_sorted = sorted(S)
        
        for idx, x in enumerate(S_sorted):
            others = S_sorted[:idx] + S_sorted[idx+1:]
            m = len(others)
            if m < 2:
                continue
            
            # DP for others
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << val)
            
            # Check if x is mean of any subset of size k>=2 from others
            for k in range(2, m + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def can_add(S, new_elem):
        """Check if we can add new_elem to S without creating conflicts."""
        new_S = S + [new_elem]
        return check_conflicts(new_S)

    # Strategy: Greedy with multiple orderings, then local search
    best_set = []
    
    # Try ascending greedy on base3
    S = []
    for x in base3:
        if len(S) >= 48:
            break
        if time.time() > deadline:
            break
        if can_add(S, x):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
    if len(best_set) >= 48:
        return best_set

    # Try descending greedy
    if time.time() < deadline:
        S = []
        for x in reversed(base3):
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
        if len(best_set) >= 48:
            return best_set

    # Try several random shuffles of base3
    random.seed(42)
    for trial in range(30):
        if time.time() >= deadline:
            break
        cands = base3[:]
        random.shuffle(cands)
        S = []
        for x in cands:
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
        if len(best_set) >= 48:
            return best_set

    # Try greedy on all numbers with various orderings
    for ordering in ['ascending', 'descending']:
        if time.time() >= deadline:
            break
        if ordering == 'ascending':
            cands = list(range(1, 513))
        else:
            cands = list(range(512, 0, -1))
        S = []
        for x in cands:
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
        if len(best_set) >= 48:
            return best_set

    # Local search: try to improve by adding from pool
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        # Try all numbers not in best_set
        pool = [x for x in range(1, 513) if x not in used]
        random.shuffle(pool)
        
        improved = True
        while improved and len(best_set) < 48 and time.time() < deadline:
            improved = False
            for x in pool:
                if len(best_set) >= 48 or time.time() > deadline:
                    break
                if can_add(best_set, x):
                    best_set.append(x)
                    pool.remove(x)
                    improved = True
                    break

    # Try swaps to potentially allow more additions
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        pool = [x for x in range(1, 513) if x not in used]
        random.shuffle(pool)
        
        for _ in range(50):
            if len(best_set) >= 48 or time.time() > deadline:
                break
            # Try swapping one element out and two in, or one out one in
            swapped = False
            for x_new in pool[:50]:
                if time.time() > deadline:
                    break
                for j, x_old in enumerate(best_set):
                    new_S = best_set[:j] + best_set[j+1:] + [x_new]
                    if can_add(new_S, x_new):
                        # Check if this swap allows us to add more
                        test_S = new_S + [x_new]
                        if can_add(new_S, x_new):
                            best_set = new_S
                            if x_old in pool:
                                pool.remove(x_old)
                            if x_new in pool:
                                pool.remove(x_new)
                            swapped = True
                            break
                if swapped:
                    break

    # Final expansion
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        all_nums = [x for x in range(1, 513) if x not in used]
        random.shuffle(all_nums)
        for x in all_nums:
            if len(best_set) >= 48 or time.time() > deadline:
                break
            if can_add(best_set, x):
                best_set.append(x)

    # Final validation
    if check_conflicts(best_set):
        return best_set
    # If invalid, try to fix by removing problematic elements
    S_sorted = sorted(best_set)
    for i in range(len(S_sorted)):
        test = S_sorted[:i] + S_sorted[i+1:]
        if check_conflicts(test):
            best_set = test
            break
    return best_set

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s, Size: {len(res)}")
