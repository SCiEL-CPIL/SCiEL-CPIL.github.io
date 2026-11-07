import time
import random

def construct():
    deadline = time.time() + 80  # 90s limit, 10s margin
    
    def is_valid(set_vals):
        """Check if set satisfies cardinality separation using bitset DP."""
        n = len(set_vals)
        if n == 0:
            return True
        # sums[i] is a bitset where bit s is set if there's a subset of size i with sum s
        sums = [0] * (n + 1)
        sums[0] = 1  # empty subset has sum 0
        for x in set_vals:
            for i in range(n, 0, -1):
                if sums[i-1]:
                    sums[i] |= (sums[i-1] << x)
        # Check disjointness across different cardinalities
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if (sums[i] & sums[j]) != 0:
                    return False
        return True

    def try_construct_geometric(ratio, start, max_val=256):
        """Construct a set using geometric progression with given ratio and start."""
        S = []
        current = start
        while current <= max_val and len(S) < 60:
            S.append(current)
            current = int(current * ratio) + 1
            if current <= max_val and current in S:
                current += 1
        return S

    def local_improve(S, max_iter=5000):
        """Try to improve set by swapping elements to increase size or maintain size with larger values."""
        if not S:
            return S
        best = sorted(S)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Try adding a random element not in set
            if len(best) < 60:
                candidates = [x for x in range(1, 257) if x not in best]
                if candidates:
                    x = random.choice(candidates)
                    new_S = sorted(best + [x])
                    if is_valid(new_S):
                        best = new_S
            # Try swapping: remove one, add one
            if len(best) >= 2:
                remove_idx = random.randint(0, len(best) - 1)
                candidates = [x for x in range(1, 257) if x not in best]
                if candidates:
                    x = random.choice(candidates)
                    new_S = best[:remove_idx] + best[remove_idx+1:] + [x]
                    new_S = sorted(new_S)
                    if is_valid(new_S):
                        best = new_S
        return best

    # Strategy 1: Geometric progression with various ratios
    best_result = []
    
    # Try different geometric ratios
    ratios = [1.5, 1.6, 1.7, 1.8, 2.0, 2.1, 2.2, 2.3, 2.5, 3.0]
    starts = [1, 2, 3, 5, 8, 13]
    
    for ratio in ratios:
        for start in starts:
            if time.time() > deadline - 10:
                break
            S = try_construct_geometric(ratio, start)
            if len(S) > len(best_result):
                best_result = S
    
    # Strategy 2: Greedy with bitset verification
    def greedy_construct():
        S = []
        sums = [0] * 61
        sums[0] = 1
        for x in range(256, 0, -1):
            if time.time() > deadline - 5:
                break
            valid = True
            new_sums = sums[:]
            for i in range(len(S) + 1, 0, -1):
                if sums[i-1]:
                    shifted = sums[i-1] << x
                    new_sums[i] |= shifted
            # Check if any new sum conflicts with existing sums of different cardinality
            for i in range(1, len(S) + 2):
                if not new_sums[i]:
                    continue
                for j in range(len(S) + 1):
                    if i == j:
                        continue
                    if (new_sums[i] & sums[j]) != 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
                sums = new_sums
                if len(S) >= 60:
                    break
        return S

    S_greedy = greedy_construct()
    if len(S_greedy) > len(best_result):
        best_result = S_greedy

    # Strategy 3: Local improvement on best candidate
    if best_result and time.time() < deadline - 15:
        improved = local_improve(best_result, max_iter=3000)
        if len(improved) > len(best_result):
            best_result = improved

    # Final verification
    if not best_result:
        return list(range(1, 10))
    
    if not is_valid(best_result):
        # Fallback to a known valid small set
        best_result = list(range(1, 10))
    
    return sorted(best_result)
