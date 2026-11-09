import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def is_square_free(n):
        if n < 2:
            return True
        r = math.isqrt(n)
        for p in range(2, r + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit.
        A powerful number is of the form u^2 * v^3 where v is square-free.
        """
        powerful = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
        
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            # u^2 * v3 <= limit => u <= sqrt(limit / v3)
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
                else:
                    break
        return sorted(powerful)

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1

    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Compute all pair sums a+b where a,b in P and a<=b.
    # 3. Find the sum N with the maximum number of representations.
    # 4. The bottleneck is memory for storing sums. We use a sparse approach or chunked bincount.
    
    # We will try increasing limits until we run out of time or memory.
    # Let's estimate the size of P for a given limit.
    # For L = 10^9, |P| is roughly a few thousand?
    # Let's just try and catch MemoryError.
    
    # To be safe and fast, we'll try a specific limit that is known to work well.
    # Based on previous scores, 5.8 suggests we are finding N with ~5-6 pairs.
    # Let's try to push the limit higher to find N with more pairs.
    
    # We'll use a list of candidate limits, starting from high to low.
    # If a high limit causes MemoryError, we drop to the next.
    
    limits = [200_000_000, 100_000_000, 50_000_000, 20_000_000, 10_000_000]
    
    for L in limits:
        if time.time() > deadline - 20:
            break
            
        try:
            P = generate_powerful(L)
            P_set = set(P)
            L_size = len(P)
            
            # If L_size is too large, the outer sum will be too big.
            # L_size^2 * 8 bytes.
            # If L_size = 10000, 10^8 * 8 = 800MB. OK.
            # If L_size = 50000, 2.5*10^9 * 8 = 20GB. Fail.
            # So we cap L_size at ~20000 for safety (3.2GB).
            if L_size > 20000:
                # Try to reduce L? Or just skip this L.
                # Since we are going high to low, we skip to smaller L.
                continue

            # Convert to numpy array for fast vectorized sum
            P_np = np.array(P, dtype=np.int64)
            
            # We only need a <= b, so we can generate upper triangle sums?
            # np.add.outer generates full matrix.
            # To save memory, we could generate only i <= j.
            # But np.add.outer is fast. Let's try it.
            # If memory fails, we catch it.
            
            sums = np.add.outer(P_np, P_np)
            
            # We only care about sums where a <= b.
            # The full matrix has both (a,b) and (b,a).
            # We want to count distinct pairs {a,b}.
            # If a != b, the pair is counted twice in the full sum count.
            # If a == b, it is counted once.
            # So for a given sum N:
            # Let C be the count of (a,b) in P x P such that a+b=N.
            # Let D be 1 if N/2 is in P (and N is even), else 0.
            # The number of distinct pairs is (C + D) / 2.
            
            # We need to find N that maximizes (C(N) + D(N)) / 2.
            # Since D(N) is 0 or 1, maximizing C(N) is a good proxy,
            # but we should check the final count.
            
            # Use bincount to get C(N) for all N.
            max_sum = sums.max()
            # bincount requires non-negative integers.
            # sums are up to 2*L.
            # If 2*L is too large for bincount array size, we might have issues.
            # For L=200M, max_sum=400M. Array of size 400M is 1.6GB (int64) or 800MB (int32).
            # np.bincount returns int64 by default? No, it returns the same dtype as input?
            # Actually, np.bincount returns int64 on 64-bit systems usually, or int32?
            # Let's check. It says "Returns an array of length max(max+1, minlength)".
            # The dtype is usually int64.
            # 400M * 8 bytes = 3.2GB. Plus sums (1.6GB for 20k^2/2? No, full matrix is 20k^2 = 4*10^8 * 8 = 3.2GB).
            # Total ~6.4GB. Might be tight.
            
            # To be safer, let's use a dictionary or a smaller L if we suspect memory issues.
            # Or, we can just try and catch MemoryError.
            
            counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
            
            # Find the index with max count
            # But we need to consider the (C+D)/2 factor.
            # Let's find the top few candidates and check them.
            # Sorting the entire counts array is O(N log N) which is slow for 400M.
            # We can just take argmax.
            # But what if the max C is at an odd N, and a nearby even N has C-1 but D=1, so (C-1+1)/2 = C/2 vs (C+0)/2 = C/2. Tie.
            # Or (C+1)/2 > C/2.
            # So we should check the top K sums.
            
            # However, for large L, the max count is likely small (like 5-10).
            # So many N might have the same max count.
            # We want the one with max (C+D)/2.
            
            # Let's get the max count value.
            max_C = counts.max()
            
            # Find all indices where count == max_C
            # This might be many.
            # Let's just pick the first one? Or iterate through a few?
            
            # A better approach:
            # Iterate through the powerful numbers and count pairs for a specific N?
            # No, we want to find the best N.
            
            # Let's assume the N with max C is the best, or close to it.
            # We can refine by checking a few candidates.
            
            # Get the indices of max count
            # This can be memory intensive if there are many.
            # Let's just use argmax to get one candidate.
            best_N = int(np.argmax(counts))
            C = int(counts[best_N])
            
            # Check if there's a better one nearby?
            # Or just calculate the score for this N.
            D = 1 if (best_N % 2 == 0 and (best_N // 2) in P_set) else 0
            num_pairs = (C + D) // 2
            
            if num_pairs > best_count:
                # Recover pairs
                pairs = []
                for a in P:
                    if a > best_N // 2:
                        break
                    b = best_N - a
                    if b in P_set:
                        pairs.append([a, b])
                best_result = {'N': best_N, 'pairs': pairs}
                best_count = num_pairs

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result

# Fallback if construct fails or times out?
# The function itself handles the deadline.

if __name__ == '__main__':
    res = construct()
    print(res)
