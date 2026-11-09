import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def generate_square_free(limit):
        """Sieve for square-free numbers up to limit."""
        if limit < 1:
            return []
        is_sq_free = [True] * (limit + 1)
        is_sq_free[0] = False
        # Mark multiples of p^2 as not square-free
        p = 2
        while p * p <= limit:
            if is_sq_free[p]:
                sq = p * p
                for j in range(sq, limit + 1, sq):
                    is_sq_free[j] = False
            p += 1
        return [i for i in range(1, limit + 1) if is_sq_free[i]]

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = generate_square_free(max_v)
        
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

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    # Try limits in descending order of size to maximize potential pairs
    # M=10^8 -> |P|~10k, P^2=10^8 sums, ~800MB memory. Safe.
    # M=5*10^8 -> |P|~22k, P^2=5*10^8 sums, ~4GB memory. Risky.
    # Let's try 2*10^8 first as a balance.
    limits_to_try = [200_000_000, 100_000_000, 50_000_000]
    
    for M in limits_to_try:
        if time.time() > deadline - 5:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            if L < 10:
                continue

            P_np = np.array(P, dtype=np.int64)
            
            # Check memory estimate
            # L * L * 8 bytes
            if L * L * 8 > 4 * 1024**3: # 4GB limit
                continue

            sums = np.add.outer(P_np, P_np)
            max_sum_val = sums.max()
            
            # bincount is efficient but creates a large array
            # minlength=max_sum_val+1 can be huge if M is large
            # However, max_sum_val is approx 2*M.
            # For M=2*10^8, array size is 4*10^8 ints. 3.2GB.
            # This is very large.
            
            # Alternative: Use a dictionary or sparse counting?
            # Or just reduce M further.
            # Let's try M=50_000_000. |P|~7000. P^2=5*10^7. 400MB. Safe.
            # Let's try M=100_000_000. |P|~10000. P^2=10^8. 800MB. Safe.
            
            # If we use M=100_000_000:
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            # Calculate distinct pairs
            P_set = set(P)
            pairs = []
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            current_score = len(pairs)
            if current_score > best_score:
                best_score = current_score
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we found a good score, we might not need to try smaller M
            # unless the larger M failed or was slow.
            # But larger M usually gives more pairs.
            # If M=200M worked, it's likely better than 100M.
            # So we return immediately if the largest feasible M succeeds.
            if M == limits_to_try[0]:
                return best_result

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result
