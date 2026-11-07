import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin

    def time_check():
        return time.time() < deadline

    N = 256

    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(N + 1)]
    for x in range(2, N + 1):
        n = x
        d = 2
        while d * d <= n:
            if n % d == 0:
                prime_factors[x].append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            prime_factors[x].append(n)

    # Get list of primes
    is_prime_arr = [True] * (N + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, N+1, i):
                is_prime_arr[j] = False
    primes_list = [i for i in range(2, N+1) if is_prime_arr[i]]
    num_primes = len(primes_list)
    prime_index = {p: i for i, p in enumerate(primes_list)}

    # Precompute prime mask for each number
    num_prime_mask = [0] * (N + 1)
    for x in range(2, N + 1):
        mask = 0
        for pf in prime_factors[x]:
            mask |= (1 << prime_index[pf])
        num_prime_mask[x] = mask

    # Precompute candidates for each prime, sorted by value descending
    # candidate[p] = list of (value, prime_mask)
    prime_candidates = {p: [] for p in primes_list}
    for p in primes_list:
        for x in range(p, N + 1, p):
            prime_candidates[p].append((x, num_prime_mask[x]))
        # Sort by value descending
        prime_candidates[p].sort(key=lambda t: -t[0])

    # Greedy initial solution to set a strong lower bound
    def greedy_solution():
        used_mask = 0
        total_sum = 1 # Always include 1
        chosen = [1]
        # Sort primes by some heuristic? Let's just iterate.
        # A better greedy: pick the number with highest value that doesn't conflict, remove its conflicts.
        # But here we are constrained by primes.
        # Let's use the B&B structure for greedy: for each prime, pick the best available.
        for p in primes_list:
            for x, pm in prime_candidates[p]:
                if pm & used_mask == 0:
                    used_mask |= pm
                    total_sum += x
                    chosen.append(x)
                    break
        return total_sum, chosen

    init_sum, init_chosen = greedy_solution()
    best_total = init_sum
    best_selection_mask = 0
    for x in init_chosen:
        best_selection_mask |= (1 << (x - 1))

    # Helper to calculate dynamic upper bound for remaining primes
    # This is expensive, so we might need to optimize or use a looser bound if too slow.
    # UB = sum_{q in remaining} max { x in candidates(q) | x compatible with used_mask }
    # If no candidate is compatible, contribution is 0.
    
    def calc_ub(prime_idx, used_mask):
        ub = 0
        for i in range(prime_idx, num_primes):
            p = primes_list[i]
            found = 0
            for x, pm in prime_candidates[p]:
                if pm & used_mask == 0:
                    found = x
                    break
            ub += found
        return ub

    import sys
    sys.setrecursionlimit(10000)

    nodes_visited = [0]

    def bb(prime_idx, used_mask, current_sum, chosen_mask):
        nonlocal best_total, best_selection_mask, nodes_visited
        nodes_visited[0] += 1
        
        if not time_check():
            return

        if prime_idx == num_primes:
            if current_sum > best_total:
                best_total = current_sum
                best_selection_mask = chosen_mask
            return

        # Upper Bound Check
        # If current_sum + max possible addition <= best_total, prune
        # We use a dynamic UB
        ub_rem = calc_ub(prime_idx, used_mask)
        if current_sum + ub_rem <= best_total:
            return

        p = primes_list[prime_idx]
        
        # Option 1: Skip this prime
        bb(prime_idx + 1, used_mask, current_sum, chosen_mask)

        # Option 2: Pick a candidate for this prime
        # We only consider top K candidates to limit branching factor
        # If the best candidate doesn't lead to a solution better than best_total, 
        # smaller ones are unlikely to, but they might unblock other primes? 
        # No, picking a smaller number uses the same prime p and potentially fewer other primes.
        # Using fewer other primes is "good" for compatibility with future primes.
        # So a smaller number might be better if it uses "cheaper" primes?
        # Actually, all primes are "equally" blocking. 
        # However, a number like 2*3=6 blocks 2 and 3. A number like 2*127=254 blocks 2 and 127.
        # If we pick 6, we block 3. If we pick 254, we block 127.
        # It's complex. Let's just try top 3 candidates.
        
        count = 0
        for x, pm in prime_candidates[p]:
            if count >= 3:
                break
            if pm & used_mask:
                continue
            
            new_used = used_mask | pm
            new_sum = current_sum + x
            new_chosen = chosen_mask | (1 << (x - 1))
            
            if new_sum > best_total:
                best_total = new_sum
                best_selection_mask = new_chosen
            
            # Prune if even with max remaining we can't beat best
            # Re-calc UB for next state
            # Optimization: we can pass a slightly tighter bound or just call calc_ub
            # calc_ub is the bottleneck.
            # Let's just call bb and let the first check in bb handle the UB.
            bb(prime_idx + 1, new_used, new_sum, new_chosen)
            count += 1

    # Start B&B
    # We always include 1.
    # Bit 0 in chosen_mask corresponds to number 1.
    initial_chosen_mask = 1
    bb(0, 0, 1, initial_chosen_mask)

    # Extract result
    result = []
    for i in range(N):
        if (best_selection_mask >> i) & 1:
            result.append(i + 1)
    
    # Ensure 1 is there (it should be, but just in case)
    if 1 not in result:
        result.append(1)
        
    result.sort()
    
    # Final Validation
    if len(set(result)) != len(result):
        return [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]

    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            return [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]

    return [int(x) for x in result]
