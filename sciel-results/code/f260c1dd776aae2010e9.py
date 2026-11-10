import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 512
    MAX_SIZE = 48
    
    # Precompute: for each value v (1..512), the set of pairs (i,j) with i<j, i+j=2v
    # conflict_pairs[v] = list of (i,j) pairs
    conflict_pairs = {}
    for v in range(1, N+1):
        pairs = []
        target = 2*v
        for i in range(1, target):
            j = target - i
            if j <= i:
                continue
            if j <= N:
                pairs.append((i, j))
        conflict_pairs[v] = pairs
    
    # For fast checking: given a set S (as a boolean array or set), check if v conflicts
    # v conflicts if there exists (i,j) in conflict_pairs[v] with i,j both in S and i!=v, j!=v
    
    def check_3ap(v, S):
        """Check if v forms a 3-AP with any two elements in S (excluding v itself)."""
        for (i, j) in conflict_pairs[v]:
            if i == v or j == v:
                continue
            if i in S and j in S:
                return True
        return False
    
    def check_all_subsets(a, S):
        """Check if a is the mean of any subset of size >=2 from S (S excludes a)."""
        # S is a set of other elements
        n = len(S)
        if n < 2:
            return False
        elements = list(S)
        # For each k from 2 to n, check if any k-subset sums to k*a
        # This is expensive for large n, but n is at most ~47
        # Use subset sum DP for each k
        for k in range(2, n+1):
            target = k * a
            # Check if any k-subset of elements sums to target
            # Use combinations for small k, DP for larger
            if k <= 4:
                for combo in combinations(elements, k):
                    if sum(combo) == target:
                        return True
            else:
                # DP: can we pick exactly k elements summing to target?
                # dp[i][j] = True if we can pick j elements from first i summing to some value
                # This is O(n * k * target) which can be large
                # Alternative: for k close to n, check if total - sum(complement) = target
                # i.e., sum(complement of size n-k) = total_sum - target
                if k > n//2:
                    complement_size = n - k
                    total_sum = sum(elements)
                    complement_target = total_sum - target
                    if complement_size <= 4:
                        for combo in combinations(elements, complement_size):
                            if sum(combo) == complement_target:
                                return True
                    else:
                        # DP for complement
                        dp = [[False]*(complement_target+1) for _ in range(complement_size+1)]
                        dp[0][0] = True
                        for elem in elements:
                            for c in range(complement_size, 0, -1):
                                for s in range(complement_target, elem-1, -1):
                                    if dp[c-1][s-elem]:
                                        dp[c][s] = True
                        if dp[complement_size][complement_target]:
                            return True
                else:
                    # DP for k elements summing to target
                    if target > sum(elements):
                        continue
                    dp = [[False]*(target+1) for _ in range(k+1)]
                    dp[0][0] = True
                    for elem in elements:
                        for c in range(min(k, len(elements)), 0, -1):
                            for s in range(target, elem-1, -1):
                                if dp[c-1][s-elem]:
                                    dp[c][s] = True
                    if dp[k][target]:
                        return True
        return False
    
    def is_valid_set(S):
        """Check if set S is fully valid (no element is mean of any subset of >=2 others)."""
        S_list = list(S)
        for a in S_list:
            others = S - {a}
            if check_all_subsets(a, others):
                return False
        return True
    
    def greedy(order):
        """Greedy construction following given order."""
        S = set()
        for v in order:
            if time.time() > deadline:
                break
            # Check 3-AP conflicts
            if check_3ap(v, S):
                continue
            S.add(v)
            if len(S) >= MAX_SIZE:
                break
        return S
    
    # Try multiple orderings
    best_set = set()
    
    # Ordering 1: ascending
    order1 = list(range(1, N+1))
    # Ordering 2: descending
    order2 = list(range(N, 0, -1))
    # Ordering 3: primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
    primes = [x for x in range(1, N+1) if is_prime(x)]
    order3 = primes + [x for x in range(1, N+1) if x not in primes]
    # Ordering 4: powers of 2 then rest
    powers2 = [2**i for i in range(1, 10) if 2**i <= N]
    order4 = powers2 + [x for x in range(1, N+1) if x not in powers2]
    # Ordering 5: alternating high-low
    order5 = []
    for i in range((N+1)//2):
        order5.append(N - i)
        if i < N//2:
            order5.append(i+1)
    
    for order in [order1, order2, order3, order4, order5]:
        if time.time() > deadline - 30:
            break
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S
    
    # Local search: try to improve by swapping
    if len(best_set) > 0 and time.time() < deadline - 15:
        current = set(best_set)
        current_list = sorted(current)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing one element and adding another
            for i, a in enumerate(current_list):
                if time.time() > deadline - 10:
                    break
                for b in range(1, N+1):
                    if b in current:
                        continue
                    # Try swap: remove a, add b
                    new_set = current - {a} | {b}
                    # Quick 3-AP check
                    if check_3ap(b, new_set - {b}):
                        continue
                    # Check if a was needed (removing it shouldn't create issues, but adding b might)
                    # Full check is expensive, do quick check
                    valid = True
                    # Check b against others
                    for (x, y) in conflict_pairs[b]:
                        if x == b or y == b:
                            continue
                        if x in new_set and y in new_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check that removing a doesn't leave b as mean of larger subset
                    # This is expensive; do a lighter check
                    others = new_set - {b}
                    # Just check 3-AP for b and a few larger subsets
                    n = len(others)
                    if n >= 2:
                        elems = list(others)
                        # Check k=2 (already done via 3-AP)
                        # Check k=3
                        for combo in combinations(elems, 3):
                            if sum(combo) == 3*b:
                                valid = False
                                break
                    if not valid:
                        continue
                    current = new_set
                    current_list = sorted(current)
                    improved = True
                    break
                if improved:
                    break
    
    # Final verification and return
    if len(best_set) > len(current):
        final_set = best_set
    else:
        final_set = current
    
    # Verify the final set is actually valid
    if final_set and is_valid_set(final_set):
        return sorted(final_set)
    else:
        # Return best valid subset found during greedy
        for order in [order1, order2]:
            S = greedy(order)
            if is_valid_set(S):
                return sorted(S)
        return sorted(list(range(1, 49)))  # fallback

