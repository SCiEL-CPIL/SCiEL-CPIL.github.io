import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def is_valid_set(S):
        """
        Check if set S is valid: no element is the mean of any subset of >=2 other elements.
        """
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i+1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    def can_add(S, x, dp_S=None, sum_S=None):
        """
        Fast check if x can be added to S.
        If dp_S and sum_S are provided, reuse them.
        """
        n = len(S)
        if n == 0:
            return True
        
        if dp_S is None:
            dp_S = build_dp(S)
        if sum_S is None:
            sum_S = sum(S)
        
        # Check 1: x is not the mean of any subset of S of size >= 2
        for k in range(2, n + 1):
            t = k * x
            if t > sum_S:
                break
            if (dp_S[k] >> t) & 1:
                return False

        # Check 2: Adding x doesn't make any a in S the mean of a subset including x
        # For each a in S, check if (j+1)*a - x is achievable with j elements from S\{a}, j>=1
        for a in S:
            # We need DP of S\{a}
            # Optimization: if (j+1)*a - x is out of range for all j, skip
            # min sum of j elements from S\{a} >= j * min(S\{a})
            # max sum of j elements from S\{a} <= j * max(S\{a})
            # But we need exact DP. Let's build it but with early exit.
            
            # Build DP for S without a
            S_minus_a = [y for y in S if y != a]
            dp_sub = build_dp(S_minus_a)
            max_sub = sum_S - a
            
            for j in range(1, n):
                t = (j + 1) * a - x
                if t < 0 or t > max_sub:
                    continue
                if (dp_sub[j] >> t) & 1:
                    return False
        
        return True

    # Simulated annealing approach
    # Start with a good initial solution from greedy
    def greedy_construct(order):
        cur = []
        for x in order:
            if len(cur) >= MAXC:
                break
            if can_add(cur, x):
                cur.append(x)
        return cur

    # Initial greedy with multiple orderings
    best = []
    best_sz = 0
    
    orders = [list(range(1, N + 1)), list(range(N, 0, -1))]
    for _ in range(6):
        m = list(range(1, N + 1))
        random.shuffle(m)
        orders.append(m)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        if best_sz >= MAXC:
            break
        cur = greedy_construct(order)
        if len(cur) > best_sz:
            if is_valid_set(cur):
                best_sz = len(cur)
                best = cur[:]

    # Simulated annealing to improve
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        T = 10.0  # Initial temperature
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 2 and T > min_T:
            T *= cooling
            
            # Propose a move
            move_type = random.random()
            accepted = False
            
            if len(S) < MAXC and move_type < 0.4:
                # Try to add a random unused element
                if unused:
                    u = random.choice(unused)
                    if can_add(S, u):
                        S.append(u)
                        Sset.add(u)
                        unused.remove(u)
                        accepted = True
            elif len(S) > 2 and move_type < 0.7:
                # Try to swap
                if unused:
                    i = random.randrange(len(S))
                    u = random.choice(unused)
                    s = S[i]
                    newS = S[:i] + S[i+1:]
                    if can_add(newS, u):
                        S = newS + [u]
                        Sset.discard(s)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s)
                        accepted = True
            elif len(S) > 1 and move_type < 0.85:
                # Try to remove an element (to allow better additions later)
                i = random.randrange(len(S))
                s = S[i]
                S = S[:i] + S[i+1:]
                Sset.discard(s)
                unused.append(s)
                accepted = True
            else:
                # Try to add a random unused element even if it might not fit
                if unused:
                    u = random.choice(unused)
                    if can_add(S, u):
                        S.append(u)
                        Sset.add(u)
                        unused.remove(u)
                        accepted = True
            
            # Track best
            if len(S) > best_sz and is_valid_set(S):
                best_sz = len(S)
                best = S[:]
            
            # If we reached MAXC, stop
            if best_sz >= MAXC:
                break

    # Final verification
    if not is_valid_set(best):
        for sz in range(len(best), 0, -1):
            if is_valid_set(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
