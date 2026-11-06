import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0
    N = 192
    max_sum = 3 * N  # 576

    # Precompute divisors for all sums up to max_sum
    divisors = [[] for _ in range(max_sum + 1)]
    for s in range(3, max_sum + 1):
        for d in range(1, N + 1):
            if s % d == 0:
                divisors[s].append(d)

    def is_valid_set(S):
        """Verify set S is valid."""
        S_list = sorted(S)
        K = len(S_list)
        if K < 4:
            return True
        S_set = set(S_list)
        for idx_a, a in enumerate(S_list):
            others = [x for x in S_list if x != a]
            # Check a does not divide sum of any 3 distinct others
            # Use the multiple-target method for efficiency
            multiples = [k * a for k in range(1, max_sum // a + 1)]
            for M in multiples:
                if M < 3 or M > max_sum:
                    continue
                for i in range(len(others)):
                    x = others[i]
                    if x > M:
                        continue
                    for j in range(i + 1, len(others)):
                        y = others[j]
                        if x + y > M:
                            continue
                        z = M - x - y
                        if z == x or z == y:
                            continue
                        if z in S_set:
                            return False
        return True

    def can_add(a, S):
        """Check if adding a to S keeps validity. Uses divisors lookup for part 2."""
        S_list = list(S)
        K = len(S_list)
        S_set = set(S_list)

        # Check 1: a does not divide sum of any 3 distinct elements in S
        # Use multiple-target method: for each multiple M of a, check if 3 distinct elements in S sum to M
        multiples = [k * a for k in range(1, max_sum // a + 1)]
        for M in multiples:
            if M < 3 or M > max_sum:
                continue
            for i in range(K):
                x = S_list[i]
                if x > M:
                    continue
                for j in range(i + 1, K):
                    y = S_list[j]
                    if x + y > M:
                        continue
                    z = M - x - y
                    if z == x or z == y:
                        continue
                    if z in S_set:
                        return False

        # Check 2: For each pair (u, v) in S, the sum T = a + u + v
        # must not be divisible by any b in S where b != a, b != u, b != v
        for i in range(K):
            u = S_list[i]
            for j in range(i + 1, K):
                v = S_list[j]
                T = a + u + v
                if T <= max_sum:
                    for b in divisors[T]:
                        if b == a or b == u or b == v:
                            continue
                        if b in S_set:
                            return False
        return True

    # Strategy: Start with upper half [97, 192] greedily, then add smaller numbers
    S = set()
    
    # Phase 1: Greedily add from 192 down to 97
    for x in range(N, 96, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    # Phase 2: Try to add smaller numbers
    for x in range(96, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # Local search
    rng = random.Random(12345)
    
    all_elements = set(range(1, N + 1))
    
    while time.time() < deadline - 3.0:
        S_list = list(S)
        U = list(all_elements - S)
        
        if not U:
            break
        
        # Try random additions
        if len(S) < 48:
            for _ in range(50):
                if time.time() > deadline:
                    break
                a = rng.choice(U)
                if can_add(a, S):
                    S.add(a)
                    U.remove(a)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    if len(S) >= 48:
                        break
        
        # Try swaps: remove one, add one
        if S_list:
            r = rng.choice(S_list)
            S_mid = S - {r}
            U_mid = list(all_elements - S_mid)
            for _ in range(30):
                if time.time() > deadline:
                    break
                a = rng.choice(U_mid)
                if can_add(a, S_mid):
                    S = S_mid | {a}
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
            else:
                # Try double swap: remove two, add two
                if len(S_list) >= 2:
                    r2 = rng.choice([x for x in S_list if x != r])
                    S_mid2 = S - {r, r2}
                    U_mid2 = list(all_elements - S_mid2)
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        a1 = rng.choice(U_mid2)
                        if can_add(a1, S_mid2):
                            S_temp = S_mid2 | {a1}
                            U_mid2.remove(a1)
                            a2 = rng.choice(U_mid2)
                            if can_add(a2, S_temp):
                                S = S_temp | {a2}
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                break
                    else:
                        S = S_mid2 | {r, r2}  # revert
                else:
                    S = S_mid2 | {r} if len(S_list) == 1 else S

    # Final verification
    if not is_valid_set(best_S):
        # Fallback: rebuild
        S = set()
        for x in range(N, 0, -1):
            if can_add(x, S):
                S.add(x)
        best_S = S

    return list(best_S)
