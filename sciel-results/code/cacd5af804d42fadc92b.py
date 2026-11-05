import time
import random
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin

    N = 128
    
    # Precompute bad masks
    # bad_mask[a-1][b-1] is a bitmask of c (1..128) such that (a,b,c) is a bad triple
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            mask = 0
            # Check all c
            # Optimization: 
            # Let L = lcm(a,b).
            # c must be a divisor of L.
            # Also lcm(a,c)=L and lcm(b,c)=L.
            # This implies c must contain all prime factors of L that are not fully contained in a or b?
            # Actually, just iterating c is fine for N=128.
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask

    def calc_conflict_mask(S):
        """Calculate the union of bad masks for all pairs in S."""
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask

    def try_add(x, S, conflict_mask):
        """
        Try to add x to S.
        Returns (success, new_conflict_mask).
        """
        # Check if x conflicts with any pair in S
        if (conflict_mask >> (x - 1)) & 1:
            return False, conflict_mask
        
        # If not, calculate new conflict mask
        new_mask = conflict_mask
        for y in S:
            new_mask |= bad_mask[x - 1][y - 1]
        
        return True, new_mask

    def greedy_construct(order):
        S = set()
        conflict_mask = 0
        for x in order:
            ok, new_mask = try_add(x, S, conflict_mask)
            if ok:
                S.add(x)
                conflict_mask = new_mask
        return S, conflict_mask

    best_set = set()
    best_size = 0

    # Generate diverse orders
    orders = []
    # 1. Ascending
    orders.append(list(range(1, N + 1)))
    # 2. Descending
    orders.append(list(range(N, 0, -1)))
    
    # 3. By number of prime factors (high to low)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    orders.append(sorted(range(1, N + 1), key=lambda x: (-num_prime_factors(x), x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: (num_prime_factors(x), x)))

    # 4. Random orders
    for _ in range(10):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    def local_search(S, conflict_mask, time_limit):
        """
        Perform local search to improve S.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break

            # 1. Try to add any remaining element
            added_any = False
            for x in range(1, N + 1):
                if x not in S:
                    ok, new_mask = try_add(x, S, conflict_mask)
                    if ok:
                        S.add(x)
                        conflict_mask = new_mask
                        added_any = True
                        improved = True
                        # Continue adding in this pass
            if added_any:
                continue

            # 2. Try 1-for-1 swaps
            S_list = list(S)
            swap_found = False
            for a in S_list:
                if time.time() - start_time > time_limit:
                    break
                S_temp = S - {a}
                mask_temp = calc_conflict_mask(S_temp)
                for b in range(1, N + 1):
                    if b not in S_temp:
                        ok, new_mask = try_add(b, S_temp, mask_temp)
                        if ok:
                            # Swap is valid and size is same, but might enable further additions?
                            # Actually, if size is same, it's not an improvement in size.
                            # But it might change the structure to allow additions later.
                            # However, for now, we only accept if we can add MORE afterwards.
                            # Let's try to add more to S_temp + {b}
                            S_candidate = S_temp | {b}
                            mask_candidate = new_mask
                            # Try to add any element to S_candidate
                            for c in range(1, N + 1):
                                if c not in S_candidate:
                                    ok2, mask_c = try_add(c, S_candidate, mask_candidate)
                                    if ok2:
                                        # We found a swap that allows an addition
                                        S = S_candidate | {c}
                                        conflict_mask = mask_c
                                        improved = True
                                        swap_found = True
                                        break
                            if swap_found:
                                break
                if swap_found:
                    break
            
            if swap_found:
                continue

            # 3. Try 1-for-2 swaps
            # Remove a, add b, c
            S_list = list(S)
            swap_found = False
            for a in S_list:
                if time.time() - start_time > time_limit:
                    break
                S_temp = S - {a}
                mask_temp = calc_conflict_mask(S_temp)
                for b in range(1, N + 1):
                    if b not in S_temp:
                        ok_b, mask_b = try_add(b, S_temp, mask_temp)
                        if ok_b:
                            S_b = S_temp | {b}
                            for c in range(1, N + 1):
                                if c not in S_b and c != b:
                                    ok_c, mask_c = try_add(c, S_b, mask_b)
                                    if ok_c:
                                        # Success: removed 1, added 2. Size increased by 1.
                                        S = S_b | {c}
                                        conflict_mask = mask_c
                                        improved = True
                                        swap_found = True
                                        break
                                if swap_found:
                                    break
                    if swap_found:
                        break
                if swap_found:
                    break
            
            if swap_found:
                continue

            # 4. Try 2-for-3 swaps (more expensive, use sparingly)
            # This is O(N^4) which is too slow. Skip for now or limit heavily.
            # Instead, do a random perturbation if stuck.
            
            # Perturbation: Remove 3 random elements, then re-greedy
            if len(S) > 3:
                to_remove = random.sample(list(S), 3)
                for x in to_remove:
                    S.discard(x)
                conflict_mask = calc_conflict_mask(S)
                
                # Re-greedy
                remaining = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(remaining)
                for x in remaining:
                    ok, new_mask = try_add(x, S, conflict_mask)
                    if ok:
                        S.add(x)
                        conflict_mask = new_mask
                improved = True # Force loop to continue

        return S, conflict_mask

    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
        
        S, mask = greedy_construct(order)
        S, mask = local_search(S, mask, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Final verification
    final_set = set(best_set)
    changed = True
    while changed:
        changed = False
        for a in list(final_set):
            for b in final_set:
                if a >= b:
                    continue
                for c in final_set:
                    if c <= b:
                        continue
                    if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                        final_set.discard(c)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    return list(final_set)
