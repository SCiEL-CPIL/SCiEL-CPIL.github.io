# EVOLVE-BLOCK-START

import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 55.0

    n = 128

    def mask_of(S):
        m = 0
        for x in S:
            m |= (1 << (x - 1))
        return m

    def bits(m):
        res = []
        mm = m
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            res.append(idx + 1)
            mm ^= lsb
        return res

    def popcount(m):
        return bin(m).count('1')

    def check_full(mask):
        # Valid iff no distinct a<b<c in S with a+b,a+c,b+c all in S.
        # Equivalent to: no distinct a<b in S with a+b in S and (exists c in S, c!=a,b, a+c in S and b+c in S).
        for a in range(1, n + 1):
            if not (mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (mask & (1 << (b - 1))):
                    continue
                s = a + b
                if s > n:
                    break
                if not (mask & (1 << (s - 1))):
                    continue
                Va = mask & (mask >> a)
                Vb = mask & (mask >> b)
                common = Va & Vb
                a_bit = 1 << (a - 1)
                b_bit = 1 << (b - 1)
                if common & ~(a_bit | b_bit):
                    return False
        return True

    def try_add(mask, x):
        if mask & (1 << (x - 1)):
            return True, mask
        # Case 1: x as a member
        Ux = mask & (mask >> x)
        m = Ux
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            Vb = mask & (mask >> b_val)
            if Ux & Vb & ~lsb_b:
                return False, mask
        # Case 2: x as sum
        for a in range(1, x):
            b = x - a
            if b <= a:
                continue
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            Va = mask & (mask >> a)
            Vb = mask & (mask >> b)
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if Va & Vb & ~(a_bit | b_bit):
                return False, mask
        return True, mask | (1 << (x - 1))

    def extend(mask):
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if mask & (1 << (x - 1)):
                    continue
                ok, newm = try_add(mask, x)
                if ok:
                    mask = newm
                    changed = True
        return mask

    def find_violation(mask):
        for a in range(1, n + 1):
            if not (mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (mask & (1 << (b - 1))):
                    continue
                s = a + b
                if s > n:
                    break
                if not (mask & (1 << (s - 1))):
                    continue
                Va = mask & (mask >> a)
                Vb = mask & (mask >> b)
                a_bit = 1 << (a - 1)
                b_bit = 1 << (b - 1)
                check = Va & Vb & ~(a_bit | b_bit)
                if check:
                    c_bit = check & -check
                    c_val = c_bit.bit_length()
                    return [a, b, c_val, a + b, a + c_val, b + c_val]
        return None

    def repair(mask, max_steps=200):
        steps = 0
        while steps < max_steps:
            viol = find_violation(mask)
            if viol is None:
                break
            elem = random.choice(viol)
            if elem <= n:
                mask &= ~(1 << (elem - 1))
            steps += 1
        return mask

    # Initial structural mask
    K = 64
    H = list(range(K, n + 1))
    max_L = (K - 1) // 2
    start_L = max_L // 2 + 1
    L = list(range(start_L, max_L + 1))
    base_mask = mask_of(H + L)
    if not check_full(base_mask):
        base_mask = repair(base_mask)
    base_mask = extend(base_mask)

    best_mask = base_mask
    best_size = popcount(base_mask)

    def local_search(mask, time_budget=3.0):
        # Simulated annealing-ish local search with bit flips
        mask = extend(mask)
        cur_size = popcount(mask)
        t_end = time.time() + time_budget
        T = 1.0
        while time.time() < t_end:
            # Try flipping one bit
            x = random.randint(1, n)
            new_mask = mask ^ (1 << (x - 1))
            # If we're removing a bit, validity is preserved; if adding, we need to check.
            if (mask & (1 << (x - 1))) == 0:
                ok, new_mask = try_add(new_mask, x)
                if not ok:
                    continue
            # If removing, it's always valid, but we want to ensure we don't drop below current size unless exploring
            new_size = popcount(new_mask)
            if new_size > cur_size:
                mask = new_mask
                cur_size = new_size
                T = max(T * 0.99, 0.01)
            elif random.random() < (1.0 / (1.0 + (cur_size - new_size) * 0.1)):
                mask = new_mask
                cur_size = new_size
                T = max(T * 0.95, 0.01)
            else:
                T = max(T * 0.99, 0.01)
        mask = extend(mask)
        return mask

    # Multi-start local search
    random.seed(42)
    num_starts = 30
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate a random starting point
        if i % 3 == 0:
            start_mask = base_mask
        else:
            start_mask = 0
            for x in range(1, n + 1):
                if random.random() < 0.5:
                    start_mask |= (1 << (x - 1))
            start_mask = repair(start_mask)
            start_mask = extend(start_mask)
        
        # Run local search
        result_mask = local_search(start_mask, time_budget=1.0)
        
        if popcount(result_mask) > best_size:
            best_size = popcount(result_mask)
            best_mask = result_mask

    # Final polish on the best mask found
    if time.time() < deadline - 1.0:
        final_mask = extend(best_mask)
        if popcount(final_mask) > best_size:
            best_mask = final_mask
            best_size = popcount(final_mask)
        
        # One more round of local search on the best
        result_mask = local_search(best_mask, time_budget=2.0)
        if popcount(result_mask) > best_size:
            best_mask = result_mask
            best_size = popcount(final_mask)

    result = bits(best_mask)
    return sorted(result)

# EVOLVE-BLOCK-END
