"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    N = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad triples
    bad_partners = [[0] * N for _ in range(N)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_partners[a-1][b-1] |= (1 << (c-1))
                    bad_partners[b-1][a-1] |= (1 << (c-1))
                    bad_partners[a-1][c-1] |= (1 << (b-1))
                    bad_partners[c-1][a-1] |= (1 << (b-1))
                    bad_partners[b-1][c-1] |= (1 << (a-1))
                    bad_partners[c-1][b-1] |= (1 << (a-1))
    
    def can_add(x, mask):
        xi = x - 1
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            if (bad_partners[xi][ai] & mask) != 0:
                return False
            m ^= lb
        return True
    
    def get_random_bit(mask):
        if mask == 0:
            return -1
        idx = random.randint(0, mask.bit_count() - 1)
        tmp = mask
        for _ in range(idx):
            tmp &= tmp - 1
        return (tmp & (-tmp)).bit_length() - 1
    
    # Analyze structure: group numbers by their "core" (odd part)
    # Numbers with same odd part are powers of 2 times that odd part
    # These have special LCM properties
    
    def odd_part(x):
        while x % 2 == 0:
            x //= 2
        return x
    
    # For each number, compute its prime factorization signature
    def prime_sig(x):
        sig = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                sig[d] = sig.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            sig[x] = sig.get(x, 0) + 1
        return sig
    
    # Key insight: if a, b, c all divide L and lcm(a,b)=lcm(a,c)=lcm(b,c)=L,
    # then for each prime p, the max exponent in {a,b,c} equals L's exponent,
    # and at least two of {a,b,c} must have that max exponent (otherwise the third
    # would need to supply it, but then all three would have it, making lcm larger
    # unless the third also has it).
    # Actually: for each prime p with v_p(L) = e > 0, at least two of a,b,c must
    # have v_p = e. If only one has v_p = e, say a, then v_p(b) < e and v_p(c) < e,
    # so v_p(lcm(b,c)) < e = v_p(L), contradiction.
    
    # This means: for each "critical" prime (one where L has positive exponent),
    # at least 2 of the 3 numbers must achieve the full exponent in L.
    
    # Let's use this to generate stronger constraints.
    # For a given L, the set of valid triples is very restricted.
    
    # Better approach: identify "conflict classes"
    # Two numbers a, b are in conflict if there exists c such that {a,b,c} is bad.
    # The bad_partners table already captures this.
    
    # Let's try a more sophisticated local search with better move operators
    
    def local_search_v2(start_mask, max_iters=80000):
        current_mask = start_mask
        current_size = current_mask.bit_count()
        best_local_mask = current_mask
        best_local_size = current_size
        
        T = 3.0
        T_min = 0.005
        cooling = 0.99995
        
        full_mask = (1 << N) - 1
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Add
                not_in = full_mask & ~current_mask
                if not_in:
                    x_idx = get_random_bit(not_in)
                    if x_idx != -1:
                        x = x_idx + 1
                        if can_add(x, current_mask):
                            new_mask = current_mask | (1 << x_idx)
                            new_size = new_mask.bit_count()
                            delta = new_size - current_size
                            if delta > 0 or random.random() < (T if T > T_min else 0.0):
                                current_mask = new_mask
                                current_size = new_size
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
            elif move_type < 0.55:
                # Remove
                if current_mask:
                    x_idx = get_random_bit(current_mask)
                    if x_idx != -1:
                        new_mask = current_mask & ~(1 << x_idx)
                        new_size = new_mask.bit_count()
                        delta = new_size - current_size
                        if delta > 0 or random.random() < (T if T > T_min else 0.001):
                            current_mask = new_mask
                            current_size = new_size
            elif move_type < 0.8:
                # Swap (1-for-1)
                if current_mask:
                    x_idx = get_random_bit(current_mask)
                    not_in = full_mask & ~current_mask
                    if x_idx != -1 and not_in:
                        y_idx = get_random_bit(not_in)
                        if y_idx != -1:
                            temp_mask = current_mask & ~(1 << x_idx)
                            y = y_idx + 1
                            if can_add(y, temp_mask):
                                new_mask = temp_mask | (1 << y_idx)
                                new_size = new_mask.bit_count()
                                delta = new_size - current_size
                                if delta > 0 or random.random() < (T if T > T_min else 0.01):
                                    current_mask = new_mask
                                    current_size = new_size
                                    if current_size > best_local_size:
                                        best_local_size = current_size
                                        best_local_mask = current_mask
            else:
                # 2-for-1 or 1-for-2 moves
                if current_mask:
                    x_idx = get_random_bit(current_mask)
                    if x_idx != -1:
                        # Try removing x and adding two
                        temp_mask = current_mask & ~(1 << x_idx)
                        not_in = full_mask & ~temp_mask
                        if not_in:
                            # Try adding two random elements
                            candidates = []
                            m = not_in
                            count = 0
                            while m and count < 50:
                                lb = m & (-m)
                                idx = lb.bit_length() - 1
                                candidates.append(idx)
                                m ^= lb
                                count += 1
                            
                            if len(candidates) >= 2:
                                # Try pairs from candidates
                                for i in range(min(len(candidates), 20)):
                                    if time.time() >= deadline:
                                        break
                                    c1 = candidates[i]
                                    if can_add(c1 + 1, temp_mask):
                                        temp1 = temp_mask | (1 << c1)
                                        for j in range(i + 1, min(len(candidates), 20)):
                                            c2 = candidates[j]
                                            if can_add(c2 + 1, temp1):
                                                new_mask = temp1 | (1 << c2)
                                                new_size = new_mask.bit_count()
                                                delta = new_size - current_size
                                                if delta > 0 or random.random() < (T if T > T_min else 0.005):
                                                    current_mask = new_mask
                                                    current_size = new_size
                                                    if current_size > best_local_size:
                                                        best_local_size = current_size
                                                        best_local_mask = current_mask
                                                    break
                            else:
                                # Try 1-for-2: remove two, add one
                                m2 = current_mask
                                rem_candidates = []
                                count = 0
                                while m2 and count < 30:
                                    lb = m2 & (-m2)
                                    idx = lb.bit_length() - 1
                                    rem_candidates.append(idx)
                                    m2 ^= lb
                                    count += 1
                                
                                if len(rem_candidates) >= 2 and not_in:
                                    for i in range(min(len(rem_candidates), 10)):
                                        if time.time() >= deadline:
                                            break
                                        r1 = rem_candidates[i]
                                        for j in range(i + 1, min(len(rem_candidates), 10)):
                                            r2 = rem_candidates[j]
                                            temp2 = current_mask & ~(1 << r1) & ~(1 << r2)
                                            # Try adding one element
                                            m3 = full_mask & ~temp2
                                            if m3:
                                                add_idx = get_random_bit(m3)
                                                if add_idx != -1:
                                                    if can_add(add_idx + 1, temp2):
                                                        new_mask = temp2 | (1 << add_idx)
                                                        new_size = new_mask.bit_count()
                                                        delta = new_size - current_size
                                                        if delta > 0 or random.random() < (T if T > T_min else 0.005):
                                                            current_mask = new_mask
                                                            current_size = new_size
                                                            if current_size > best_local_size:
                                                                best_local_size = current_size
                                                                best_local_mask = current_mask
                                    break
            T *= cooling
            
        return best_local_mask, best_local_size
    
    # Greedy construction with multiple orderings
    nums = list(range(1, n + 1))
    
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
        
    omega_cache = {x: omega(x) for x in nums}
    
    orders = [
        nums,
        nums[::-1],
        sorted(nums, key=lambda x: x.bit_count()),
        sorted(nums, key=lambda x: -x.bit_count()),
        sorted(nums, key=lambda x: omega_cache[x]),
        sorted(nums, key=lambda x: -omega_cache[x]),
        sorted(nums, key=lambda x: odd_part(x)),
        sorted(nums, key=lambda x: -odd_part(x)),
    ]
    
    for _ in range(8):
        rand_order = nums[:]
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask
    
    best_mask = 0
    best_size = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy(order)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Run local search from best greedy
    if time.time() < deadline:
        mask, sz = local_search_v2(best_mask, max_iters=60000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Run local search from several random valid starts
    for _ in range(4):
        if time.time() >= deadline - 15:
            break
        init_mask = 0
        rand_order = nums[:]
        random.shuffle(rand_order)
        for x in rand_order:
            if can_add(x, init_mask):
                init_mask |= (1 << (x - 1))
        
        mask, sz = local_search_v2(init_mask, max_iters=40000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final polish: try to add any single element
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask & (1 << (x - 1))):
                    if can_add(x, best_mask):
                        best_mask |= (1 << (x - 1))
                        best_size += 1
                        improved = True
    
    # Final local search pass
    if time.time() < deadline - 5:
        mask, sz = local_search_v2(best_mask, max_iters=30000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Convert mask to list
    result = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    
    # Final verification
    assert len(result) > 0
    for a, b, c in combinations(result, 3):
        assert not (_lcm(a, b) == _lcm(a, c) == _lcm(b, c)), f"Bad triple: {a},{b},{c}"
    
    return result
# EVOLVE-BLOCK-END
