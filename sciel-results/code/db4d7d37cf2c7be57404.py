"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _square_free_kernel(x):
    """Compute the square-free kernel (radical) of x."""
    result = 1
    d = 2
    while d * d <= x:
        if x % (d * d) == 0:
            while x % (d * d) == 0:
                x //= (d * d)
        d += 1
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    n = 128
    N = 128  # elements 1..128, indexed 0..127
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad triples conflicts
    bad_partners = [[0] * N for _ in range(N)]
    
    # Identify all bad triples using algebraic structure
    # Key insight: if lcm(a,b)=lcm(a,c)=lcm(b,c)=L, then a|L, b|L, c|L
    # and the square-free kernels of a,b,c must be compatible
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            if L_ab > 128 * 128:  # LCM can't exceed this for numbers <= 128
                continue
            # c must divide L_ab and be > b
            # Also lcm(a,c) == L_ab and lcm(b,c) == L_ab
            # This means c must have the same prime power structure as needed
            for c in range(b + 1, min(n + 1, L_ab + 1)):
                if L_ab % c != 0:
                    continue
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    # Found bad triple {a,b,c}
                    bad_partners[a-1][b-1] |= (1 << (c-1))
                    bad_partners[b-1][a-1] |= (1 << (c-1))
                    bad_partners[a-1][c-1] |= (1 << (b-1))
                    bad_partners[c-1][a-1] |= (1 << (b-1))
                    bad_partners[b-1][c-1] |= (1 << (a-1))
                    bad_partners[c-1][b-1] |= (1 << (a-1))

    # Precompute conflict degree for each element
    conflict_degree = [0] * N
    for i in range(N):
        degree = 0
        for j in range(N):
            if i != j and bad_partners[i][j]:
                degree += bad_partners[i][j].bit_count()
        conflict_degree[i] = degree

    def can_add(x, mask):
        """Check if element x (1-indexed) can be added to set represented by bitmask mask."""
        xi = x - 1
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            if (bad_partners[xi][ai] & mask) != 0:
                return False
            m ^= lb
        return True

    # Helper to get a random set bit index from a mask
    def get_random_bit(mask):
        if mask == 0:
            return -1
        idx = random.randint(0, mask.bit_count() - 1)
        tmp = mask
        for _ in range(idx):
            tmp &= tmp - 1
        return (tmp & (-tmp)).bit_length() - 1

    # Greedy construction with conflict-degree ordering
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask

    best_mask = 0
    best_size = 0
    
    nums = list(range(1, n + 1))
    
    # Compute omega (number of prime factors with multiplicity)
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
    kernel_cache = {x: _square_free_kernel(x) for x in nums}
    
    # Orderings based on structural properties
    orders = [
        nums,
        nums[::-1],
        sorted(nums, key=lambda x: conflict_degree[x-1]),  # Low conflict first
        sorted(nums, key=lambda x: -conflict_degree[x-1]),  # High conflict first
        sorted(nums, key=lambda x: kernel_cache[x]),  # By square-free kernel
        sorted(nums, key=lambda x: -kernel_cache[x]),
        sorted(nums, key=lambda x: omega_cache[x]),
        sorted(nums, key=lambda x: -omega_cache[x]),
        sorted(nums, key=lambda x: (kernel_cache[x], x)),  # Group by kernel
    ]
    
    # Randomized shuffles
    for _ in range(8):
        rand_order = nums[:]
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy(order)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Enhanced local search with better move proposals
    def local_search(start_mask, max_iters=200000):
        current_mask = start_mask
        current_size = current_mask.bit_count()
        best_local_mask = current_mask
        best_local_size = current_size
        
        T = 1.5
        T_min = 0.03
        cooling = 0.99993
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add: prefer elements with low conflict degree
                not_in = ((1 << N) - 1) & ~current_mask
                if not_in:
                    # Weighted random selection favoring low conflict
                    candidates = []
                    m = not_in
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        candidates.append((conflict_degree[idx], idx))
                        m ^= lb
                    if candidates:
                        # Pick from lower conflict elements
                        candidates.sort()
                        pick_idx = random.randint(0, min(10, len(candidates) - 1))
                        x_idx = candidates[pick_idx][1]
                        x = x_idx + 1
                        if can_add(x, current_mask):
                            new_mask = current_mask | (1 << x_idx)
                            current_mask = new_mask
                            current_size += 1
                            if current_size > best_local_size:
                                best_local_size = current_size
                                best_local_mask = current_mask
            elif move_type < 0.55:
                # Remove: prefer high conflict elements
                if current_mask:
                    candidates = []
                    m = current_mask
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        candidates.append((conflict_degree[idx], idx))
                        m ^= lb
                    if candidates:
                        candidates.sort(reverse=True)
                        pick_idx = random.randint(0, min(10, len(candidates) - 1))
                        x_idx = candidates[pick_idx][1]
                        new_mask = current_mask & ~(1 << x_idx)
                        current_mask = new_mask
                        current_size -= 1
            elif move_type < 0.8:
                # Swap: remove high conflict, add low conflict
                if current_mask:
                    not_in = ((1 << N) - 1) & ~current_mask
                    if not_in:
                        # Remove candidate (high conflict)
                        rem_candidates = []
                        m = current_mask
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            rem_candidates.append((conflict_degree[idx], idx))
                            m ^= lb
                        rem_candidates.sort(reverse=True)
                        
                        # Add candidate (low conflict)
                        add_candidates = []
                        m = not_in
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            add_candidates.append((conflict_degree[idx], idx))
                            m ^= lb
                        add_candidates.sort()
                        
                        if rem_candidates and add_candidates:
                            x_idx = rem_candidates[0][1]
                            y_idx = add_candidates[0][1]
                            temp_mask = current_mask & ~(1 << x_idx)
                            y = y_idx + 1
                            if can_add(y, temp_mask):
                                new_mask = temp_mask | (1 << y_idx)
                                current_mask = new_mask
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
            else:
                # 2-swap: remove two high conflict, add two low conflict
                if current_mask.bit_count() >= 2:
                    not_in = ((1 << N) - 1) & ~current_mask
                    if not_in:
                        rem_candidates = []
                        m = current_mask
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            rem_candidates.append((conflict_degree[idx], idx))
                            m ^= lb
                        rem_candidates.sort(reverse=True)
                        
                        add_candidates = []
                        m = not_in
                        while m:
                            lb = m & (-m)
                            idx = lb.bit_length() - 1
                            add_candidates.append((conflict_degree[idx], idx))
                            m ^= lb
                        add_candidates.sort()
                        
                        if len(rem_candidates) >= 2 and len(add_candidates) >= 2:
                            x1_idx = rem_candidates[0][1]
                            x2_idx = rem_candidates[1][1]
                            y1_idx = add_candidates[0][1]
                            y2_idx = add_candidates[1][1]
                            
                            temp_mask = current_mask & ~(1 << x1_idx) & ~(1 << x2_idx)
                            y1 = y1_idx + 1
                            y2 = y2_idx + 1
                            if can_add(y1, temp_mask) and can_add(y2, temp_mask | (1 << y1_idx)):
                                new_mask = temp_mask | (1 << y1_idx) | (1 << y2_idx)
                                current_mask = new_mask
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
            T *= cooling
            
        return best_local_mask, best_local_size

    # Run LS from best greedy
    if time.time() < deadline - 5:
        mask, sz = local_search(best_mask, max_iters=180000)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Run LS from multiple diverse starts
    restarts = 0
    while time.time() < deadline - 8 and restarts < 8:
        # Different initialization strategies
        if restarts % 3 == 0:
            # Low conflict greedy
            init_mask = greedy(sorted(nums, key=lambda x: conflict_degree[x-1]))
        elif restarts % 3 == 1:
            # Kernel-grouped greedy
            init_mask = greedy(sorted(nums, key=lambda x: (kernel_cache[x], x)))
        else:
            # Random
            init_mask = 0
            rand_order = nums[:]
            random.shuffle(rand_order)
            for x in rand_order:
                if can_add(x, init_mask):
                    init_mask |= (1 << (x - 1))
        
        mask, sz = local_search(init_mask, max_iters=100000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
        restarts += 1

    # Final polish: exhaustive single-element addition
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
        
        # Try 1-for-1 swaps with greedy completion
        if time.time() < deadline - 1:
            for x in range(1, n + 1):
                if best_mask & (1 << (x - 1)):
                    temp_mask = best_mask & ~(1 << (x - 1))
                    for y in range(1, n + 1):
                        if not (best_mask & (1 << (y - 1))):
                            if can_add(y, temp_mask):
                                new_mask = temp_mask | (1 << (y - 1))
                                new_size = new_mask.bit_count()
                                # Try to add more after swap
                                for z in range(1, n + 1):
                                    if not (new_mask & (1 << (z - 1))):
                                        if can_add(z, new_mask):
                                            new_mask |= (1 << (z - 1))
                                            new_size += 1
                                if new_size > best_size:
                                    best_mask = new_mask
                                    best_size = new_size
                                    break

    # Convert mask to list
    result = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    
    # Final verification
    assert len(result) > 0
    for a, b, c in combinations(result, 3):
        assert not (_lcm(a, b) == _lcm(a, c) == _lcm(b, c)), f"Bad triple: {a},{b},{c}"
    
    return result
# EVOLVE-BLOCK-END
