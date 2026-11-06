import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Use 0-indexed internally: element x -> index x-1
    # Bitmask: bit i set means element (i+1) is in the set

    # Precompute bad triples and conflicts
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed

    # conflicts[x] = list of (u, v) pairs such that (u,v,x) is a bad triple
    conflicts = [[] for _ in range(N)]
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # For each x, also precompute conflict_mask_pairs: for quick bitmask checking
    # conflict_pair_mask[x] is a list where each entry is (u, v, bitmask) for fast checking
    # Actually, let's precompute for each x a set of frozenset({u,v}) for O(1) lookup
    # But that's memory heavy. Instead, let's use a different approach:
    # For each x, build a bitmask-based structure:
    # For each pair (u,v) in conflicts[x], we need to check if both u and v are in mask.
    # We can precompute: for each x, a list of (u, v) sorted, and use early termination.

    def is_valid_add(mask, x):
        """Check if adding x (0-indexed) to set represented by mask is valid."""
        for (u, v) in conflicts[x]:
            if ((mask >> u) & 1) and ((mask >> v) & 1):
                return False
        return True

    # Greedy construction with multiple orders
    random.seed(42)
    best_mask = 0
    best_size = 0

    def greedy(order):
        mask = 0
        size = 0
        for x in order:
            if is_valid_add(mask, x):
                mask |= (1 << x)
                size += 1
        return mask, size

    # Compute degree
    degree = [len(conflicts[x]) for x in range(N)]

    # Create various orderings
    all_elements = list(range(N))
    orders = [
        all_elements[:],
        all_elements[::-1],
        sorted(all_elements, key=lambda x: degree[x]),
        sorted(all_elements, key=lambda x: -degree[x]),
        sorted(all_elements, key=lambda x: (degree[x], -x)),
        sorted(all_elements, key=lambda x: (-degree[x], x)),
    ]

    # Random orders - increase count
    for _ in range(50):
        random_order = all_elements[:]
        random.shuffle(random_order)
        orders.append(random_order)

    # Prime-heavy
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(N) if is_prime(x + 1)]
    composites = [x for x in range(N) if not is_prime(x + 1)]
    orders.append(primes + composites)
    orders.append(composites + primes)

    # Power-of-2 heavy
    def is_power_of_2(n):
        return (n & (n - 1)) == 0 and n > 0
    p2 = [x for x in range(N) if is_power_of_2(x + 1)]
    np2 = [x for x in range(N) if not is_power_of_2(x + 1)]
    orders.append(p2 + np2)
    orders.append(np2 + p2)

    # Mixed: interleave primes and composites
    mixed1 = []
    pi, ci = 0, 0
    while pi < len(primes) or ci < len(composites):
        if pi < len(primes):
            mixed1.append(primes[pi])
            pi += 1
        if ci < len(composites):
            mixed1.append(composites[ci])
            ci += 1
    orders.append(mixed1)

    for order in orders:
        mask, size = greedy(order)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local search using bitmask operations
    def try_add_one(mask):
        """Try to add one element. Return new mask if successful, else None."""
        for x in range(N):
            if not ((mask >> x) & 1):
                if is_valid_add(mask, x):
                    return mask | (1 << x)
        return None

    def try_swap_1_for_2(mask):
        """Try to remove 1 element and add 2."""
        if time.time() > deadline:
            return None
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not ((mask >> x) & 1)]
        
        random.shuffle(in_mask)
        
        for u in in_mask:
            if time.time() > deadline:
                return None
            temp_mask = mask & ~(1 << u)
            
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 2:
                for i in range(len(valid_candidates)):
                    x = valid_candidates[i]
                    temp_with_x = temp_mask | (1 << x)
                    for j in range(i + 1, len(valid_candidates)):
                        y = valid_candidates[j]
                        if is_valid_add(temp_with_x, y):
                            return temp_mask | (1 << x) | (1 << y)
        return None

    def try_swap_2_for_3(mask):
        """Try to remove 2 elements and add 3."""
        if time.time() > deadline - 3.0:
            return None
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not ((mask >> x) & 1)]
        
        random.shuffle(in_mask)
        
        # Use actual combinations for correctness
        max_pairs = 50
        pair_count = 0
        for i in range(len(in_mask)):
            if pair_count >= max_pairs:
                break
            for j in range(i + 1, len(in_mask)):
                if pair_count >= max_pairs:
                    break
                if time.time() > deadline - 1.0:
                    break
                u, v = in_mask[i], in_mask[j]
                pair_count += 1
                
                temp_mask = mask & ~(1 << u) & ~(1 << v)
                
                valid_candidates = []
                for x in not_in_mask:
                    if is_valid_add(temp_mask, x):
                        valid_candidates.append(x)
                
                if len(valid_candidates) >= 3:
                    max_triples = 200
                    triple_count = 0
                    for a in range(len(valid_candidates)):
                        if triple_count >= max_triples:
                            break
                        x = valid_candidates[a]
                        temp_x = temp_mask | (1 << x)
                        for b in range(a + 1, len(valid_candidates)):
                            if triple_count >= max_triples:
                                break
                            y = valid_candidates[b]
                            temp_xy = temp_x | (1 << y)
                            for c in range(b + 1, len(valid_candidates)):
                                if triple_count >= max_triples:
                                    break
                                z = valid_candidates[c]
                                if is_valid_add(temp_xy, z):
                                    return temp_mask | (1 << x) | (1 << y) | (1 << z)
                                triple_count += 1
        return None

    # Main local search loop
    S_mask = best_mask
    current_size = best_size
    stuck_count = 0
    no_improve_count = 0

    while time.time() < deadline:
        improved = False

        # N0: Add 1
        new_mask = try_add_one(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # N1: 1-for-2
        new_mask = try_swap_1_for_2(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # N2: 2-for-3
        new_mask = try_swap_2_for_3(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # 1-for-1 swap to change structure
        improved_structure = False
        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]
        random.shuffle(in_mask)

        for u in in_mask:
            if time.time() > deadline:
                break
            temp_mask = S_mask & ~(1 << u)
            for v in not_in_mask:
                if is_valid_add(temp_mask, v):
                    S_mask = temp_mask | (1 << v)
                    improved_structure = True
                    break
            if improved_structure:
                break

        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart
                S_mask = best_mask
                stuck_count = 0
                no_improve_count = 0

                # Perturb with 1-3 random swaps
                in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not in_mask or not not_in_mask:
                        break
                    u = random.choice(in_mask)
                    v = random.choice(not_in_mask)
                    temp_mask = S_mask & ~(1 << u)
                    if is_valid_add(temp_mask, v):
                        S_mask = temp_mask | (1 << v)
                        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                        not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]

    # Convert best_mask back to list of 1-indexed elements
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return result if result else [1]
