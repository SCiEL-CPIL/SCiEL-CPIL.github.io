import time
import random
from math import gcd

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

    # Precompute conflict bitmasks for faster checking
    # conflict_mask[x] has bits set for all y that form a conflict pair with some z when x is added
    # Actually, let's precompute: for each x, a bitmask of all y such that there exists z where (x,y,z) is bad
    # But we need to check if BOTH u and v are in mask, so we need the pair structure.
    # Let's precompute for each x: list of (u,v) pairs, but also a bitmask for quick "any conflict" check
    
    # For speed: precompute pair_bitmask[x] where bit (u*N+v) is set if (u,v) is a conflict pair for x
    # But N=128, so u*N+v can be up to 128*128=16384 bits. That's fine with Python ints.
    
    # Alternative: for each x, store the list of (u,v) pairs. The current approach is fine.
    # Let's add a precomputed "neighbors" bitmask: neighbor_mask[x] = bitmask of all y that appear in any conflict pair for x
    # This helps quickly reject: if mask & neighbor_mask[x] == 0, then definitely valid.
    neighbor_mask = [0] * N
    for x in range(N):
        for (u, v) in conflicts[x]:
            neighbor_mask[x] |= (1 << u) | (1 << v)

    def is_valid_add(mask, x):
        """Check if adding x (0-indexed) to set represented by mask is valid."""
        # Quick reject: if no neighbors of x are in mask, it's valid
        if mask & neighbor_mask[x] == 0:
            return True
        for (u, v) in conflicts[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
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

    # Compute degree (number of conflict pairs)
    degree = [len(conflicts[x]) for x in range(N)]

    # Also compute "neighborhood size" - how many distinct elements are in conflict with x
    neighbor_count = [bin(neighbor_mask[x]).count('1') for x in range(N)]

    # Create various orderings
    all_elements = list(range(N))
    orders = [
        all_elements[:],
        all_elements[::-1],
        sorted(all_elements, key=lambda x: degree[x]),
        sorted(all_elements, key=lambda x: -degree[x]),
        sorted(all_elements, key=lambda x: (degree[x], -x)),
        sorted(all_elements, key=lambda x: (-degree[x], x)),
        sorted(all_elements, key=lambda x: neighbor_count[x]),
        sorted(all_elements, key=lambda x: -neighbor_count[x]),
    ]

    # Random orders
    for _ in range(30):
        random_order = all_elements[:]
        random.shuffle(random_order)
        orders.append(random_order)

    # Prime-heavy
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(N) if is_prime(x + 1)]
    composites = [x for x in range(N) if not is_prime(x + 1)]
    orders.append(primes + composites)
    orders.append(composites + primes)

    # Power-of-2 based ordering
    def popcount(n):
        return bin(n).count('1')
    
    powers_order = sorted(all_elements, key=lambda x: (popcount(x + 1), x + 1))
    orders.append(powers_order)
    orders.append(powers_order[::-1])

    for order in orders:
        mask, size = greedy(order)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local search using bitmask operations
    def try_add_one(mask):
        """Try to add one element. Return new mask if successful, else None."""
        for x in range(N):
            if not (mask >> x) & 1:
                if is_valid_add(mask, x):
                    return mask | (1 << x)
        return None

    def try_swap_1_for_2(mask):
        """Try to remove 1 element and add 2."""
        if time.time() > deadline:
            return None
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        
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
        if time.time() > deadline - 2.0:
            return None
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        
        random.shuffle(in_mask)
        n_in = len(in_mask)
        
        # Enumerate pairs properly
        max_pairs = min(50, n_in * (n_in - 1) // 2)
        pair_list = []
        for i in range(n_in):
            for j in range(i + 1, n_in):
                if len(pair_list) >= max_pairs:
                    break
                pair_list.append((in_mask[i], in_mask[j]))
            if len(pair_list) >= max_pairs:
                break
        
        for u, v in pair_list:
            if time.time() > deadline - 1.0:
                break
            
            temp_mask = mask & ~(1 << u) & ~(1 << v)
            
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 3:
                max_triples = 200
                triple_count = 0
                for i2 in range(len(valid_candidates)):
                    if triple_count >= max_triples:
                        break
                    x = valid_candidates[i2]
                    temp_x = temp_mask | (1 << x)
                    for j2 in range(i2 + 1, len(valid_candidates)):
                        if triple_count >= max_triples:
                            break
                        y = valid_candidates[j2]
                        temp_xy = temp_x | (1 << y)
                        for k2 in range(j2 + 1, len(valid_candidates)):
                            if triple_count >= max_triples:
                                break
                            z = valid_candidates[k2]
                            if is_valid_add(temp_xy, z):
                                return temp_mask | (1 << x) | (1 << y) | (1 << z)
                            triple_count += 1
        return None

    # Branch and bound for maximum independent set in 3-uniform hypergraph
    def branch_and_bound(start_mask, start_size, remaining_candidates, depth=0):
        """
        B&B solver: try to extend start_mask by adding elements from remaining_candidates.
        Returns (best_mask, best_size) found.
        """
        nonlocal best_mask, best_size
        if time.time() > deadline:
            return (start_mask, start_size)
        
        if not remaining_candidates:
            if start_size > best_size:
                best_size = start_size
                best_mask = start_mask
            return (start_mask, start_size)
        
        # Upper bound: current size + all remaining candidates
        upper_bound = start_size + len(remaining_candidates)
        if upper_bound <= best_size:
            return (start_mask, start_size)
        
        # If we can add all remaining, do it greedily
        if len(remaining_candidates) <= 3:
            mask = start_mask
            size = start_size
            for x in remaining_candidates:
                if is_valid_add(mask, x):
                    mask |= (1 << x)
                    size += 1
            if size > best_size:
                best_size = size
                best_mask = mask
            return (mask, size)
        
        # Try adding the first candidate
        x = remaining_candidates[0]
        rest = remaining_candidates[1:]
        
        # Option 1: Include x
        if is_valid_add(start_mask, x):
            new_mask = start_mask | (1 << x)
            # Filter rest to only those compatible with new_mask
            compatible_rest = [y for y in rest if is_valid_add(new_mask, y)]
            result = branch_and_bound(new_mask, start_size + 1, compatible_rest, depth + 1)
            if result[1] > best_size:
                best_size = result[1]
                best_mask = result[0]
        
        # Option 2: Exclude x
        result = branch_and_bound(start_mask, start_size, rest, depth + 1)
        if result[1] > best_size:
            best_size = result[1]
            best_mask = result[0]
        
        return (best_mask, best_size)

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

        # Try B&B on a subset of candidates if we have time
        if time.time() < deadline - 10.0:
            in_mask = [x for x in range(N) if (S_mask >> x) & 1]
            not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]
            
            # Filter not_in_mask to only those that can be added to S_mask
            addable = [x for x in not_in_mask if is_valid_add(S_mask, x)]
            
            if len(addable) > 0 and len(addable) <= 25:
                # Run B&B to try to add more elements
                result = branch_and_bound(S_mask, current_size, addable)
                if result[1] > current_size:
                    S_mask = result[0]
                    current_size = result[1]
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                    no_improve_count = 0
                    stuck_count = 0
                    improved = True
                    continue

        # 1-for-1 swap to change structure (try multiple, pick best)
        improved_structure = False
        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]
        random.shuffle(in_mask)

        best_swap_mask = None
        best_swap_score = -1
        
        for u in in_mask:
            if time.time() > deadline:
                break
            temp_mask = S_mask & ~(1 << u)
            # Score this swap by how many elements we could then add
            addable_after = 0
            for v in not_in_mask:
                if is_valid_add(temp_mask, v):
                    addable_after += 1
                    if addable_after > best_swap_score:
                        best_swap_score = addable_after
                        best_swap_mask = temp_mask | (1 << v)
                        improved_structure = True

        if best_swap_mask is not None:
            S_mask = best_swap_mask
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart with perturbation
                S_mask = best_mask
                stuck_count = 0
                no_improve_count = 0

                # Perturb with 1-3 random swaps
                in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]
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
                        not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]

    # Convert best_mask back to list of 1-indexed elements
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return result if result else [1]
