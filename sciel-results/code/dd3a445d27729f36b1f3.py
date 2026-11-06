def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 85  # Safety margin for 90s limit

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict structure:
    # conflicts[x] contains pairs (a, b) such that {x, a, b} is a bad triple.
    # a < b, a != x, b != x
    conflicts = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    conflicts[x].append((a, b))

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    def can_add_fast(x, S_set):
        """Faster can_add using set intersection for the conflict pairs."""
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def potential(S):
        """Count number of elements in [1, 128] \\ S that can be added to S."""
        S_set = S if isinstance(S, set) else set(S)
        count = 0
        for x in range(1, N + 1):
            if x not in S_set:
                if can_add_fast(x, S_set):
                    count += 1
        return count

    def greedy_init(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        """Improve S using Greedy Add, Potential-Guided 1-Swaps, and 2-Swaps."""
        S = set(S)
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add any element (Greedy Add)
            for x in range(1, N + 1):
                if x not in S:
                    if can_add_fast(x, S):
                        S.add(x)
                        improved = True
                        break # Restart loop to try adding more
            
            if improved:
                continue
            
            # 2. If no add possible, try 1-Swap to maximize potential
            current_pot = potential(S)
            best_pot = current_pot
            best_swap = None
            
            S_list = list(S)
            for a in S_list:
                S_minus_a = S - {a}
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if can_add_fast(x, S_minus_a):
                        S_new = S_minus_a | {x}
                        pot = potential(S_new)
                        if pot > best_pot:
                            best_pot = pot
                            best_swap = (a, x)
            
            if best_swap:
                a, x = best_swap
                S.remove(a)
                S.add(x)
                continue
            
            # 3. Try 2-Swaps: remove two elements, add two new ones
            S_list = list(S)
            n = len(S_list)
            if n < 2:
                break
            
            best_2pot = current_pot
            best_2swap = None
            
            # Limit 2-swap search to save time
            max_2swap_pairs = 50
            pair_count = 0
            for i in range(n):
                if time.time() > deadline:
                    break
                a = S_list[i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    b = S_list[j]
                    S_minus_ab = S - {a, b}
                    # Try adding pairs (x, y) not in S
                    for x in range(1, N + 1):
                        if x in S:
                            continue
                        if not can_add_fast(x, S_minus_ab):
                            continue
                        for y in range(x + 1, N + 1):
                            if y in S:
                                continue
                            if not can_add_fast(y, S_minus_ab):
                                continue
                            # Check that x and y don't form a bad triple with each other and S_minus_ab
                            S_new = S_minus_ab | {x, y}
                            # Verify x,y pair is valid with S_minus_ab
                            # can_add_fast already checks against S_minus_ab, but need to check x-y interaction
                            # Actually can_add_fast(x, S_minus_ab) checks x against all pairs in conflicts[x]
                            # where both are in S_minus_ab. It does NOT check x-y pair.
                            # So we need to verify the full set is valid.
                            # Quick check: for each z in S_minus_ab, gcd(x,z) != gcd(y,z) or gcd(x,y) != that
                            # Simpler: just verify the new set
                            valid = True
                            for z in S_minus_ab:
                                gxz = gcd_table[x][z]
                                gyz = gcd_table[y][z]
                                gxy = gcd_table[x][y]
                                if gxz == gyz == gxy:
                                    valid = False
                                    break
                                if gxz == gxy:
                                    # Check if there's another w in S_minus_ab with gcd(y,w)==gxz and gcd(x,w)==gxz
                                    # This is expensive; skip for now and rely on final validation
                                    pass
                                if gyz == gxy:
                                    pass
                            if not valid:
                                continue
                            pot = potential(S_new)
                            if pot > best_2pot:
                                best_2pot = pot
                                best_2swap = (a, b, x, y)
                            pair_count += 1
                            if pair_count >= max_2swap_pairs:
                                break
                        if pair_count >= max_2swap_pairs:
                            break
                    if pair_count >= max_2swap_pairs:
                        break
                if pair_count >= max_2swap_pairs:
                    break
            
            if best_2swap:
                a, b, x, y = best_2swap
                S.discard(a)
                S.discard(b)
                S.add(x)
                S.add(y)
                continue
            else:
                # Stuck in local optimum
                break
        
        return S

    best_set = set()
    best_size = 0
    
    # Different initialization orders
    orders = []
    orders.append(list(range(1, N + 1))) # Ascending
    orders.append(list(range(N, 0, -1))) # Descending
    
    # Random orders
    random.seed(42)
    for _ in range(20):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)
    
    # Degree-sorted orders (high conflict degree first or last)
    degrees = {x: len(conflicts[x]) for x in range(1, N + 1)}
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True))
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x]))

    for order in orders:
        if time.time() > deadline:
            break
        
        S = greedy_init(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # Early termination if we found a very good set
            if best_size >= 32:
                break

    # Final check and return
    # Ensure valid
    # The local_search maintains validity by construction (can_add checks)
    # But just in case, we could verify.
    # Verification is O(N^3) which is fine for N=128 once.
    def is_valid(S):
        elems = sorted(S)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, len(elems)):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    if is_valid(best_set):
        return list(best_set)
    else:
        return [1]
