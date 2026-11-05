import time
import random
import bisect

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Precompute conflict tables for speed
    div_sums = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2 * a, 2 * n + 1, a):
            div_sums[a].add(s)
    
    def is_valid_add(S_list, m):
        """Check if adding m to S_list (sorted) maintains the property. Returns True if valid."""
        pos = bisect.bisect_left(S_list, m)
        L = len(S_list)
        
        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        div_m = div_sums[m]
        for j in range(pos, L):
            b = S_list[j]
            for k in range(j, L):
                c = S_list[k]
                if (b + c) in div_m:
                    return False
        
        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a in S with a < m, c in S with c >= m
        for i in range(0, pos):
            a = S_list[i]
            div_a = div_sums[a]
            for k in range(pos, L):
                c = S_list[k]
                if (m + c) in div_a:
                    return False
        
        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b in S with b <= m
        for i in range(0, pos):
            a = S_list[i]
            div_a = div_sums[a]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) in div_a:
                    return False
        
        return True

    def conflict_count(S_list, m):
        """Count how many constraints m would violate if added to S_list."""
        pos = bisect.bisect_left(S_list, m)
        L = len(S_list)
        count = 0
        
        div_m = div_sums[m]
        for j in range(pos, L):
            b = S_list[j]
            for k in range(j, L):
                c = S_list[k]
                if (b + c) in div_m:
                    count += 1
        
        for i in range(0, pos):
            a = S_list[i]
            div_a = div_sums[a]
            for k in range(pos, L):
                c = S_list[k]
                if (m + c) in div_a:
                    count += 1
        
        for i in range(0, pos):
            a = S_list[i]
            div_a = div_sums[a]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) in div_a:
                    count += 1
        
        return count

    def local_search(S_list):
        """Try to add more elements via local search. Returns improved list."""
        S = list(S_list)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        
        while time.time() < deadline:
            improved = False
            # Try candidates in order of increasing conflict count (easiest to add first)
            scored = [(conflict_count(S, m), m) for m in not_in_S]
            scored.sort(key=lambda x: (x[0], -x[1]))  # ascending conflicts, descending value
            
            for _, add_m in scored:
                if time.time() > deadline:
                    break
                if is_valid_add(S, add_m):
                    pos = bisect.bisect_left(S, add_m)
                    S.insert(pos, add_m)
                    in_S.add(add_m)
                    not_in_S.remove(add_m)
                    improved = True
                    break
            
            if not improved:
                # Try simple ascending/descending orders
                for add_m in sorted(not_in_S, reverse=True):
                    if time.time() > deadline:
                        break
                    if is_valid_add(S, add_m):
                        pos = bisect.bisect_left(S, add_m)
                        S.insert(pos, add_m)
                        in_S.add(add_m)
                        not_in_S.remove(add_m)
                        improved = True
                        break
                if not improved:
                    for add_m in sorted(not_in_S):
                        if time.time() > deadline:
                            break
                        if is_valid_add(S, add_m):
                            pos = bisect.bisect_left(S, add_m)
                            S.insert(pos, add_m)
                            in_S.add(add_m)
                            not_in_S.remove(add_m)
                            improved = True
                            break
            
            if not improved:
                break
        return S

    def two_for_one_swap(S_list):
        """Try removing 2 elements and adding 1 different element to enable further growth."""
        S = list(S_list)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        improved = False
        
        # Try pairs of removals
        S_size = len(S)
        for i in range(S_size):
            if time.time() > deadline:
                break
            for j in range(i + 1, S_size):
                if time.time() > deadline:
                    break
                S_test = [x for k, x in enumerate(S) if k != i and k != j]
                # Try adding candidates not in original S
                for add_m in not_in_S:
                    if time.time() > deadline:
                        break
                    if is_valid_add(S_test, add_m):
                        new_S = S_test + [add_m]
                        new_S.sort()
                        # Check if this enables further additions
                        new_in_S = set(new_S)
                        new_not_in_S = [m for m in range(1, n + 1) if m not in new_in_S]
                        extra = 0
                        for check_m in sorted(new_not_in_S, reverse=True):
                            if time.time() > deadline:
                                break
                            if is_valid_add(new_S, check_m):
                                extra += 1
                                if extra >= 2:
                                    break
                        if extra >= 1:
                            S = new_S
                            in_S = set(S)
                            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        return S

    # Strategy: Start with {171,...,256}, try multiple strategies to add elements
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)
    
    random.seed(42)
    
    # Multiple random restarts with different addition orders
    for restart in range(400):
        if time.time() > deadline:
            break
        
        S = list(base_set)
        candidates = list(range(1, 171))
        
        r = restart % 10
        if r == 0:
            # Descending (larger first - fewer constraints)
            candidates.sort(reverse=True)
        elif r == 1:
            # Ascending
            candidates.sort()
        elif r == 2:
            # Random
            random.shuffle(candidates)
        elif r == 3:
            # Descending with slight randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -x + random.random() * 20)
        elif r == 4:
            # Random with bias toward larger
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -x * 0.8 + random.random() * 50)
        elif r == 5:
            # Random with bias toward smaller
            random.shuffle(candidates)
            candidates.sort(key=lambda x: x * 0.5 + random.random() * 30)
        elif r == 6:
            # Pure random
            random.shuffle(candidates)
        elif r == 7:
            # Conflict-aware: pick candidate with fewest conflicts first
            S_temp = list(base_set)
            remaining = list(range(1, 171))
            while remaining:
                if time.time() > deadline:
                    break
                scored = [(conflict_count(S_temp, m), m) for m in remaining]
                scored.sort(key=lambda x: (x[0], -x[1]))
                _, best_m = scored[0]
                if is_valid_add(S_temp, best_m):
                    pos = bisect.bisect_left(S_temp, best_m)
                    S_temp.insert(pos, best_m)
                    remaining.remove(best_m)
            S = S_temp
            continue  # Skip the greedy loop below
        elif r == 8:
            # Random with gap-filling bias: prefer numbers that fill gaps
            gaps = []
            prev = 0
            for x in sorted(base_set):
                gaps.append((x - prev - 1, x))
                prev = x
            gaps.append((n - prev, n + 1))
            # Bias toward mid-range numbers
            random.shuffle(candidates)
            candidates.sort(key=lambda x: abs(x - 128) + random.random() * 40)
        else:
            # Weighted random: smaller numbers get higher weight
            weights = [1.0 / (m + 1) for m in candidates]
            S_temp = list(base_set)
            remaining = list(range(1, 171))
            while remaining:
                if time.time() > deadline:
                    break
                w = [1.0 / (m + 1) for m in remaining]
                total_w = sum(w)
                r_val = random.random() * total_w
                cum = 0
                chosen = remaining[-1]
                for idx, m in enumerate(remaining):
                    cum += w[idx]
                    if r_val <= cum:
                        chosen = m
                        break
                if is_valid_add(S_temp, chosen):
                    pos = bisect.bisect_left(S_temp, chosen)
                    S_temp.insert(pos, chosen)
                    remaining.remove(chosen)
            S = S_temp
            continue
        
        # Greedy add for non-special cases
        for m in candidates:
            if time.time() > deadline:
                break
            if is_valid_add(S, m):
                pos = bisect.bisect_left(S, m)
                S.insert(pos, m)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Local search: try to add more elements
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # Two-for-one swap search
    best_S = two_for_one_swap(best_S)
    best_size = len(best_S)
    
    # Another local search pass after swaps
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # One more two-for-one swap
    best_S = two_for_one_swap(best_S)
    best_size = len(best_S)
    
    # Final local search
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # Final verification
    S_final = sorted(best_S)
    valid = True
    for i, a in enumerate(S_final):
        for j in range(i + 1, len(S_final)):
            b = S_final[j]
            for k in range(j, len(S_final)):
                c = S_final[k]
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(171, 257))
    
    return S_final
