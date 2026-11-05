import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Keep 10s margin before 150s limit

    NEED = set(range(1, 31))
    MAXN = 32

    def get_diffs(s):
        s = sorted(set(s))
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - si)
        return diffs

    def valid_sidon(s):
        s = sorted(set(s))
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                d = s[j] - si
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def has_all_need(s):
        s = sorted(set(s))
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                diffs.add(s[j] - si)
        return NEED.issubset(diffs)

    def is_valid(s):
        return valid_sidon(s) and has_all_need(s)

    def get_cost(s):
        s = sorted(set(s))
        if 0 not in s or len(s) > MAXN:
            return 10**9
        diffs = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                d = s[j] - si
                if d in diffs:
                    return 10**8 + max(s)
                diffs.add(d)
        missing = NEED - diffs
        if missing:
            return 10**6 + len(missing) * 1000 + max(s)
        return max(s)

    def can_add(x, A, diffs):
        """Check if x can be added to A while maintaining Sidon property."""
        for a in A:
            d = x - a
            if d <= 0:
                continue
            if d in diffs:
                return False
        return True

    def greedy_smart():
        """Smart greedy that prioritizes covering missing differences."""
        A = [0]
        diffs = set()
        remaining = set(NEED)
        
        # First, try to cover small differences efficiently
        for target_d in sorted(remaining):
            if time.time() > deadline - 40:
                break
            if target_d not in remaining:
                continue
            
            # Try all possible pairs (a, x) where x - a = target_d
            # Prefer smaller x values
            candidates = []
            for a in A:
                x = a + target_d
                if x > 0 and x not in A:
                    # Check if x can be added
                    if can_add(x, A, diffs):
                        candidates.append((x, a))
            
            if candidates:
                # Pick the candidate with smallest x
                candidates.sort()
                x, a = candidates[0]
                # Add x
                for b in A:
                    d = x - b
                    if d > 0:
                        diffs.add(d)
                A.append(x)
                A.sort()
                # Update remaining
                remaining -= diffs
            
            # Also try to add elements that create multiple needed differences
            if remaining and time.time() < deadline - 35:
                for x in range(1, max(A) + target_d + 10):
                    if time.time() > deadline - 35:
                        break
                    if x in A:
                        continue
                    if not can_add(x, A, diffs):
                        continue
                    
                    new_diffs = set()
                    for a in A:
                        d = x - a
                        if d > 0:
                            new_diffs.add(d)
                    
                    new_needed = new_diffs & remaining
                    if len(new_needed) >= 2:
                        for d in new_diffs:
                            diffs.add(d)
                        A.append(x)
                        A.sort()
                        remaining -= diffs
                        break
        
        # Fill in any remaining differences
        if remaining:
            for x in range(1, 50000):
                if time.time() > deadline - 30:
                    break
                if x in A:
                    continue
                if not can_add(x, A, diffs):
                    continue
                
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d > 0:
                        new_diffs.add(d)
                
                if new_diffs & remaining:
                    for d in new_diffs:
                        diffs.add(d)
                    A.append(x)
                    A.sort()
                    remaining -= diffs
                    if not remaining:
                        break
        
        return A

    def greedy_with_order(order):
        """Greedy that tries to cover differences in a specific order."""
        A = [0]
        diffs = set()
        remaining = set(NEED)
        
        for d in order:
            if d not in remaining:
                continue
            if time.time() > deadline - 40:
                break
            # Try to find smallest x such that x - a = d for some a in A
            # and A U {x} is Sidon
            found = False
            for a in sorted(A, reverse=True):
                x = a + d
                if x <= 0 or x in A:
                    continue
                # Check Sidon
                ok = can_add(x, A, diffs)
                if not ok:
                    continue
                # Add x
                for b in A:
                    dd = x - b
                    if dd > 0:
                        diffs.add(dd)
                A.append(x)
                A.sort()
                remaining.discard(d)
                # Also remove any other d' in remaining that are now covered
                for d2 in list(remaining):
                    if d2 in diffs:
                        remaining.discard(d2)
                found = True
                break
            if not remaining:
                break
        
        # Fill in any remaining differences with greedy
        if remaining:
            for x in range(1, 50000):
                if time.time() > deadline - 30:
                    break
                if x in A:
                    continue
                if not can_add(x, A, diffs):
                    continue
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d > 0:
                        new_diffs.add(d)
                if new_diffs & remaining:
                    for d in new_diffs:
                        diffs.add(d)
                    A.append(x)
                    A.sort()
                    remaining -= diffs
                    if not remaining:
                        break
        
        return A

    def local_search_reduce_max(cur, cur_cost, time_limit):
        """Local search focused on reducing the maximum element."""
        best_a = list(cur)
        best_cost = cur_cost
        temp = 50.0
        cool = 0.9995
        steps = 0
        
        while time.time() < time_limit:
            steps += 1
            a = list(cur)
            mt = random.randint(0, 4)
            
            if mt == 0:  # Modify a non-max element
                if len(a) > 2:
                    # Don't modify the last (max) element in this move type
                    idx = random.randint(1, len(a) - 2)
                    delta = random.randint(-15, 15)
                    new_val = a[idx] + delta
                    if new_val < 1:
                        new_val = random.randint(1, a[idx])
                    elif new_val == a[idx]:
                        continue
                    a[idx] = new_val
            
            elif mt == 1:  # Try to reduce the max element
                if len(a) > 1:
                    max_val = a[-1]
                    # Try smaller values
                    for new_max in range(max_val - 1, max(1, max_val - 30), -1):
                        if new_max in a:
                            continue
                        a[-1] = new_max
                        a = sorted(set(a))
                        if 0 not in a:
                            a.append(0)
                        new_cost = get_cost(a)
                        if new_cost < cur_cost:
                            cur = a
                            cur_cost = new_cost
                            if cur_cost < best_cost:
                                best_a = list(cur)
                                best_cost = cur_cost
                        break
            
            elif mt == 2:  # Swap two elements
                if len(a) > 2:
                    i, j = random.sample(range(1, len(a)), 2)
                    a[i], a[j] = a[j], a[i]
            
            elif mt == 3:  # Remove an element (if we have more than needed)
                if len(a) > 5:
                    idx = random.randint(1, len(a) - 1)
                    a.pop(idx)
            
            else:  # Add a small element
                if len(a) < MAXN:
                    val = random.randint(1, max(a) + 10)
                    a.append(val)
            
            # Clean up
            a = sorted(set(a))
            if 0 not in a:
                a.append(0)
            
            new_cost = get_cost(a)
            
            if new_cost < cur_cost:
                cur = a
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_a = list(cur)
                    best_cost = cur_cost
            else:
                delta_cost = new_cost - cur_cost
                if delta_cost == 0:
                    accept = True
                else:
                    accept = random.random() < (1.0 / (1.0 + delta_cost / (temp + 1e-9)))
                
                if accept:
                    cur = a
                    cur_cost = new_cost
            
            temp *= cool
        
        return best_a, best_cost

    # Generate multiple diverse initial solutions
    best_a = greedy_smart()
    best_cost = get_cost(best_a)
    
    # Try multiple orderings
    orderings = [
        list(range(1, 31)),           # natural
        list(range(30, 0, -1)),       # descending
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 1],
        [1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
    
    for order in orderings:
        if time.time() > deadline - 80:
            break
        a = greedy_with_order(order)
        cost = get_cost(a)
        if cost < best_cost:
            best_cost = cost
            best_a = list(a)
    
    # Local search with time budget
    if time.time() < deadline - 20:
        search_end = deadline - 5
        best_a, best_cost = local_search_reduce_max(best_a, best_cost, search_end)
    
    # Final validation
    if is_valid(best_a):
        return best_a
    
    # Fallback: return a known valid set (large diameter)
    A = [0]
    diffs = set()
    for x in range(1, 20000):
        new_diffs = diffs.copy()
        ok = True
        for a in A:
            d = x - a
            if d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        A.append(x)
        diffs = new_diffs
        if NEED.issubset(diffs):
            return A
            
    return A
