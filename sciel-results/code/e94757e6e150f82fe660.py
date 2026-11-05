import random
import time

def construct():
    start_time = time.time()
    deadline = 135.0  # Keep 15s margin before 150s limit

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

    def make_greedy(order):
        """Build a Sidon set by adding elements one at a time, 
        prioritizing differences in 'order'."""
        A = [0]
        diffs = set()
        
        # Try to add elements that cover missing differences
        # We'll iterate through candidate x values in a smart order
        # For each x, check if adding it maintains Sidon property
        # and if it helps cover a missing difference
        
        # Strategy: for each missing d, try to find smallest x such that
        # x - a = d for some a in A, and A U {x} is Sidon
        # Or x = a + d for some a in A
        
        # Alternative: just greedily add smallest x that maintains Sidon
        # and covers a missing difference
        
        # Let's try: for x in 1, 2, 3, ...:
        #   if A U {x} is Sidon and (x - a for some a in A) covers a missing d:
        #     add x
        #   else if A U {x} is Sidon and len(A) < some threshold:
        #     maybe add x (to build up the set)
        
        # Actually, the simplest effective greedy:
        # Add x if it maintains Sidon and adds at least one new needed difference
        # If no such x found, add smallest x that maintains Sidon (to expand set)
        
        for x in range(1, 50000):
            if time.time() > deadline - 30:
                break
            # Check if A U {x} is Sidon
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
            
            # Check if this adds a needed difference
            new_needed = new_diffs & NEED - diffs
            if new_needed:
                A.append(x)
                diffs = new_diffs
                if NEED.issubset(diffs):
                    break
            elif len(A) < 15:
                # Add some elements to build up the set
                A.append(x)
                diffs = new_diffs
                if NEED.issubset(diffs):
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
            if time.time() > deadline - 30:
                break
            # Try to find smallest x such that x - a = d for some a in A
            # and A U {x} is Sidon
            found = False
            for a in sorted(A, reverse=True):
                x = a + d
                if x <= 0:
                    continue
                # Check Sidon
                new_diffs = diffs.copy()
                ok = True
                for b in A:
                    dd = x - b
                    if dd in new_diffs:
                        ok = False
                        break
                    new_diffs.add(dd)
                if not ok:
                    continue
                # Also check if x is not already in A
                if x in A:
                    continue
                A.append(x)
                diffs = new_diffs
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
                remaining -= diffs
                if not remaining:
                    break
        
        return A

    # Initial best from simple greedy
    best_a = make_greedy(None)
    best_cost = get_cost(best_a)
    
    # Try multiple orderings
    orderings = [
        list(range(1, 31)),           # natural
        list(range(30, 0, -1)),       # descending
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 1],  # prime-biased
        [1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16],  # interleaved
    ]
    
    for order in orderings:
        if time.time() > deadline - 60:
            break
        a = greedy_with_order(order)
        cost = get_cost(a)
        if cost < best_cost:
            best_cost = cost
            best_a = list(a)
    
    # Local Search (Simulated Annealing)
    cur = list(best_a)
    cur_cost = get_cost(cur)
    
    temp = 200.0
    cool = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        a = list(cur)
        # Random move
        mt = random.randint(0, 3)
        
        if mt == 0:  # Modify
            if len(a) > 1:
                idx = random.randint(1, len(a) - 1)
                delta = random.randint(-20, 20)
                a[idx] += delta
                if a[idx] < 1:
                    a[idx] = random.randint(1, max(a) + 10)
        
        elif mt == 1:  # Swap
            if len(a) > 2:
                i, j = random.sample(range(1, len(a)), 2)
                a[i], a[j] = a[j], a[i]
        
        elif mt == 2:  # Insert
            if len(a) < MAXN:
                val = random.randint(1, max(a) + 20)
                a.append(val)
        
        else:  # Remove
            if len(a) > 1:
                idx = random.randint(1, len(a) - 1)
                a.pop(idx)
        
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
            # Metropolis
            delta_cost = new_cost - cur_cost
            if delta_cost == 0:
                accept = True
            else:
                accept = random.random() < (1.0 / (1.0 + delta_cost / (temp + 1e-9)))
            
            if accept:
                cur = a
                cur_cost = new_cost
        
        temp *= cool

    # Final validation
    if is_valid(best_a):
        return best_a
    
    # If best_a is somehow invalid, try to repair or return cur
    if is_valid(cur):
        return cur
        
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
