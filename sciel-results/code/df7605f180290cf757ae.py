import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit

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
        diffs = get_diffs(s)
        return NEED.issubset(diffs)

    def is_valid(s):
        return valid_sidon(s) and has_all_need(s)

    def get_cost(s):
        s = sorted(set(s))
        if 0 not in s or len(s) > MAXN:
            return 10**9
        diffs = set()
        dup = False
        for i in range(len(s)):
            si = s[i]
            for j in range(i + 1, len(s)):
                d = s[j] - si
                if d in diffs:
                    dup = True
                    break
                diffs.add(d)
            if dup:
                break
        if dup:
            return 10**8 + max(s)
        missing = NEED - diffs
        if missing:
            return 10**6 + len(missing) * 1000 + max(s)
        return max(s)

    def make_greedy():
        """Build a Sidon set greedily, adding smallest x that maintains Sidon property."""
        A = [0]
        diffs = set()
        for x in range(1, 10000):
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
                break
        return A

    def local_search_reduce_max(a, time_limit=10.0):
        """Try to reduce the maximum element while maintaining validity."""
        a = sorted(set(a))
        if not is_valid(a):
            return a
        
        cur = list(a)
        cur_max = max(cur)
        best = list(cur)
        best_max = cur_max
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Find the largest element
            max_idx = len(cur) - 1
            max_val = cur[max_idx]
            
            # Try to move the largest element to a smaller position
            found = False
            for new_val in range(max_val - 1, 0, -1):
                if new_val in cur:
                    continue
                test = list(cur)
                test[max_idx] = new_val
                test = sorted(set(test))
                if is_valid(test):
                    cur = test
                    cur_max = new_val
                    if cur_max < best_max:
                        best = list(cur)
                        best_max = cur_max
                    found = True
                    break
            
            if not found:
                # Try moving other elements to free up space
                for idx in range(len(cur) - 2, 0, -1):
                    if cur[idx] >= max_val:
                        continue
                    for new_val in range(cur[idx] - 1, 0, -1):
                        if new_val in cur:
                            continue
                        test = list(cur)
                        test[idx] = new_val
                        test = sorted(set(test))
                        if is_valid(test):
                            cur = test
                            cur_max = max(cur)
                            if cur_max < best_max:
                                best = list(cur)
                                best_max = cur_max
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Try a random perturbation
                idx = random.randint(1, len(cur) - 1)
                delta = random.randint(-5, 5)
                new_val = cur[idx] + delta
                if new_val <= 0 or new_val in cur:
                    continue
                test = list(cur)
                test[idx] = new_val
                test = sorted(set(test))
                if is_valid(test):
                    cur = test
                    cur_max = max(cur)
                    if cur_max < best_max:
                        best = list(cur)
                        best_max = cur_max
        
        return best

    # Initial greedy construction
    best_a = make_greedy()
    best_cost = get_cost(best_a)
    
    # Try multiple random restarts with different strategies
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed starting point
        if restart == 0:
            start_a = list(best_a)
        else:
            # Start from a slightly modified greedy
            A = [0]
            diffs = set()
            for x in range(1, 10000):
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
                # Randomly skip some elements
                if random.random() < 0.1 and len(A) < 5:
                    continue
                A.append(x)
                diffs = new_diffs
                if NEED.issubset(diffs):
                    break
            start_a = A
        
        if not is_valid(start_a):
            continue
        
        # Run local search to reduce max
        result = local_search_reduce_max(start_a, time_limit=min(5.0, max(0.5, deadline - time.time() - 5)))
        
        if is_valid(result):
            cost = get_cost(result)
            if cost < best_cost:
                best_a = result
                best_cost = cost
    
    # Final validation
    if is_valid(best_a):
        return best_a
    
    # Fallback: return a known valid set
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
