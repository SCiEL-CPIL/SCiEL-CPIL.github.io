import time
import random

def construct():
    start = time.time()
    deadline = start + 115.0
    best = [1, 2, 5, 11]
    best_len = 4
    
    def sums_ok(a, sums_used):
        """Check if all pairwise sums are distinct."""
        seen = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s = a[i] + a[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    def build_sums(a):
        s = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                s.add(a[i] + a[j])
        return s
    
    def greedy_extend(a, sums_used):
        """Greedy: try to add elements in order from 12 to 284."""
        a = list(a)
        sums_used = set(sums_used)
        for x in range(12, 285):
            if x in a:
                continue
            new_sums = set()
            ok = True
            for y in a:
                s = x + y
                if s in sums_used or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            s2 = 2 * x
            if s2 in sums_used or s2 in new_sums:
                ok = False
            else:
                new_sums.add(s2)
            if ok:
                a.append(x)
                sums_used |= new_sums
        return a, sums_used
    
    def local_search(a, sums_used, max_kicks=5000):
        """Local search: try to improve by swapping elements."""
        a = list(a)
        sums_used = set(sums_used)
        n = len(a)
        a_set = set(a)
        
        def try_remove_add(a, sums_used, remove_idx, add_val):
            """Try removing a[remove_idx] and adding add_val."""
            if add_val in a_set:
                return None, None
            # Check if we can add add_val to a without remove_idx
            a_temp = [x for i, x in enumerate(a) if i != remove_idx]
            sums_temp = set()
            for i in range(len(a_temp)):
                for j in range(i, len(a_temp)):
                    sums_temp.add(a_temp[i] + a_temp[j])
            
            # Check if add_val fits
            ok = True
            for y in a_temp:
                s = add_val + y
                if s in sums_temp:
                    ok = False
                    break
            if ok:
                s2 = 2 * add_val
                if s2 in sums_temp:
                    ok = False
            if not ok:
                return None, None
            
            new_a = a_temp + [add_val]
            new_sums = sums_temp | {add_val + y for y in a_temp} | {2 * add_val}
            return new_a, new_sums
        
        improved = True
        kicks = 0
        while improved and kicks < max_kicks and time.time() < deadline:
            improved = False
            kicks += 1
            random.shuffle(a)
            for i in range(n):
                x = a[i]
                # Try to find a replacement
                for new_x in random.sample(range(1, 285), min(20, 284)):
                    if new_x in a_set or new_x == x:
                        continue
                    new_a, new_sums = try_remove_add(a, sums_used, i, new_x)
                    if new_a is not None:
                        # Check if this actually improves (same size, but different structure)
                        # For now, accept any valid swap
                        a = new_a
                        sums_used = new_sums
                        a_set = set(a)
                        improved = True
                        break
                if improved:
                    break
        return a, sums_used
    
    def backtrack_search(a, sums_used, target_len, depth=0):
        """Backtracking search to reach target_len elements."""
        if time.time() >= deadline:
            return a, sums_used
        if len(a) >= target_len:
            return a, sums_used
        
        n = len(a)
        a_set = set(a)
        
        # Try adding elements
        candidates = [x for x in range(12, 285) if x not in a_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() >= deadline:
                break
            # Check if x can be added
            ok = True
            new_sums = set()
            for y in a:
                s = x + y
                if s in sums_used or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            s2 = 2 * x
            if ok and (s2 in sums_used or s2 in new_sums):
                ok = False
            if not ok:
                continue
            
            new_a = a + [x]
            new_sums = sums_used | new_sums | {s2}
            
            result_a, result_sums = backtrack_search(new_a, new_sums, target_len, depth + 1)
            if len(result_a) > len(a):
                return result_a, result_sums
        
        return a, sums_used
    
    # Strategy: multiple rounds of greedy + local search + backtracking
    for round_num in range(30):
        if time.time() >= deadline:
            break
        
        # Start with prescribed elements
        a = [1, 2, 5, 11]
        sums_used = build_sums(a)
        
        # Greedy extend
        a, sums_used = greedy_extend(a, sums_used)
        
        # Local search to improve
        a, sums_used = local_search(a, sums_used, max_kicks=2000)
        
        if len(a) > best_len:
            best = a
            best_len = len(a)
        
        # Try backtracking to push further
        target = min(best_len + 3, 22)
        a_bt, sums_bt = backtrack_search(a, sums_used, target)
        if len(a_bt) > best_len:
            best = a_bt
            best_len = len(a_bt)
        
        # If we hit 20+, try to push to 21
        if best_len >= 20 and time.time() < deadline:
            a_bt2, sums_bt2 = backtrack_search(a, sums_used, best_len + 1)
            if len(a_bt2) > best_len:
                best = a_bt2
                best_len = len(a_bt2)
    
    # Final greedy on best to see if we can squeeze more
    if time.time() < deadline - 5:
        a_final, sums_final = greedy_extend(best, build_sums(best))
        if len(a_final) > best_len:
            best = a_final
            best_len = len(a_final)
    
    return best
# EVOLVE-BLOCK-END
