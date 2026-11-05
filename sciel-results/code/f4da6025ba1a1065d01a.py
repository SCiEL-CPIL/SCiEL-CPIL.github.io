import random
import time
from collections import defaultdict

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

    def backtrack_search(max_elem, num_elements):
        """
        Constraint-based backtracking search.
        We build the set incrementally, ensuring Sidon property at each step.
        Variable ordering: choose next element to minimize conflicts.
        Value ordering: try smaller values first (to minimize diameter).
        
        Key insight: if we have elements a1 < a2 < ... < ak, then for a new
        element x > ak, the differences x - ai must all be new (not in diffs).
        This is a hard constraint that prunes the search space significantly.
        """
        best = [None]
        best_cost = [10**9]
        
        def dfs(A, diffs, covered):
            if time.time() > deadline - 5:
                return
            if covered == NEED:
                cost = max(A) if A else 0
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    best[0] = list(A)
                return
            
            if len(A) >= num_elements:
                return
            
            # Determine which differences are still needed
            missing = NEED - covered
            
            # For each candidate x, check if it's valid
            # We try x from A[-1]+1 up to max_elem
            last = A[-1] if A else 0
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, max_elem + 1):
                # Check Sidon property
                ok = True
                new_diffs = set()
                for a in A:
                    d = x - a
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    new_covered = covered | new_diffs
                    # Only add if it helps cover missing differences
                    # or if we need more elements
                    if (new_covered - covered) & missing:
                        valid_candidates.append((x, new_diffs, new_covered))
            
            # If no candidates help, we might need to add elements that don't
            # directly help but allow future coverage. For now, just try all
            # valid candidates sorted by x.
            if not valid_candidates:
                # Try adding any valid element to expand the set
                for x in range(last + 1, max_elem + 1):
                    ok = True
                    new_diffs = set()
                    for a in A:
                        d = x - a
                        if d in diffs or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        new_covered = covered | new_diffs
                        valid_candidates.append((x, new_diffs, new_covered))
                        break
            
            # Try candidates in order of x (smallest first)
            for x, new_diffs, new_covered in valid_candidates:
                if time.time() > deadline - 5:
                    return
                dfs(A + [x], diffs | new_diffs, new_covered)
        
        dfs([0], set(), set())
        return best[0]

    def iterative_deepening(max_elem_start, max_elem_end, max_steps=100000):
        """
        Try increasing diameter bounds until we find a valid solution.
        Use backtracking with time limit.
        """
        for max_elem in range(max_elem_start, max_elem_end + 1):
            if time.time() > deadline - 10:
                break
            # Try with different numbers of elements
            for num_elements in range(4, min(MAXN, 20) + 1):
                if time.time() > deadline - 5:
                    break
                result = backtrack_search(max_elem, num_elements)
                if result is not None:
                    return result
        return None

    def greedy_construct():
        """Fast greedy to get an initial upper bound."""
        A = [0]
        diffs = set()
        covered = set()
        
        for x in range(1, 2000):
            new_diffs = set()
            ok = True
            for a in A:
                d = x - a
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                A.append(x)
                diffs |= new_diffs
                covered |= new_diffs
                if NEED.issubset(covered):
                    return A
        return A

    # Phase 1: Get an initial solution via greedy
    greedy_sol = greedy_construct()
    if greedy_sol and is_valid(greedy_sol):
        best_sol = greedy_sol
        best_cost = max(greedy_sol)
    else:
        best_sol = None
        best_cost = 10**9

    # Phase 2: Try to improve with backtracking search
    # Start from a reasonable upper bound
    if best_sol:
        upper = best_cost
        # Try to find smaller solutions
        for target_max in range(50, upper + 1):
            if time.time() > deadline - 15:
                break
            # Use backtracking with this max element bound
            # Try different sizes
            for num_elements in range(4, min(MAXN, 16) + 1):
                if time.time() > deadline - 10:
                    break
                result = backtrack_search(target_max, num_elements)
                if result is not None:
                    if is_valid(result) and max(result) < best_cost:
                        best_sol = result
                        best_cost = max(result)
                        break
            if best_cost <= target_max:
                # We found a solution with this diameter, try smaller
                continue
    
    # Phase 3: Local search to polish
    if best_sol:
        cur = list(best_sol)
        cur_cost = get_cost(cur)
        temp = 50.0
        cool = 0.9995
        no_improve = 0
        
        while time.time() < deadline - 5:
            a = list(cur)
            mt = random.randint(0, 2)
            
            if mt == 0 and len(a) > 1:
                idx = random.randint(1, len(a) - 1)
                delta = random.randint(-15, 15)
                a[idx] += delta
                if a[idx] < 1:
                    a[idx] = random.randint(1, max(a) + 5)
            elif mt == 1 and len(a) > 2:
                i, j = random.sample(range(1, len(a)), 2)
                a[i], a[j] = a[j], a[i]
            elif mt == 2:
                if len(a) < MAXN:
                    val = random.randint(1, max(a) + 10)
                    a.append(val)
                elif len(a) > 2:
                    idx = random.randint(1, len(a) - 1)
                    a.pop(idx)
            
            a = sorted(set(a))
            if 0 not in a:
                a.append(0)
            
            new_cost = get_cost(a)
            
            if new_cost < cur_cost:
                cur = a
                cur_cost = new_cost
                no_improve = 0
                if cur_cost < best_cost:
                    best_sol = list(cur)
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
                    no_improve += 1
                else:
                    no_improve += 1
            
            temp *= cool
            if no_improve > 5000:
                break

    # Final validation
    if best_sol and is_valid(best_sol):
        return best_sol
    
    # Fallback
    A = [0]
    diffs = set()
    for x in range(1, 5000):
        new_diffs = set()
        ok = True
        for a in A:
            d = x - a
            if d in diffs or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            A.append(x)
            diffs |= new_diffs
            if NEED.issubset(diffs):
                return A
    return A
