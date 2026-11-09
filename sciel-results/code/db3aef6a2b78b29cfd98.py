"""Rewritten program: conflict-graph based sum-free subset search with local search."""
import time
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def sum_free_check(S):
        """Return True if S is sum-free (no a+b=c with a,b,c in S, a=b allowed)."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def build_sum_free(arr):
        """Build a large sum-free subset using greedy + local search with swaps."""
        n = len(arr)
        if n == 0:
            return []
        
        elem_to_idx = {v: i for i, v in enumerate(arr)}
        rng = random.Random(12345)
        
        # Greedy initialization with multiple orderings
        def greedy_init(order, current_vals):
            included = set()
            for idx in order:
                c_val = arr[idx]
                ok = True
                # Check: is there a,b in current_vals such that a+b = c_val?
                # i.e., for some a in current_vals, c_val - a is in current_vals
                for a_val in current_vals:
                    if (c_val - a_val) in current_vals:
                        ok = False
                        break
                if ok:
                    # Check: is there b in current_vals such that c_val + b is in current_vals?
                    for b_val in current_vals:
                        if (c_val + b_val) in current_vals:
                            ok = False
                            break
                if ok:
                    included.add(idx)
                    current_vals.add(c_val)
            return included, current_vals
        
        # Multiple greedy starts
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            list(range(n))
        ]
        rng.shuffle(orders[2])
        
        best_included = set()
        best_vals = set()
        for order in orders:
            inc, vals = greedy_init(order, set())
            if len(inc) > len(best_included):
                best_included = inc
                best_vals = vals
        
        # Local search with efficient can_add
        included = set(best_included)
        current_vals = set(best_vals)
        
        def can_add(idx, vals):
            """Check if idx can be added to current without violating sum-free."""
            c_val = arr[idx]
            # Check 1: for all a in vals, c_val - a is not in vals
            for a_val in vals:
                if (c_val - a_val) in vals:
                    return False
            # Check 2: for all b in vals, c_val + b is not in vals
            for b_val in vals:
                if (c_val + b_val) in vals:
                    return False
            return True
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding excluded elements
            excluded = [i for i in range(n) if i not in included]
            rng.shuffle(excluded)
            for idx in excluded:
                if time.time() >= deadline - 5.0:
                    break
                if can_add(idx, current_vals):
                    included.add(idx)
                    current_vals.add(arr[idx])
                    improved = True
            
            # Try swaps: remove one included, add one excluded
            if time.time() >= deadline - 5.0:
                break
            excluded = [i for i in range(n) if i not in included]
            inc_list = list(included)
            rng.shuffle(excluded)
            rng.shuffle(inc_list)
            
            # Try more swaps
            for add_idx in excluded[:min(len(excluded), 50)]:
                if time.time() >= deadline - 5.0:
                    break
                for rem_idx in inc_list[:min(len(inc_list), 30)]:
                    if time.time() >= deadline - 5.0:
                        break
                    # Simulate removal
                    trial_vals = current_vals - {arr[rem_idx]}
                    if can_add(add_idx, trial_vals):
                        included = (included - {rem_idx}) | {add_idx}
                        current_vals = trial_vals | {arr[add_idx]}
                        improved = True
                        break
        
        # Final polish: try more aggressive local search
        if time.time() < deadline - 2.0:
            for _ in range(3):
                if time.time() >= deadline - 2.0:
                    break
                improved = False
                excluded = [i for i in range(n) if i not in included]
                rng.shuffle(excluded)
                for idx in excluded:
                    if time.time() >= deadline - 2.0:
                        break
                    if can_add(idx, current_vals):
                        included.add(idx)
                        current_vals.add(arr[idx])
                        improved = True
                if not improved:
                    break
        
        result = [arr[i] for i in included]
        if not sum_free_check(result):
            result = [x for x in arr if x % 2]
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() >= deadline - 5.0:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        sf = build_sum_free(vals)
        if not sum_free_check(sf):
            sf = [x for x in vals if x % 2]
        results.append(sf)

    return results
