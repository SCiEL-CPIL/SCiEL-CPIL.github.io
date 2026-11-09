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
        def greedy_init(order):
            included = set()
            for idx in order:
                c_val = arr[idx]
                ok = True
                for a_idx in included:
                    a_val = arr[a_idx]
                    b_val = c_val - a_val
                    if b_val in elem_to_idx and elem_to_idx[b_val] in included:
                        ok = False
                        break
                if ok:
                    for b_idx in included:
                        b_val = arr[b_idx]
                        s_val = c_val + b_val
                        if s_val in elem_to_idx and elem_to_idx[s_val] in included:
                            ok = False
                            break
                if ok:
                    included.add(idx)
            return included
        
        # Multiple greedy starts
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            list(range(n))
        ]
        rng.shuffle(orders[2])
        
        best_included = set()
        for order in orders:
            inc = greedy_init(order)
            if len(inc) > len(best_included):
                best_included = inc
        
        # Local search
        included = set(best_included)
        
        def can_add(idx, current):
            """Check if idx can be added to current without violating sum-free."""
            c_val = arr[idx]
            for a_idx in current:
                a_val = arr[a_idx]
                b_val = c_val - a_val
                if b_val in elem_to_idx and elem_to_idx[b_val] in current:
                    return False
            for b_idx in current:
                b_val = arr[b_idx]
                s_val = c_val + b_val
                if s_val in elem_to_idx and elem_to_idx[s_val] in current:
                    return False
            return True
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try adding excluded elements
            excluded = [i for i in range(n) if i not in included]
            rng.shuffle(excluded)
            for idx in excluded:
                if time.time() >= deadline - 3.0:
                    break
                if can_add(idx, included):
                    included.add(idx)
                    improved = True
            
            # Try swaps: remove one included, add one excluded
            if time.time() >= deadline - 3.0:
                break
            excluded = [i for i in range(n) if i not in included]
            inc_list = list(included)
            rng.shuffle(excluded)
            rng.shuffle(inc_list)
            
            for add_idx in excluded[:min(len(excluded), 30)]:
                if time.time() >= deadline - 3.0:
                    break
                for rem_idx in inc_list[:min(len(inc_list), 15)]:
                    if time.time() >= deadline - 3.0:
                        break
                    trial = (included - {rem_idx}) | {add_idx}
                    if can_add(add_idx, trial - {add_idx}):
                        included = trial
                        improved = True
                        break
        
        result = [arr[i] for i in included]
        if not sum_free_check(result):
            result = [x for x in arr if x % 2]
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() >= deadline - 3.0:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        sf = build_sum_free(vals)
        if not sum_free_check(sf):
            sf = [x for x in vals if x % 2]
        results.append(sf)

    return results
