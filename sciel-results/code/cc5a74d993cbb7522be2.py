import time
from random import Random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return []
        
        # Precompute value to index map for quick lookup
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # Heuristic for sorting: Larger absolute values are "safer" to add
        # because their sums/differences are more likely to be out of range or not in the set.
        # Also, odd numbers are mutually sum-free.
        
        def can_add(val, current_set_vals):
            """Check if val can be added to current_set_vals (a set of values)."""
            for x in current_set_vals:
                if val + x in current_set_vals:
                    return False
                if val - x in current_set_vals:
                    return False
            return True

        def greedy_fill(order, start_set=None, start_vals=None):
            """Greedy fill given an order of indices."""
            if start_set is None:
                current_set = set()
                current_vals = set()
            else:
                current_set = set(start_set)
                current_vals = set(start_vals)
            
            for idx in order:
                if idx in current_set:
                    continue
                val = vals[idx]
                if can_add(val, current_vals):
                    current_set.add(idx)
                    current_vals.add(val)
            return current_set, current_vals

        # 1. Multiple Greedy Initializations
        best_set = set()
        best_vals = set()
        best_size = 0
        
        rng_init = Random(12345)
        
        # Strategy 1: Descending absolute value
        order1 = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
        s, v = greedy_fill(order1)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 2: Odds first (descending abs), then Evens (descending abs)
        # Odds are sum-free among themselves.
        odd_idx = [i for i in range(n) if vals[i] % 2 != 0]
        even_idx = [i for i in range(n) if vals[i] % 2 == 0]
        order2 = sorted(odd_idx, key=lambda i: abs(vals[i]), reverse=True) + \
                 sorted(even_idx, key=lambda i: abs(vals[i]), reverse=True)
        s, v = greedy_fill(order2)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 3: Random shuffles
        for _ in range(10):
            order3 = list(range(n))
            rng_init.shuffle(order3)
            s, v = greedy_fill(order3)
            if len(s) > best_size:
                best_size = len(s)
                best_set, best_vals = s, v

        # 2. Simulated Annealing / Local Search
        current_set = set(best_set)
        current_vals = set(best_vals)
        current_size = best_size
        
        T = 1.0
        alpha = 0.995  # Cooling rate
        min_T = 1e-6
        
        rng_sa = Random(54321)
        
        # Order for refilling: Descending absolute value is generally good
        refill_order = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
        
        # Also prepare a random refill order for diversity?
        # Let's stick to one good deterministic order for speed, 
        # but maybe alternate with a random one occasionally?
        # For now, use the abs-desc order.
        
        while time.time() < deadline - 5.0:
            if T < min_T:
                break
            
            # Move: Remove k random elements, then refill
            # k=1 is fast. k=2 is slower but can escape deeper local minima.
            # Let's try k=1 mostly, and k=2 sometimes.
            k = 1 if rng_sa.random() < 0.7 else 2
            
            if len(current_set) < k:
                break
            
            # Pick k random elements to remove
            remove_indices = rng_sa.sample(list(current_set), k)
            remove_vals = {vals[i] for i in remove_indices}
            
            temp_set = current_set - set(remove_indices)
            temp_vals = current_vals - remove_vals
            
            # Greedily fill
            new_set, new_vals = greedy_fill(refill_order, start_set=temp_set, start_vals=temp_vals)
            
            new_size = len(new_set)
            delta = new_size - current_size
            
            if delta > 0:
                current_set = new_set
                current_vals = new_vals
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_vals = set(current_vals)
            else:
                # Accept worse move with probability exp(delta / T)
                # delta is <= 0
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(delta / T)
                if rng_sa.random() < prob:
                    current_set = new_set
                    current_vals = new_vals
                    current_size = new_size
            
            T *= alpha
        
        # Final polish: Try to add any missing element
        final_set, final_vals = greedy_fill(refill_order, start_set=best_set, start_vals=best_vals)
        
        return [vals[i] for i in sorted(final_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if time runs out
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(solve_instance(vals))

    return results
