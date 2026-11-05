import time
from fractions import Fraction

def construct():
        start_time = time.time()
        deadline = 58.0  # 2s safety margin for 60s limit
        
        denominators = list(range(2, 49))
        n = len(denominators)
        
        # Create list of (value, original_index)
        items = [(1.0 / d, i) for i, d in enumerate(denominators)]
        # Sort by value descending
        items.sort(key=lambda x: x[0], reverse=True)
        sorted_vals = [item[0] for item in items]
        orig_indices = [item[1] for item in items]
        
        # Suffix sums for pruning
        suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sorted_vals[i]
            
        best_sum_float = -1.0
        best_sum_frac = Fraction(0)
        best_indices = []
        
        # To avoid recursion limit issues (depth 47 is fine), but iterative might be safer?
        # Recursion depth 47 is well within Python's default limit (1000).
        
        def calculate_exact_sum(indices):
            s = Fraction(0)
            for idx in indices:
                s += Fraction(1, denominators[idx])
            return s

        def dfs(idx, cur_sum_float, selected):
            nonlocal best_sum_float, best_sum_frac, best_indices
            
            if time.time() > deadline:
                return
                
            # Pruning 1: Current sum is too high
            if cur_sum_float >= 1.0:
                return
                
            # Pruning 2: Even taking all remaining, we can't beat best
            if cur_sum_float + suffix[idx] <= best_sum_float:
                return
                
            # Update best if current is better
            # We check if current sum is potentially better
            if cur_sum_float > best_sum_float:
                # Exact verification
                exact_sum = calculate_exact_sum(selected)
                if exact_sum < 1:
                    best_sum_float = float(exact_sum)
                    best_sum_frac = exact_sum
                    best_indices = list(selected)
                    # Update float best to the exact value's float representation?
                    # Or keep the path's float?
                    # Using exact float is safer for pruning consistency?
                    # Actually, if we use exact float, subsequent prunings are more accurate.
            
            if idx == n:
                return
            
            # Option 1: Include
            next_sum_float = cur_sum_float + sorted_vals[idx]
            if next_sum_float < 1.0:
                selected.append(orig_indices[idx])
                dfs(idx + 1, next_sum_float, selected)
                selected.pop()
            
            # Option 2: Exclude
            dfs(idx + 1, cur_sum_float, selected)
            
        dfs(0, 0.0, [])
        
        if not best_indices:
            # Fallback: return empty or a trivial valid set?
            # Task says "Return nonempty".
            # 1/2 is always valid.
            return [0]
            
        return best_indices
