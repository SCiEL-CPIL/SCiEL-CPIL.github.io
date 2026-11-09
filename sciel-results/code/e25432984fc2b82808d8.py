from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # We know N=3,4,5,6,7 are impossible. Start from 8.
    for N in range(8, 257):
        if time.time() > deadline:
            break
            
        found = None
        # DFS to find N distinct denominators
        # State: current_sum (Fraction), next_min_d (int), terms_left (int), current_list (list)
        
        def dfs(current_sum, next_min_d, terms_left, current_list):
            if time.time() > deadline:
                return None
                
            if terms_left == 0:
                if current_sum == Fraction(1, 1):
                    return current_list
                return None
                
            if current_sum > Fraction(1, 1):
                return None
                
            # Remaining sum needed
            remaining = Fraction(1, 1) - current_sum
            
            # If terms_left is 1
            if terms_left == 1:
                # We need 1/d = remaining => d = 1/remaining
                # d must be integer and >= next_min_d
                if remaining.numerator == 0:
                    return None
                d = remaining.denominator // remaining.numerator
                if remaining.denominator % remaining.numerator != 0:
                    return None
                if d >= next_min_d and d <= 10000000:
                    return current_list + [d]
                return None

            # General case
            # Lower bound for d:
            # 1/d <= remaining => d >= 1/remaining
            lb = remaining.denominator // remaining.numerator
            if remaining.denominator % remaining.numerator != 0:
                lb += 1
            lb = max(lb, next_min_d)
            
            # Upper bound for d:
            # Sum of reciprocals of d, d+1, ..., d+terms_left-1 must be >= remaining
            # Let's iterate d from lb upwards.
            # We can estimate an upper bound to stop early.
            # Sum approx terms_left / d.
            # terms_left / d >= remaining => d <= terms_left / remaining
            ub = (terms_left * remaining.denominator) // remaining.numerator + 10
            # Also d must be such that the max possible sum is sufficient.
            
            # Iterate d
            # To avoid infinite loops or too many iterations, we rely on the pruning condition.
            # But we can also stop if d becomes very large.
            # However, the sum of reciprocals drops off fast.
            
            d = lb
            # We need to check if it's possible to reach 'remaining' with 'terms_left' terms starting at d.
            # The maximum sum we can get with 'terms_left' terms starting at d is sum(1/(d+i) for i in range(terms_left))
            # If this max sum < remaining, we can stop increasing d.
            
            while d <= 10000000:
                # Check if max possible sum from d is sufficient
                max_sum = Fraction(0, 1)
                for i in range(terms_left):
                    max_sum += Fraction(1, d + i)
                
                if max_sum < remaining:
                    break
                
                # If 1/d > remaining, we can't pick this d (and larger d will have even smaller 1/d)
                if Fraction(1, d) > remaining:
                    # Actually, if 1/d > remaining, we can't pick d.
                    # But we are iterating d upwards. 1/d decreases.
                    # So if 1/d > remaining, we continue? No.
                    # If 1/d > remaining, then picking d makes sum > 1.
                    # So we must skip this d.
                    # But as d increases, 1/d decreases.
                    # So we should start from a d where 1/d <= remaining.
                    # Which is exactly what lb does.
                    # So this case shouldn't happen if lb is correct.
                    pass
                else:
                    # Try picking d
                    new_sum = current_sum + Fraction(1, d)
                    result = dfs(new_sum, d + 1, terms_left - 1, current_list + [d])
                    if result is not None:
                        return result
                
                d += 1
                
            return None

        found = dfs(Fraction(0, 1), 5, N, [])
        if found is not None:
            return found
            
    raise RuntimeError("no solution found within time limit")
