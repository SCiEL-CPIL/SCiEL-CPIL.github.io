import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    target_num = 5
    target_den = 31
    
    best_max = 10_000_000
    best_solution = None
    
    # We will search for solutions with 2 to 6 terms.
    # The problem allows up to 256, but small term counts usually yield smaller max denominators for such fractions.
    # If no solution is found with <= 6 terms, we can fall back to greedy or more terms, 
    # but given the target 5/31 ~ 0.16, 3-5 terms is typical.
    
    def dfs(rem_num, rem_den, min_d, terms_left, current_max, path):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        # If we have reached 0, we have a valid solution
        if rem_num == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(path)
            return

        if terms_left == 0:
            return

        # Pruning: 
        # 1. We need 1/d <= rem_num/rem_den => d >= rem_den/rem_num
        d_min = max(min_d, (rem_den + rem_num - 1) // rem_num)
        
        # 2. Upper bound on d.
        # The remainder R' = rem_num/rem_den - 1/d must be representable by terms_left-1 terms 
        # with denominators > d.
        # Max sum of terms_left-1 terms with denominators > d is (terms_left-1)/(d+1).
        # So R' <= (terms_left-1)/(d+1)
        # rem_num/rem_den - 1/d <= (terms_left-1)/(d+1)
        # Let k = terms_left - 1
        # rem_num/rem_den <= 1/d + k/(d+1) = (d+1+kd)/(d(d+1)) = ((k+1)d + 1)/(d(d+1))
        # rem_num * d * (d+1) <= rem_den * ((k+1)d + 1)
        # rem_num * (d^2 + d) <= rem_den * (k+1) * d + rem_den
        # rem_num * d^2 + rem_num * d - rem_den * (k+1) * d - rem_den <= 0
        # This is a quadratic inequality in d. We can solve for the upper bound.
        # Alternatively, just iterate and break if condition fails.
        # Since d_min is usually small, we can iterate d from d_min and check the condition.
        # If the condition fails for some d, it will fail for all larger d?
        # Let f(d) = rem_num/rem_den - 1/d - (terms_left-1)/(d+1)
        # As d increases, 1/d decreases, (terms_left-1)/(d+1) decreases.
        # So f(d) is increasing.
        # We want f(d) <= 0.
        # So there is a max d.
        
        # Also, we can prune if current_max is already >= best_max and d >= current_max.
        # Actually, if d > best_max, we can't improve.
        # If d <= best_max, we might find a solution with max <= best_max.
        
        d = d_min
        # We also know d cannot exceed best_max if we want to improve, 
        # but we might find a solution with max == best_max? No, we want strictly smaller or equal?
        # "Minimize the LARGEST denominator". If we find one with max=288, and another with max=288, it doesn't matter.
        # If we find one with max=280, it's better.
        # So we can prune d > best_max.
        d_max_limit = best_max
        
        # Calculate a tighter upper bound from the quadratic or just loop until condition breaks.
        # Loop is safer.
        
        while d <= d_max_limit:
            if time.time() > deadline:
                return
            
            # Check if remainder is positive
            # 1/d < rem_num/rem_den  => d > rem_den/rem_num
            # Our d_min ensures 1/d <= rem. If 1/d == rem, then rem becomes 0.
            # If 1/d > rem, invalid. But d_min = ceil(rem_den/rem_num) ensures 1/d <= rem.
            # So rem' >= 0.
            
            # Check if remainder is too large to be formed by remaining terms
            # rem' = rem_num/rem_den - 1/d
            # We need rem' <= (terms_left-1)/(d+1)
            # (rem_num * d - rem_den) / (rem_den * d) <= (terms_left - 1) / (d + 1)
            # (rem_num * d - rem_den) * (d + 1) <= rem_den * d * (terms_left - 1)
            
            lhs = (rem_num * d - rem_den) * (d + 1)
            rhs = rem_den * d * (terms_left - 1)
            
            if lhs > rhs:
                # For larger d, lhs grows as d^2, rhs as d. 
                # Wait, if d is small, lhs might be negative?
                # rem_num * d - rem_den >= 0 because d >= rem_den/rem_num.
                # So lhs is positive.
                # As d increases, lhs increases faster than rhs.
                # So if lhs > rhs, it will remain true for larger d.
                break
            
            # Calculate new remainder
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num == 0:
                # We have a solution
                new_max = max(current_max, d)
                if new_max < best_max:
                    best_max = new_max
                    best_solution = list(path) + [d]
                # We can continue searching for even better solutions?
                # If new_max is small, maybe we can find smaller?
                # But this branch is done.
                continue
            else:
                # Reduce fraction
                g = gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                # Recurse
                path.append(d)
                dfs(new_num, new_den, d + 1, terms_left - 1, max(current_max, d), path)
                path.pop()
            
            d += 1

    # Search for solutions with 2 to 6 terms
    # Start with 2 terms, then 3, etc.
    # Often, more terms allow smaller max denominators?
    # Example: 1/2 + 1/3 + 1/6 = 1. Max 6.
    # 1/1 ... no.
    # For 5/31:
    # 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31^2 = 961.
    # Factors of 961: 1, 31, 961.
    # 5a-31=1 => 5a=32 (no)
    # 5a-31=31 => 5a=62 (no)
    # 5a-31=961 => 5a=992 (no)
    # So no 2-term solution.
    
    # 3 terms:
    # We'll let the DFS find it.
    
    for k in range(2, 7):
        if time.time() > deadline:
            break
        # Initial call
        # rem = 5/31
        # min_d = 1 (but 1/1 = 1 > 5/31, so d_min will be adjusted)
        # current_max = 0
        # path = []
        dfs(5, 31, 1, k, 0, [])
        
        # If we found a solution with max < 100, we might stop?
        # The current best is 288. If we find < 288, we keep searching for smaller.
        # If we find a very small one, e.g., < 50, we can stop.
        if best_max < 50:
            break

    if best_solution is None:
        # Fallback: Greedy algorithm
        rem_num, rem_den = 5, 31
        values = []
        used = set()
        while rem_num > 0:
            d = (rem_den + rem_num - 1) // rem_num
            while d in used:
                d += 1
            values.append(d)
            used.add(d)
            rem_num, rem_den = rem_num * d - rem_den, rem_den * d
            g = gcd(rem_num, rem_den)
            if g > 1:
                rem_num //= g
                rem_den //= g
        best_solution = values
        best_max = max(values)

    return best_solution
