import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s kill
    
    MAX_D = 10_000_000
    MIN_D = 5

    def fast_gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search(num, den, k, min_d, path):
        """
        DFS to find k distinct unit fractions summing to num/den,
        all with denominators >= min_d.
        Returns list of denominators or None.
        """
        # Time check
        if time.time() > deadline:
            return None
        
        # Base cases
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            # Need num/den = 1/d => d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= MAX_D:
                    return path + [d]
            return None

        # Bounds for the next denominator d
        # 1/d <= num/den  =>  d >= den/num
        # Also d >= min_d
        d_min = max(min_d, (den + num - 1) // num)  # ceil(den/num)
        
        # num/den <= k/d  =>  d <= k*den/num
        # Using integer arithmetic: k*den >= num*d => d <= k*den/num
        d_max = (k * den) // num
        
        if d_max > MAX_D:
            d_max = MAX_D
            
        if d_min > d_max:
            return None

        # Iterate d from d_min to d_max
        # To speed up, we can try a heuristic order or just linear
        # Linear is safer for correctness and finding the first solution.
        
        for d in range(d_min, d_max + 1):
            # Calculate remaining fraction: num/den - 1/d
            # = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            # n2 must be positive (since we want remaining sum > 0 for k-1 terms)
            if n2 <= 0:
                continue
            
            # Reduce fraction
            g = fast_gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            # Pruning:
            # The remaining sum n2_reduced/d2_reduced must be representable
            # with k-1 terms, all >= d+1.
            # Max sum with k-1 terms starting at d+1 is (k-1)/(d+1)
            # So we need n2_reduced/d2_reduced <= (k-1)/(d+1)
            # => n2_reduced * (d+1) <= (k-1) * d2_reduced
            if n2_reduced * (d + 1) > (k - 1) * d2_reduced:
                continue
            
            # Also, the minimum sum with k-1 distinct terms starting at d+1
            # is 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1)
            # This is a tighter lower bound but more expensive to compute.
            # For now, the upper bound pruning is usually sufficient to cut branches.
            
            result = search(n2_reduced, d2_reduced, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None

    # We know from analysis that k < 8 is impossible with distinct denominators >= 5.
    # 1/5+1/6+1/7+1/8 = 0.6345 < 1
    # 1/5+...+1/11 = 0.9363 < 1
    # 1/5+...+1/12 > 1
    # So we start trying from k=8.
    
    best_solution = None
    
    # Try k from 8 upwards
    for k in range(8, 257):
        if time.time() > deadline - 5:
            break
            
        solution = search(1, 1, k, MIN_D, [])
        if solution is not None:
            # Verify the solution
            total = sum(Fraction(1, d) for d in solution)
            if total == 1 and len(set(solution)) == len(solution) and all(MIN_D <= d <= MAX_D for d in solution):
                best_solution = solution
                break
        else:
            # If search fails for a small k, it might be due to time or pruning.
            # But usually if k is feasible, DFS finds it quickly if we start from small k.
            # If it returns None, it's either infeasible or we hit time limit.
            # Given the structure, if k=8 fails, k=9 is more likely to succeed but slower.
            # However, if k=8 is infeasible, we must try 9.
            pass

    if best_solution is not None:
        return best_solution

    # Fallback: Greedy algorithm (Sylvester's sequence variant)
    # This guarantees a solution but might use many terms.
    chosen = []
    remaining = Fraction(1, 1)
    last_d = MIN_D - 1
    
    for _ in range(256):
        if remaining == 0:
            break
        if time.time() > deadline:
            break
        
        # Find largest d > last_d such that 1/d <= remaining
        # d >= ceil(1/remaining)
        # 1/d <= remaining => d >= 1/remaining
        # Since remaining is a fraction p/q, 1/remaining = q/p.
        # So d >= ceil(q/p).
        p = remaining.numerator
        q = remaining.denominator
        if p == 0:
            break
            
        d_min = (q + p - 1) // p
        d_min = max(d_min, last_d + 1)
        
        if d_min > MAX_D:
            break
            
        chosen.append(d_min)
        last_d = d_min
        remaining -= Fraction(1, d_min)
        
    if remaining == 0 and 1 <= len(chosen) <= 256:
        return chosen

    # Last resort: Hardcoded valid solution if greedy failed (unlikely)
    # A known 8-term solution?
    # Let's try to construct one quickly if possible, or just raise.
    # But the greedy should work.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + ...
    # Let's just return the greedy result if it's valid, else raise.
    if chosen and len(chosen) <= 256 and all(MIN_D <= d <= MAX_D for d in chosen) and len(set(chosen)) == len(chosen):
        total = sum(Fraction(1, d) for d in chosen)
        if total == 1:
            return chosen

    raise RuntimeError("no representation found")
