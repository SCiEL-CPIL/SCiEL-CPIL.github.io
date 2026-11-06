from fractions import Fraction
import time

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 50  # 50s safety margin
    
    # Try k=7 first (4-6 are impossible based on max sum with min denom 5)
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11 ≈ 0.2+0.167+0.143+0.125+0.111+0.1+0.091 = 0.937 < 1
    # So 7 terms starting at 5 can't reach 1. Start from k=8.
    for k in range(8, 257):
        if time.time() > _DEADLINE:
            break
        result = search_k(Fraction(1, 1), 5, k)
        if result:
            return result
            
    # Fallback if no solution found (should not happen)
    return greedy_solution()

def search_k(target, min_denom, terms_left):
    """
    Finds 'terms_left' distinct denominators >= min_denom 
    that sum to 'target'.
    Returns a list of denominators or None.
    """
    global _DEADLINE
    
    if time.time() > _DEADLINE:
        return None

    if terms_left == 0:
        if target == 0:
            return []
        else:
            return None
            
    if terms_left == 1:
        # 1/d = target => d = 1/target
        if target.numerator != 1:
            return None
        d = target.denominator
        if d >= min_denom:
            return [d]
        return None

    # Pruning: Max possible sum with smallest available denominators
    max_sum = Fraction(0, 1)
    for i in range(terms_left):
        max_sum += Fraction(1, min_denom + i)
        
    if max_sum < target:
        return None

    # Pruning: Min possible sum - if even the largest denominators can't be small enough
    # The minimum sum occurs when we use the largest possible denominators
    # But we need at least terms_left distinct values >= min_denom
    # The minimum sum is achieved by using very large denominators, which approaches 0
    # So this pruning isn't very useful directly, but we can bound d_max

    # Iterate possible next denominators
    if target.numerator == 0:
        return None
        
    # Lower bound: d must satisfy 1/d <= target, so d >= 1/target
    d_min = min_denom
    if target.denominator > 0:
        inv_target = Fraction(target.denominator, target.numerator)
        if inv_target.numerator % inv_target.denominator == 0:
            d_from_target = inv_target.numerator // inv_target.denominator
        else:
            d_from_target = inv_target.numerator // inv_target.denominator + 1
        d_min = max(d_min, d_from_target)
        
    # Upper bound: after choosing d, we need terms_left-1 terms with denominators > d
    # that sum to target - 1/d. The maximum sum of those terms is:
    # 1/(d+1) + 1/(d+2) + ... + 1/(d+terms_left-1)
    # We need target - 1/d <= max_rest, which gives us a constraint on d
    # Also, we need target - 1/d > 0, so 1/d < target, already handled by d_min
    
    # For the upper bound, we need:
    # target - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
    # As d increases, 1/d decreases, so target - 1/d increases
    # And the RHS decreases as d increases
    # So there's a maximum d where this still holds
    
    # Simple upper bound: d can't be too large or the remaining sum will be too small
    # The remaining target is target - 1/d, and the max sum of remaining terms is approximately
    # (terms_left-1)/d for large d
    # So we need target - 1/d <= (terms_left-1)/d roughly
    # target <= terms_left/d
    # d <= terms_left / target
    # More precisely, let's compute a tighter bound
    
    d_max = 10000000  # Default upper bound
    
    # Tighter upper bound: 
    # We need target - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
    # For a given d, check if this is satisfiable
    # Since both sides decrease/increase monotonically, we can binary search or use a formula
    
    # A simpler approach: the minimum value of (target - 1/d) / sum_{i=1}^{terms_left-1} 1/(d+i) 
    # must be achievable. But let's just iterate and prune.
    
    # Better upper bound: d_max such that 1/d < target (already have) and
    # target - 1/d >= minimum possible sum of remaining terms
    # Minimum sum of remaining terms with denominators > d is approximately 0 (as denominators go to infinity)
    # So this doesn't give a useful upper bound directly
    
    # Instead, let's use: the remaining sum target - 1/d must be representable by terms_left-1 
    # distinct denominators > d. The maximum such sum is sum_{i=1}^{terms_left-1} 1/(d+i).
    # So we need target - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
    # This gives us an upper bound on d. Let's find it.
    
    # For efficiency, let's compute d_max by checking when the inequality fails
    # Since the LHS increases with d and RHS decreases with d, there's a threshold
    
    # Approximate upper bound: d <= (terms_left - 1) / (target - 1/d) is circular
    # Let's use: d_max = min(10000000, int(terms_left / target) + 1) as a rough bound
    # But this might not be tight enough. Let's compute it properly.
    
    # Compute exact d_max by finding the largest d such that:
    # target - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
    
    # This is equivalent to: target <= 1/d + sum_{i=1}^{terms_left-1} 1/(d+i) = sum_{i=0}^{terms_left-1} 1/(d+i)
    # Which is exactly the max_sum condition but with d as the minimum!
    # So we need: sum_{i=0}^{terms_left-1} 1/(d+i) >= target
    # This means d can't be too large.
    
    # Let's find d_max by binary search or iteration
    # For efficiency, let's compute a reasonable upper bound
    
    # The sum sum_{i=0}^{terms_left-1} 1/(d+i) >= target
    # For large d, this is approximately terms_left/d
    # So d <= terms_left/target approximately
    d_max_approx = int(terms_left / target) + 2
    d_max = min(10000000, max(d_min, d_max_approx))
    
    # Now iterate d from d_min to d_max
    for d in range(d_min, d_max + 1):
        if time.time() > _DEADLINE:
            return None
            
        term = Fraction(1, d)
        if term > target:
            break
            
        new_target = target - term
        
        if new_target <= 0:
            continue
            
        # Check if new_target can be formed by terms_left-1 terms > d
        # Max possible sum of remaining terms
        max_rest = Fraction(0, 1)
        for i in range(terms_left - 1):
            max_rest += Fraction(1, d + 1 + i)
            
        if new_target > max_rest:
            break
            
        # Min possible sum of remaining terms (with very large denominators) is > 0
        # So we just need new_target > 0, which we checked
        
        res = search_k(new_target, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res
            
    return None

def greedy_solution():
    """
    Returns a valid list of denominators using the greedy algorithm.
    This is a fallback to ensure a non-zero score.
    """
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        # Find smallest d >= D such that 1/d <= R
        # 1/d <= R => d >= 1/R
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        # Ensure d >= D
        if d < D:
            d = D
            
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        
        # Safety break
        if len(denoms) > 256:
            break
            
    return denoms
