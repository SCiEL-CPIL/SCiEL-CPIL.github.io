"""Program for large reciprocal-subsum-free sets using Integer Linear Programming."""
import time
import math
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

_START_TIME = time.time()
_DEADLINE = 88.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _find_bad_subsets(N=24):
    """
    Find all pairs (i, T) such that 1/i = sum(1/j for j in T)
    where T is a subset of {1, ..., N} \ {i} with distinct elements.
    Returns a list of (i, T) tuples.
    """
    bad_configs = []
    # We iterate over i from 1 to N
    for i in range(1, N + 1):
        # Target sum is 1/i
        target_num = 1
        target_den = i
        
        # Possible elements for T are {1, ..., N} \ {i}
        # To ensure distinct elements and sum to 1/i, we can use recursion or combinations.
        # Since N=24 is small, we can check subsets of increasing size.
        # Max size of T: 1/j >= 1/N, so |T| * 1/N <= 1/i => |T| <= N/i.
        # Also 1/j > 1/N, so |T| < N/i roughly.
        
        # We will generate subsets of {1, ..., N} \ {i}
        candidates = [j for j in range(1, N + 1) if j != i]
        
        # We can prune: if sum of largest k reciprocals < 1/i, stop.
        # Actually, just iterate by size.
        max_size = min(N - 1, int(N / i) + 1) # Safe upper bound
        
        for k in range(1, max_size + 1):
            # If time is running out, break
            if _time_remaining() < 2.0:
                return bad_configs
            
            for T in combinations(candidates, k):
                # Check if sum of reciprocals equals 1/i
                # Use exact fraction arithmetic to avoid floating point errors
                # Sum = sum(1/j for j in T)
                # We want sum == 1/i
                # i * sum(1/j for j in T) == 1
                # sum(i/j for j in T) == 1
                # This is integer check if we use LCM?
                # Let's use fractions.Fraction for precision
                s = 0
                for j in T:
                    s += 1 / j
                # Floating point check with tolerance might be risky for exact equality
                # Better: use Fraction
                from fractions import Fraction
                s_frac = Fraction(0)
                for j in T:
                    s_frac += Fraction(1, j)
                
                if s_frac == Fraction(1, i):
                    bad_configs.append((i, T))
                    
    return bad_configs


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    Uses Integer Linear Programming to maximize cardinality.
    """
    N = 24
    # Find all bad configurations
    bad_configs = _find_bad_subsets(N)
    
    # If no bad configs, we can take all elements
    if not bad_configs:
        return list(range(1, N + 1))
        
    # Define variables x_1, ..., x_N
    # Objective: maximize sum(x_i)
    # c = -1 for all i (since milp minimizes)
    c = [-1.0] * N
    
    # Constraints:
    # 1. x_i in {0, 1}
    integrality = [1] * N
    
    # 2. For each bad config (i, T):
    #    x_i + sum_{j in T} x_j <= |T|
    #    In matrix form A_ub @ x <= b_ub
    #    A_ub row for this constraint:
    #    index i-1 has 1
    #    indices j-1 for j in T have 1
    #    b_ub is |T|
    
    A_ub = []
    b_ub = []
    
    for i, T in bad_configs:
        row = [0.0] * N
        row[i - 1] = 1.0
        for j in T:
            row[j - 1] = 1.0
        A_ub.append(row)
        b_ub.append(len(T))
        
    # If no constraints, A_ub is empty
    if A_ub:
        A_ub_matrix = A_ub
        b_ub_vector = b_ub
    else:
        A_ub_matrix = [[0.0] * N]
        b_ub_vector = [0.0] # Dummy constraint, will be relaxed? No, just handle empty case.
        
    # Actually, if A_ub is empty, we can just return all 1s.
    if not bad_configs:
        return list(range(1, N + 1))

    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=[0.0] * N, ub=[1.0] * N)
    
    # Linear constraints
    # We only have upper bounds in A_ub x <= b_ub
    # milp expects LinearConstraint(A, lb, ub)
    # Here A is A_ub, lb is -inf, ub is b_ub
    lin_cons = LinearConstraint(A_ub_matrix, lb=[-float('inf')] * len(A_ub_matrix), ub=b_ub_vector)
    
    # Solve
    res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to greedy if ILP fails
        return _greedy_fallback(N)
        
    # Extract solution
    # res.x is the optimal solution
    # It might be slightly off due to floating point, so round
    x = res.x
    selected = []
    for i in range(N):
        if x[i] > 0.5:
            selected.append(i + 1)
            
    if not selected:
        return [1] # Ensure non-empty
        
    return sorted(selected)


def _greedy_fallback(N=24):
    """
    Greedy fallback if ILP fails or times out.
    """
    selected = []
    # Sort candidates in descending order to prioritize larger numbers?
    # Or ascending? Larger numbers have smaller reciprocals, less likely to form sums.
    # Actually, small numbers form sums more easily.
    # Let's try descending.
    for candidate in range(N, 0, -1):
        if _time_remaining() < 1.0:
            break
        if _is_subsum_free(selected, candidate):
            selected.append(candidate)
            
    if not selected:
        return [1]
        
    return sorted(selected)


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    """
    from fractions import Fraction
    
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any subset of selected
    if selected:
        target = Fraction(1, candidate)
        # Meet-in-the-middle or simple subset sum for small N
        # N is at most 24, so 2^24 is too big for naive, but selected is usually small in greedy
        # Let's use a set of sums
        sums = {Fraction(0)}
        for x in selected:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, x))
            sums.update(new_sums)
        
        if target in sums:
            return False
            
    # Check for each element in selected: 1/x should not equal sum of reciprocals of any subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        target = Fraction(1, x)
        sums = {Fraction(0)}
        for y in others:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, y))
            sums.update(new_sums)
            
        if target in sums:
            return False
            
    return True


if __name__ == "__main__":
    # Simple test
    print(construct())
