import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import sys

def construct():
    n = 24
    elements = list(range(1, 25))
    
    # We need to find all pairs (a, J) such that:
    # 1. a in [1, 24]
    # 2. J is a subset of [1, 24] \ {a}
    # 3. sum(1/j for j in J) == 1/a
    
    # We will collect unique constraints.
    # Constraint: x_a + sum(x_j for j in J) <= |J|
    
    constraints_data = []
    seen = set()
    
    # Helper to find subsets J summing to target 1/a
    # using backtracking on sorted elements
    def find_subsets(target_num, target_den, available_elements):
        # target is target_num / target_den
        # We use Fraction for exact comparison
        target = Fraction(target_num, target_den)
        # Sort available elements descending to prune faster? 
        # Or ascending? 
        # If we pick large numbers, sum grows slow. 
        # If we pick small numbers, sum grows fast.
        # Pruning: if current_sum + sum(1/e for e in remaining) < target, prune.
        # if current_sum > target, prune.
        
        results = []
        
        # Precompute suffix sums of reciprocals for available_elements?
        # available_elements is a tuple/list.
        
        def backtrack(idx, current_sum, current_subset):
            if current_sum == target:
                results.append(list(current_subset))
                return
            
            # Pruning: 
            # 1. If current_sum > target, return
            if current_sum > target:
                return
                
            # 2. If we've exhausted elements, return
            if idx == len(available_elements):
                return
            
            # Calculate max possible sum from remaining elements
            # sum(1/e for e in available_elements[idx:])
            # We can precompute this for the specific available_elements list
            # But available_elements changes based on what's excluded (a)
            
            # For efficiency, let's just iterate.
            # Optimization: If current_sum + sum(1/e for e in available_elements[idx:]) < target, return
            
            remaining_sum = Fraction(0,1)
            for k in range(idx, len(available_elements)):
                remaining_sum += Fraction(1, available_elements[k])
                
            if current_sum + remaining_sum < target:
                return
            
            # Option 1: Skip available_elements[idx]
            backtrack(idx + 1, current_sum, current_subset)
            
            # Option 2: Take available_elements[idx]
            new_sum = current_sum + Fraction(1, available_elements[idx])
            if new_sum <= target:
                current_subset.append(available_elements[idx])
                backtrack(idx + 1, new_sum, current_subset)
                current_subset.pop()
                
        backtrack(0, Fraction(0,1), [])
        return results

    # Iterate over each possible 'a'
    for a in range(1, 25):
        # Elements available for J are [1..24] excluding a
        avail = [e for e in elements if e != a]
        # Sort descending? 
        # If we sort descending, we pick large numbers first. 
        # 1/large is small. 
        # This might lead to many small additions before exceeding target?
        # If we sort ascending, we pick small numbers first. 1/small is large.
        # Exceeding target happens faster. 
        # Let's sort ascending.
        avail.sort()
        
        subsets = find_subsets(1, a, avail)
        
        for J in subsets:
            # J is a list of elements
            # Constraint: x_a + sum(x_j for j in J) <= len(J)
            # Note: The problem says "subset of the OTHER selected elements".
            # So if a is selected, and J is a subset of selected elements not containing a, 
            # and sum(1/j for j in J) == 1/a, this is a conflict.
            # The constraint prevents this: we cannot select a AND all elements in J.
            # x_a + sum(x_j) <= |J|
            # If x_a=1 and all x_j=1, LHS = 1 + |J| > |J|. Violation.
            # If x_a=0, LHS = sum(x_j) <= |J|. OK.
            # If some x_j=0, LHS <= 1 + |J|-1 = |J|. OK.
            
            coeffs = [0] * 24
            coeffs[a-1] = 1
            for j in J:
                coeffs[j-1] += 1
                
            key = tuple(coeffs)
            if key not in seen:
                seen.add(key)
                constraints_data.append((coeffs, len(J)))
                
    if not constraints_data:
        return list(range(1, 25))
        
    # Build ILP
    c = np.ones(n) * -1
    
    A_ub = np.array([c for c, _ in constraints_data], dtype=float)
    b_ub = np.array([s for _, s in constraints_data], dtype=float)
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        return list(range(13, 25))
