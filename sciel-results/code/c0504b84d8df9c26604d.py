import sys
from math import gcd
from functools import reduce
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    """
    Returns a nonempty subset of [1, 24] that maximizes cardinality
    such that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the other selected elements.
    """
    
    def is_valid(S):
        """
        Checks if the set S is reciprocal-subsum-free using integer arithmetic.
        Scales by LCM to convert to integer subset sum problem.
        """
        if not S:
            return True
            
        # Calculate LCM of all elements in S
        # LCM of empty set is 1, but S is non-empty here
        L = reduce(lcm, S)
        
        # Map each element x in S to its integer weight w_x = L / x
        # We need to check for each x in S:
        # Is w_x a subset sum of { w_y for y in S, y != x }?
        
        # Precompute weights
        weights = {x: L // x for x in S}
        S_list = list(S)
        
        for x in S_list:
            target = weights[x]
            others = [y for y in S_list if y != x]
            
            # Subset sum of weights of 'others'
            # We want to know if target is reachable.
            # Since all weights are positive, we can prune sums > target.
            
            # Optimization: If the sum of all weights of others < target, 
            # it's impossible to reach target.
            sum_others = sum(weights[y] for y in others)
            if sum_others < target:
                continue
                
            # Use a set to track reachable sums
            reachable = {0}
            
            for y in others:
                w_y = weights[y]
                new_sums = set()
                for s in reachable:
                    ns = s + w_y
                    if ns <= target:
                        new_sums.add(ns)
                
                if new_sums:
                    reachable.update(new_sums)
                
                # Early exit if target is found
                if target in reachable:
                    return False
            
        return True

    # Search for the largest K such that the set of the K largest numbers is valid.
    # The set of largest numbers has the smallest reciprocals (largest denominators),
    # which minimizes the risk of reciprocal subsums.
    for K in range(24, 0, -1):
        candidate = list(range(25 - K, 25))
        if is_valid(candidate):
            return candidate
    
    # Fallback
    return [1]

if __name__ == "__main__":
    result = construct()
    print(result)
