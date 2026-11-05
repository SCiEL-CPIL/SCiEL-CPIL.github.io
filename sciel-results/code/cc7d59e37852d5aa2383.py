"""Improved program: maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time
from itertools import combinations

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Strategy: Start with the known good set {66..100} (size 35) and try to add
    # elements from {1..65} while maintaining the no-subset-sum-200 property.
    # Use a greedy approach: try adding elements in order of "least conflict".
    
    base = set(range(66, 101))
    candidates = list(range(65, 0, -1))  # try larger candidates first
    
    # For each candidate, check if adding it to the current set creates a subset summing to 200
    # A subset of S ∪ {x} sums to 200 iff either:
    #   - A subset of S sums to 200 (already ensured false), or
    #   - A subset of S sums to 200 - x (using x)
    
    S = set(base)
    
    # Precompute: for the current S, which target sums are achievable?
    # We'll maintain a bitset of achievable sums up to 200
    
    def achievable_sums(element_set):
        """Return a set of all achievable subset sums up to 200 from element_set."""
        reachable = {0}
        for x in element_set:
            if x > 200:
                continue
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        return reachable
    
    # Greedily add candidates
    for x in candidates:
        if time.time() - _START > _DEADLINE - 5.0:
            break
        
        # Check if adding x creates a subset summing to 200
        # This happens if 200 - x is achievable by S
        S_test = S | {x}
        ach = achievable_sums(S_test)
        if 200 not in ach:
            S = S_test
    
    # Now try to optimize: maybe we can swap some elements
    # Try removing a base element and adding multiple candidates
    current_size = len(S)
    
    # Try individual swaps: remove one base element, add one or more candidates
    for b in list(range(66, 101)):
        if time.time() - _START > _DEADLINE - 5.0:
            break
        
        S_without_b = S - {b}
        
        # Try adding candidates to S_without_b
        S_new = set(S_without_b)
        for x in candidates:
            if x in S_new:
                continue
            S_test = S_new | {x}
            ach = achievable_sums(S_test)
            if 200 not in ach:
                S_new = S_test
        
        if len(S_new) > current_size:
            S = S_new
            current_size = len(S)
    
    # Multiple swaps: try removing 2 base elements and adding candidates
    for b1 in range(66, 101):
        if time.time() - _START > _DEADLINE - 5.0:
            break
        for b2 in range(b1 + 1, 101):
            if time.time() - _START > _DEADLINE - 5.0:
                break
            
            S_without = S - {b1, b2}
            S_new = set(S_without)
            for x in candidates:
                if x in S_new:
                    continue
                S_test = S_new | {x}
                ach = achievable_sums(S_test)
                if 200 not in ach:
                    S_new = S_test
            
            if len(S_new) > current_size:
                S = S_new
                current_size = len(S)
    
    return list(S)

def _greedy_fallback():
    """Fallback greedy approach."""
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
