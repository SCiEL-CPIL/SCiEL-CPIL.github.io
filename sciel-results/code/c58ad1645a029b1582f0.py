import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    best_set = []
    best_size = 0
    
    def can_add(x, sums):
        # Check if x is a difference of any two elements in sums
        # i.e., exists a, b in sums such that a - b = x
        # Equivalent to: exists s in sums such that s - x in sums
        for s in sums:
            if s - x in sums:
                return False
        return True

    def add_element(x, sums):
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + x)
        return new_sums

    def greedy_heuristic():
        sums = {Fraction(0, 1)}
        selected = []
        available = list(range(1, 23))
        
        while available:
            best_k = None
            best_score = -1
            
            for k in available:
                x = Fraction(1, k)
                if not can_add(x, sums):
                    continue
                
                # Heuristic: Pick the one that is "furthest" from existing sums?
                # Or just pick the largest k?
                # Let's try picking the largest k (smallest x) to keep sums small?
                # Or smallest k (largest x) to spread them?
                # Let's try smallest k first (greedy by value)
                score = -k 
                if score > best_score:
                    best_score = score
                    best_k = k
            
            if best_k is None:
                break
            
            x = Fraction(1, best_k)
            sums = add_element(x, sums)
            selected.append(best_k)
            available.remove(best_k)
            
        return selected

    # Run a few variants of greedy
    # Variant 1: Smallest k first
    set1 = greedy_heuristic()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
        
    # Variant 2: Largest k first
    def greedy_heuristic_largest():
        sums = {Fraction(0, 1)}
        selected = []
        available = list(range(1, 23))
        
        while available:
            best_k = None
            best_score = -1
            
            for k in available:
                x = Fraction(1, k)
                if not can_add(x, sums):
                    continue
                
                score = k # Prefer larger k
                if score > best_score:
                    best_score = score
                    best_k = k
            
            if best_k is None:
                break
            
            x = Fraction(1, best_k)
            sums = add_element(x, sums)
            selected.append(best_k)
            available.remove(best_k)
            
        return selected

    set2 = greedy_heuristic_largest()
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    # Randomized search
    while time.time() - start_time < time_limit:
        random.shuffle(pool := list(range(1, 23)))
        sums = {Fraction(0, 1)}
        selected = []
        
        for k in pool:
            x = Fraction(1, k)
            if not can_add(x, sums):
                continue
            
            sums = add_element(x, sums)
            selected.append(k)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected

    return best_set
