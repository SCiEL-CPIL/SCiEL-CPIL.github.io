import random
import time
from itertools import combinations

def is_sidon(s):
    """
    Check if a list of integers s is a Sidon set.
    A Sidon set requires all pairwise sums s[i] + s[j] for i <= j to be distinct.
    """
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def count_sidon_subsets(s, k):
    """
    Count the number of subsets of size k in s that are Sidon sets.
    """
    count = 0
    for subset in combinations(s, k):
        if is_sidon(list(subset)):
            count += 1
    return count

def max_sidon_size(s):
    """
    Find the maximum size of a Sidon subset in s.
    s must be a list of integers.
    """
    n = len(s)
    # Try sizes from n down to 1
    for k in range(n, 0, -1):
        for subset in combinations(s, k):
            if is_sidon(list(subset)):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Strategy:
    # 1. Use a conflict graph model to guide a local search.
    #    We want to minimize the max Sidon subset size.
    #    The objective is non-smooth (step function).
    #    We use a heuristic: maximize the number of "conflicts" (sum collisions)
    #    in the ambient set, as this tends to break Sidon structures.
    #    
    #    However, simply maximizing total collisions might not minimize the *max* subset size.
    #    A better proxy is to minimize the number of Sidon subsets of size k for small k.
    #    Specifically, if we can eliminate all Sidon-4s, max size <= 3.
    #    If we can't, we try to minimize Sidon-5s, etc.
    #
    #    Let's refine the local search to be more aggressive and use a better fitness function.
    #    Fitness: (count_sidon_4, count_sidon_5, count_sidon_6, ...) lexicographically.
    #    Or simply: sum(count_sidon_k * 10^(max_k - k)) to prioritize larger k? No, we want to minimize max k.
    #    So lexicographic on (count_4, count_5, ...) is good because count_4=0 is best.
    
    def fitness(s):
        """
        Calculate a fitness tuple for local search.
        Lower is better.
        """
        # We only need to check up to size 6 or 7 for 12 elements usually.
        # If no Sidon-4, max is 3.
        # If no Sidon-5, max is 4.
        c4 = count_sidon_subsets(s, 4)
        if c4 == 0:
            return (0, 0, 0) # Max size <= 3
        c5 = count_sidon_subsets(s, 5)
        if c5 == 0:
            return (c4, 0, 0) # Max size = 4
        c6 = count_sidon_subsets(s, 6)
        if c6 == 0:
            return (c4, c5, 0) # Max size = 5
        return (c4, c5, c6)

    def local_search(time_limit):
        end_time = time.time() + time_limit
        best_fitness = None
        best_s = None
        R = 40 # Smaller range encourages collisions
        
        # Initial random set
        s = random.sample(range(R + 1), 12)
        s.sort()
        cur_f = fitness(s)
        if best_fitness is None or cur_f < best_fitness:
            best_fitness = cur_f
            best_s = s[:]
        
        if best_fitness[0] == 0:
            return best_s
        
        while time.time() < end_time:
            # Pick a random index to move
            idx = random.randint(0, 11)
            old_val = s[idx]
            
            improved = False
            # Try a few random moves
            for _ in range(10): # Increased attempts per step
                new_val = random.randint(0, R)
                if new_val == old_val or new_val in s:
                    continue
                
                s[idx] = new_val
                s.sort()
                
                new_f = fitness(s)
                
                if new_f < cur_f:
                    cur_f = new_f
                    improved = True
                    if new_f[0] == 0:
                        return s
                    if best_fitness is None or new_f < best_fitness:
                        best_fitness = new_f
                        best_s = s[:]
                    break
                else:
                    # Revert
                    s[idx] = old_val
                    s.sort()
            
            if not improved:
                # Occasional random restart or larger jump
                if random.random() < 0.1:
                    # Jump to a new random point
                    s = random.sample(range(R + 1), 12)
                    s.sort()
                    cur_f = fitness(s)
                    if best_fitness is None or cur_f < best_fitness:
                        best_fitness = cur_f
                        best_s = s[:]
                    if cur_f[0] == 0:
                        return s
                        
        return best_s

    # Run the local search
    result = local_search(deadline - time.time())
    
    if result:
        return result
    
    # Fallback
    return list(range(12))
