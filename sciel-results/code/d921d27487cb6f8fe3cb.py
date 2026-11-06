# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill
    
    def _greedy(order):
        """
        Construct a B2[2] set by processing candidates in the given order.
        Returns the set as a list.
        """
        chosen = []
        # counts[s] = number of pairs {a,b} in chosen with a+b=s, a<=b
        counts = {}
        
        for x in order:
            # Check if adding x violates the B2[2] condition
            # New pairs: {x,x} with sum 2x, and {x,y} for all y in chosen
            valid = True
            
            # Check 2*x
            if counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                # Check x + y for all y in chosen
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                # Update counts for new pairs involving x
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in chosen[:-1]:  # All previously chosen elements
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
        
        return chosen

    def _local_improve(initial_set, time_limit):
        """
        Try to improve the set by removing elements and trying to add others.
        Returns the improved set.
        """
        chosen = list(initial_set)
        # Rebuild counts
        counts = {}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        
        # Build a set of available numbers
        available = set(range(1, 1001)) - set(chosen)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove each element and see if we can add two or more new ones
            for i in range(len(chosen)):
                if time.time() >= time_limit:
                    break
                x = chosen[i]
                # Temporarily remove x
                chosen.remove(x)
                # Rebuild counts after removal
                counts_new = {}
                for a in range(len(chosen)):
                    for b in range(a, len(chosen)):
                        s = chosen[a] + chosen[b]
                        counts_new[s] = counts_new.get(s, 0) + 1
                
                # Try to add available elements
                added = []
                for y in sorted(available):
                    if y in chosen:
                        continue
                    # Check if y can be added
                    valid = True
                    if counts_new.get(2 * y, 0) >= 2:
                        valid = False
                    else:
                        for z in chosen:
                            s = y + z
                            if counts_new.get(s, 0) >= 2:
                                valid = False
                                break
                    if valid:
                        chosen.append(y)
                        counts_new[2 * y] = counts_new.get(2 * y, 0) + 1
                        for z in chosen[:-1]:
                            s = y + z
                            counts_new[s] = counts_new.get(s, 0) + 1
                        added.append(y)
                
                if len(added) > 1:  # Net gain
                    counts = counts_new
                    available = set(range(1, 1001)) - set(chosen)
                    improved = True
                    break
                else:
                    # Put x back
                    chosen.append(x)
                    # Rebuild counts
                    counts = {}
                    for a in range(len(chosen)):
                        for b in range(a, len(chosen)):
                            s = chosen[a] + chosen[b]
                            counts[s] = counts.get(s, 0) + 1
                    available = set(range(1, 1001)) - set(chosen)
        
        return chosen

    # Initial deterministic run
    initial_order = list(range(1, 1001))
    best_set = _greedy(initial_order)
    best_size = len(best_set)
    
    # Try local improvement on the initial set
    if time.time() < deadline - 5:
        improved_set = _local_improve(best_set, deadline - 5)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Randomized restarts with local improvement
    random.seed(42)
    while time.time() < deadline - 10:
        # Random permutation
        order = list(range(1, 1001))
        random.shuffle(order)
        
        current_set = _greedy(order)
        current_size = len(current_set)
        
        # Try local improvement if we have time
        if current_size >= best_size and time.time() < deadline - 15:
            improved = _local_improve(current_set, deadline - 15)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
    
    return best_set
# EVOLVE-BLOCK-END
