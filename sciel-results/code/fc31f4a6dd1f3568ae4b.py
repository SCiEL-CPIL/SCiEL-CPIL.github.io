import time
import random

def construct():
    N = 256
    deadline = time.time() + 85.0 # Safety margin
    
    # Precompute APs? Or just check on the fly.
    # On-the-fly check is O(|S|).
    
    def can_add(S, x):
        # Check if S U {x} has a 3-AP
        # 1. x is middle: a, x, c -> a+c = 2x
        #    Need to check if there exist a, c in S such that a+c=2x and a!=c.
        #    Since S is a set, we can iterate a in S, check if 2x-a in S.
        #    But we must ensure a != c, i.e., a != x. Since x not in S, a!=x is always true for a in S.
        #    However, if a=c, then 2a=2x => a=x, which is not in S. So just check pairs.
        for a in S:
            c = 2*x - a
            if c in S and c != a: # c!=a is redundant if x not in S? 
                # If c==a, then 2a=2x => a=x. x not in S. So c!=a is guaranteed.
                return False
        
        # 2. x is first: x, y, z -> z = 2y - x
        #    Need y, z in S such that z = 2y - x.
        #    y must be in S. z must be in S.
        #    Also y > x (for non-constant AP, strictly increasing).
        #    If y <= x, then z <= y <= x, so x is not the smallest? 
        #    The AP is non-constant, so distinct terms.
        #    If x is the smallest, y > x and z > y.
        #    If x is the largest, see case 3.
        #    If x is in the middle, see case 1.
        #    So for x to be the first term, we need y, z in S with x < y < z and 2y = x+z.
        #    Equivalently, for each y in S with y > x, check if 2y - x in S.
        for y in S:
            if y > x:
                z = 2*y - x
                if z in S:
                    return False
            
        # 3. x is last: a, y, x -> a = 2y - x
        #    Need a, y in S such that a < y < x and 2y = a+x.
        #    Equivalently, for each y in S with y < x, check if 2y - x in S.
        #    Note: a must be >= 1.
        for y in S:
            if y < x:
                a = 2*y - x
                if a in S and a >= 1:
                    return False
        
        return True

    def score(S):
        return sum(1.0/i for i in S)

    # Initial Solution: Greedy Ascending (Seed)
    S = set()
    for x in range(1, N+1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = score(best_S)
    
    # Local Search
    # We will try to improve by adding or swapping.
    # To avoid infinite loops or getting stuck, we can use a simple hill climbing with random restarts.
    
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding any element
            # Sort candidates by 1/x descending (i.e., x ascending) to prefer high weight
            candidates = [x for x in range(1, N+1) if x not in S]
            candidates.sort() # Smaller x first
            
            for x in candidates:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break # Restart loop to check new state
            
            if not improved:
                # 2. Try swapping
                # Remove y, Add x where x < y (to increase sum)
                # Iterate y in S, x in not S
                # To be efficient, we can just iterate all pairs.
                # |S| ~ 100, |not S| ~ 150. Pairs ~ 15000.
                # can_add is O(|S|) ~ 100. Total ~ 1.5M ops per pass.
                # This might be slow in Python if done many times.
                # Optimization: Pre-compute conflicts?
                
                swapped = False
                # Randomize order to escape local optima?
                # Let's stick to deterministic first, then randomize if stuck.
                
                for y in list(S):
                    for x in range(1, y): # x must be smaller to improve score
                        if x not in S:
                            # Check if S - {y} + {x} is valid
                            # This is equivalent to:
                            # 1. S - {y} is valid (it is, since S is valid)
                            # 2. x can be added to S - {y}
                            
                            # We can check can_add(S - {y}, x)
                            # But creating a new set is expensive.
                            # We can modify can_add to accept a "blocked" element?
                            # Or just create a temporary set.
                            
                            temp_S = S.copy()
                            temp_S.remove(y)
                            if can_add(temp_S, x):
                                S = temp_S
                                S.add(x)
                                swapped = True
                                break
                    if swapped:
                        break
                
                if swapped:
                    improved = True
                    # Update best
                    curr_score = score(S)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_S = set(S)
        
        # If we exited the inner loop without improvement, we are at a local optimum.
        # Perturb the solution to try to escape.
        # Strategy: Remove a random element from S, then re-run greedy add/swap.
        # Or: Randomly swap a valid pair that doesn't improve score immediately?
        # Let's try: Pick a random y in S, remove it. Then run the "Add" phase again.
        # This effectively explores a neighborhood.
        
        if len(S) > 0:
            # Remove a random element
            y_to_remove = random.choice(list(S))
            S.remove(y_to_remove)
            # Re-optimize from this perturbed state
            # The inner loop above will handle adding back better elements.
        
        # Update best
        curr_score = score(S)
        if curr_score > best_score:
            best_score = curr_score
            best_S = set(S)

    return list(best_S)
