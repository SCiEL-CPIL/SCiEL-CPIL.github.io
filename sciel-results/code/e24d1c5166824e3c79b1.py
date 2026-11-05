"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_CACHE = None

def _compute_best():
    """
    Perform Simulated Annealing to find a Sidon set with minimal diameter
    covering differences 1..30.
    """
    deadline = time.time() + 140  # Leave 10s margin
    
    def cost(a):
        """
        Returns a cost value. Lower is better.
        Penalize missing diffs in 1..30 and Sidon violations.
        """
        if len(a) != len(set(a)):
            return float('inf')
        if 0 not in a:
            return float('inf')
        
        diffs = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                diffs.append(a[j] - a[i])
        
        diff_set = set(diffs)
        
        # Check Sidon: all diffs must be distinct
        collisions = len(diffs) - len(diff_set)
        
        # Check coverage
        missing = 0
        for d in range(1, 31):
            if d not in diff_set:
                missing += 1
        
        # Diameter
        diam = max(a)
        
        # Weighted cost: prioritize validity (collisions and missing)
        # Then minimize diameter.
        # If collisions > 0 or missing > 0, it's invalid.
        # We want to drive collisions and missing to 0 first.
        return (collisions * 10000 + missing * 1000) + diam

    def is_valid(a):
        diffs = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                diffs.append(a[j] - a[i])
        if len(diffs) != len(set(diffs)):
            return False
        diff_set = set(diffs)
        for d in range(1, 31):
            if d not in diff_set:
                return False
        return True

    best_set = None
    best_cost = float('inf')
    
    # Try multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random set of size 9
        # Start with 0 and 8 random numbers
        # Heuristic: keep them somewhat spread out to reduce collisions initially?
        # Or keep them close to pack them?
        # Random sampling from 1..100
        a = [0] + sorted(random.sample(range(1, 100), 8))
        a.sort()
        
        cur_cost = cost(a)
        if cur_cost < best_cost:
            best_cost = cur_cost
            if is_valid(a):
                best_set = a[:]
        
        # Simulated Annealing
        T = 1000.0
        cooling = 0.995
        
        while T > 1.0 and time.time() < deadline:
            # Pick a random element to move (index 1..8)
            i = random.randint(1, 8)
            old_val = a[i]
            
            # Choose new value
            # Bias towards nearby values
            step = random.choice([-2, -1, 1, 2, 5, -5, 10, -10])
            new_val = old_val + step
            
            # Ensure positive and distinct
            if new_val <= 0:
                continue
            if new_val in a:
                continue
            
            # Keep sorted? Not strictly necessary for cost, but good for sanity
            a[i] = new_val
            a.sort()
            
            new_cost = cost(a)
            delta = new_cost - cur_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    if is_valid(a):
                        best_set = a[:]
            else:
                # Reject
                a[i] = old_val
                a.sort()
                
            T *= cooling
            
        if best_set is not None:
            # If we found a valid set, we can try to optimize diameter further
            # But for now, break if we have a good one?
            # No, keep searching for smaller diameter if cost allows.
            # The cost function includes diameter, so it will naturally minimize it.
            pass

    if best_set is None:
        # Fallback: Greedy construction
        # This is slower but guaranteed to find *some* valid set eventually
        # We use a greedy algorithm that adds elements one by one
        # to cover the next missing difference.
        
        a = [0]
        target_diffs = set(range(1, 31))
        
        for _ in range(20): # Max 20 elements, though 9 is enough
            if not target_diffs:
                break
            if time.time() > deadline - 5:
                break
                
            # Find the smallest missing difference
            # We want to add an element x such that (x - y) or (y - x) covers missing diffs
            best_x = None
            best_score = float('inf')
            
            # Try candidates
            max_val = max(a) + 50 if a else 50
            for x in range(1, max_val + 1):
                if x in a:
                    continue
                trial = sorted(a + [x])
                
                # Check Sidon
                diffs = []
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        d = trial[j] - trial[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.append(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                diff_set = set(diffs)
                # Score: number of newly covered diffs (higher is better)
                # and diameter (lower is better)
                new_covered = len(target_diffs.intersection(diff_set))
                score = -new_covered * 1000 + max(trial)
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Try larger range
                for x in range(max_val + 1, max_val + 200):
                    if x in a:
                        continue
                    trial = sorted(a + [x])
                    diffs = []
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            d = trial[j] - trial[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.append(d)
                        if not valid:
                            break
                    if not valid:
                        continue
                    diff_set = set(diffs)
                    new_covered = len(target_diffs.intersection(diff_set))
                    score = -new_covered * 1000 + max(trial)
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is not None:
                a.append(best_x)
                a.sort()
                diffs = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial)))
                target_diffs -= diffs
            else:
                break
                
        best_set = a

    return best_set

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _compute_best()
    return _CACHE
# EVOLVE-BLOCK-END
