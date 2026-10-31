"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import itertools
import math
import random

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0
        return 1 + math.log(ratio) / denom

    # Simulated Annealing
    # State: a set of integers.
    # We represent the set as a sorted list.
    # Moves: Add, Remove, Replace, Shift (add constant to all non-zero elements? No, 0 must stay 0).
    # Better moves: Add x, Remove x, Swap x for y.
    
    max_val = 60
    max_iter = 5000
    start_temp = 1.0
    end_temp = 0.01
    cooling = 0.999
    
    best_global_set = [0, 1, 3]
    best_global_score = score_of(best_global_set)
    
    # Initialize with a few diverse starts
    starts = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 2, 5, 7],
        [0, 1, 2, 5, 9],
        [0, 3, 4, 8, 12],
    ]
    
    for start in starts:
        current_set = set(start)
        current_score = score_of(list(current_set))
        temp = start_temp
        
        for _ in range(max_iter):
            # Generate a neighbor
            move_type = random.choice(['add', 'remove', 'swap'])
            new_set = set(current_set)
            
            if move_type == 'add':
                # Add a random element
                x = random.randint(1, max_val)
                if x not in new_set:
                    new_set.add(x)
            elif move_type == 'remove':
                # Remove a random non-zero element
                removable = [x for x in new_set if x != 0]
                if removable:
                    x = random.choice(removable)
                    new_set.remove(x)
            elif move_type == 'swap':
                # Swap a non-zero element with a new one
                removable = [x for x in new_set if x != 0]
                if removable:
                    x = random.choice(removable)
                    y = random.randint(1, max_val)
                    if y not in new_set:
                        new_set.remove(x)
                        new_set.add(y)
            
            if len(new_set) < 2:
                continue
                
            new_score = score_of(list(new_set))
            
            # Accept if better, or with probability exp((new-old)/temp)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_set = new_set
                current_score = new_score
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_set = list(current_set)
            
            temp *= cooling
        
    return sorted(best_global_set)
# EVOLVE-BLOCK-END
