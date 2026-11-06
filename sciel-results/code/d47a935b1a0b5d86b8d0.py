import random
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # 1. Generate Greedy Solution
    selected = set()
    total = Fraction(0, 1)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.add(i)
            total += Fraction(1, d)
    
    best_set = set(selected)
    best_deficit = 1 - total
    
    # 2. Local Search / Simulated Annealing
    # To keep it fast and deterministic-ish, we use a fixed seed for random
    random.seed(42)
    
    current_set = set(best_set)
    current_total = sum(Fraction(1, denominators[i]) for i in current_set)
    current_deficit = 1 - current_total
    
    # Precompute reciprocals
    recips = [Fraction(1, d) for d in denominators]
    
    # Simple Hill Climbing with restarts or just iterations
    # Given the structure, simple hill climbing might get stuck.
    # Let's try a few random restarts from the greedy solution.
    
    for restart in range(10):
        # Perturb the greedy solution
        if restart == 0:
            current_set = set(best_set)
        else:
            current_set = set(best_set)
            # Randomly swap 1-2 elements
            for _ in range(2):
                if len(current_set) > 0:
                    in_set = list(current_set)
                    out_set = [i for i in range(n) if i not in current_set]
                    if in_set and out_set:
                        rem = random.choice(in_set)
                        add = random.choice(out_set)
                        current_set.remove(rem)
                        current_set.add(add)
        
        current_total = sum(recips[i] for i in current_set)
        if current_total >= 1:
            continue
        current_deficit = 1 - current_total
        
        improved = True
        while improved:
            improved = False
            # Try adding
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + recips[i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < current_deficit:
                            current_set.add(i)
                            current_total = new_total
                            current_deficit = new_deficit
                            improved = True
                            break # Restart loop
            if improved:
                continue
                
            # Try removing
            for i in list(current_set):
                new_total = current_total - recips[i]
                # Removing always keeps sum < 1, but increases deficit
                # So we only remove if we can add something better afterwards?
                # Simple hill climbing doesn't do 2-opt easily.
                # Let's stick to 1-opt (add/remove/swap)
                pass
            
            # Try swapping
            for i in list(current_set):
                for j in range(n):
                    if j not in current_set:
                        new_total = current_total - recips[i] + recips[j]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < current_deficit:
                                current_set.remove(i)
                                current_set.add(j)
                                current_total = new_total
                                current_deficit = new_deficit
                                improved = True
                                break
                if improved:
                    break
        
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_set = set(current_set)
            
    return sorted(best_set)
