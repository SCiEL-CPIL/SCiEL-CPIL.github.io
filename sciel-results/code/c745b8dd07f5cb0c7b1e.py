import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Greedy Construction
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    best_set = chosen[:]
    best_score = sum(1.0 / (512 - a) for a in best_set)
    
    # 2. Local Search
    # We try to swap elements to improve the score.
    # A swap is valid if the new element is coprime to all others.
    # We iterate until no improvement is found.
    
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Create a set for fast lookup
        chosen_set = set(chosen)
        
        # Try to add any element that is coprime to all?
        # Greedy already did this in order, but maybe order matters?
        # Actually, if a set is maximal, we can't add.
        # We focus on swaps.
        
        # To be efficient, we can iterate over candidates 'a' not in chosen
        # and see if they can replace any 'b' in chosen.
        
        # Sort chosen by weight descending? No, just iterate.
        # For each a not in chosen:
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            
            # Check if a is coprime to all in chosen
            # If yes, we could add it? No, chosen is maximal?
            # Greedy produces a maximal set. 
            # Proof: If there was an a coprime to all, greedy would have added it 
            # unless it was blocked by a larger number. 
            # But if it's coprime to ALL current chosen, it wasn't blocked.
            # Wait. Greedy adds a if coprime to ALL previously added.
            # So if a is coprime to ALL in final set, it must have been added?
            # No. Greedy adds in descending order.
            # If a is small, it's added late.
            # If a is coprime to all large ones, it is added.
            # So the greedy set is maximal.
            # So we cannot add any element.
            
            # So we must swap.
            # For each b in chosen:
            for i, b in enumerate(chosen):
                # If a and b are not coprime, skip
                if gcd(a, b) != 1:
                    continue
                
                # Check if a is coprime to all other elements in chosen
                # excluding b
                valid = True
                for j, c in enumerate(chosen):
                    if i == j:
                        continue
                    if gcd(a, c) != 1:
                        valid = False
                        break
                
                if valid:
                    # Check if swap improves score
                    # w(a) > w(b)  =>  1/(512-a) > 1/(512-b)  =>  512-b > 512-a  =>  a > b
                    # So we only swap if a > b.
                    if a > b:
                        # Perform swap
                        chosen[i] = a
                        chosen_set.remove(b)
                        chosen_set.add(a)
                        improved = True
                        # Restart the inner loop? Or just continue?
                        # Restarting is safer to find more improvements.
                        break
            
            if improved:
                break # Restart outer loop
        
    return best_set
