import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = list(range(13, 25))
    best_size = 12
    
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            x = S_list[i]
            target = Fraction(1, x)
            others = [S_list[j] for j in range(n) if j != i]
            
            # Quick check: sum of all reciprocals in others
            sum_all = sum(Fraction(1, y) for y in others)
            if sum_all < target:
                continue
                
            reachable = {Fraction(0)}
            for y in others:
                val = Fraction(1, y)
                new_reachable = set()
                for s in reachable:
                    ns = s + val
                    if ns <= target:
                        new_reachable.add(ns)
                reachable.update(new_reachable)
                if target in reachable:
                    return False
        return True

    # Local Search
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Random start
        size = random.randint(12, 16)
        current_set = set(random.sample(range(1, 25), size))
        
        # Ensure valid
        if not is_valid(current_set):
            # Remove elements until valid?
            # Just skip for now, or repair
            # Let's try to repair by removing random elements
            while not is_valid(current_set) and current_set:
                # Remove random
                to_remove = random.choice(list(current_set))
                current_set.remove(to_remove)
        
        # Hill Climb
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            candidates = [x for x in range(1, 25) if x not in current_set]
            random.shuffle(candidates)
            for x in candidates:
                new_set = current_set | {x}
                if is_valid(new_set):
                    current_set = new_set
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break
            
            # Try to swap if no add worked
            if not improved and len(current_set) < 24:
                # Pick a random element to remove
                if current_set:
                    to_remove = random.choice(list(current_set))
                    current_set.remove(to_remove)
                    # Try to add something else
                    candidates = [x for x in range(1, 25) if x not in current_set]
                    random.shuffle(candidates)
                    for x in candidates:
                        new_set = current_set | {x}
                        if is_valid(new_set):
                            current_set = new_set
                            improved = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = list(current_set)
                            break
                            
    return best_set
