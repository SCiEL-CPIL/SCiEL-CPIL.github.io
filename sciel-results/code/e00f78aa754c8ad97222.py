import random
import time

def construct():
    # Known valid set: 63..128 (size 66)
    current_set = set(range(63, 129))
    best_set = set(current_set)
    
    start_time = time.time()
    # Allow 55 seconds for search, return by 59s
    deadline = 55.0
    
    def is_valid_addition(S, x):
        # Check if adding x to S creates a forbidden triple
        # Triple must include x.
        # So we check all pairs a, b in S.
        # If a+b in S, a+x in S, b+x in S -> Invalid
        for a in S:
            # Optimization: if a > x, we might have checked (x, a) already?
            # No, we iterate all a, b.
            # We can restrict a < b to halve work?
            # But a,b are from S, x is new.
            # The condition is symmetric for a,b.
            # Let's just do all pairs. 128^2 is small.
            for b in S:
                if a == b:
                    continue
                if (a + b) in S and (a + x) in S and (b + x) in S:
                    return False
        return True

    # Hill Climbing
    while time.time() - start_time < deadline:
        # Try to add a random element
        missing = [i for i in range(1, 129) if i not in current_set]
        if missing:
            x = random.choice(missing)
            if is_valid_addition(current_set, x):
                current_set.add(x)
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
        
        # Try a random swap to explore
        if current_set:
            y = random.choice(list(current_set))
            current_set.remove(y)
            missing = [i for i in range(1, 129) if i not in current_set]
            if missing:
                x = random.choice(missing)
                if is_valid_addition(current_set, x):
                    current_set.add(x)
                    if len(current_set) > len(best_set):
                        best_set = set(current_set)
                else:
                    # Revert
                    current_set.add(y)
        
        # If stuck (no improvement in last 1000 steps), revert to best
        # This is a simple heuristic.
        if len(current_set) < len(best_set):
             # If we degraded, revert
             current_set = set(best_set)

    return list(best_set)
