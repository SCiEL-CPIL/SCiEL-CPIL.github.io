"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # Safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80
    
    def check_can_add(selected, x, triple_sums):
        """Check if x can be added to selected without creating duplicate triple sums."""
        new_sums = set()
        # 3x
        s = 3 * x
        if s in triple_sums:
            return False
        new_sums.add(s)
        
        # 2x + s for s in selected
        for s_val in selected:
            s = 2 * x + s_val
            if s in triple_sums or s in new_sums:
                return False
            new_sums.add(s)
        
        # x + s + t for s, t in selected (s <= t)
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in triple_sums or s in new_sums:
                    return False
                new_sums.add(s)
        
        return True
    
    def build_greedy(start_from=1, max_size=MAX_SIZE):
        """Greedy: try to add smallest possible numbers."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 5:
                break
            
            if check_can_add(selected, x, triple_sums):
                # Add x and update triple_sums
                selected.append(x)
                triple_sums.add(3 * x)
                for s_val in selected[:-1]:
                    triple_sums.add(2 * x + s_val)
                n = len(selected) - 1
                for i in range(n):
                    for j in range(i, n):
                        triple_sums.add(x + selected[i] + selected[j])
        
        return selected
    
    # Try greedy from 1
    result = build_greedy(1, MAX_SIZE)
    
    if len(result) >= MAX_SIZE:
        return result
    
    # If greedy didn't reach max, try different starting points or strategies
    # Try a few different strategies within remaining time
    best = result
    
    if time.time() < DEADLINE - 10:
        # Try greedy but skip some early numbers to see if we can fit more
        for skip_start in range(2, 10):
            if time.time() > DEADLINE - 5:
                break
            r = build_greedy(skip_start, MAX_SIZE)
            if len(r) > len(best):
                best = r
            if len(best) >= MAX_SIZE:
                break
    
    return best
# EVOLVE-BLOCK-END
