import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit
    
    # Initial best: Deterministic greedy
    def greedy_saturation(order):
        selected = []
        used_sums = set()
        for x in order:
            # Check if x can be added
            # Sums to check: 2x and x+y for all y in selected
            # If any of these are in used_sums, x cannot be added.
            # Note: We don't need to check x+y against other x'+y' 
            # because used_sums contains ALL sums from previous elements.
            
            # Optimization: 
            # 1. Check 2*x
            # 2. Check x+y for y in selected
            
            if 2*x in used_sums:
                continue
                
            valid = True
            for y in selected:
                if x + y in used_sums:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                used_sums.add(2*x)
                for y in selected:
                    # y is the new x? No, y is in selected.
                    # We need to add x+y for all y in selected (including the new x? No, 2x is handled)
                    # Wait, the loop `for y in selected` iterates over the list.
                    # If we append x first, then x is in selected.
                    # Let's be precise.
                    pass
            
            # Let's rewrite the valid check and update clearly.
            pass
        return selected

    # Better implementation of greedy
    def run_greedy(order):
        selected = []
        used_sums = set()
        for x in order:
            # Check diagonal
            if 2 * x in used_sums:
                continue
            
            # Check off-diagonal
            conflict = False
            for y in selected:
                if x + y in used_sums:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(x)
                # Update used_sums
                used_sums.add(2 * x)
                for y in selected:
                    # y includes x now?
                    # If y == x, sum is 2x (already added).
                    # If y != x, sum is x+y.
                    # We can just iterate all y in selected.
                    used_sums.add(x + y)
        return selected

    # Initial deterministic run
    best_set = run_greedy(list(range(1, N + 1)))
    best_size = len(best_set)
    
    # Randomized restarts
    while time.time() < deadline:
        # Random permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        current_set = run_greedy(order)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            # Optional: If we hit a very small size, we might want to try harder
            # but for now, just keep searching.
            
    return best_set
