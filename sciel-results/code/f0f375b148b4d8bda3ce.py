import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid_addition(x, reachable):
        # Check if adding x to a set with reachable sums creates sum 200
        # We don't need to update reachable, just check
        for s in reachable:
            if s + x == 200:
                return False
        # Also need to check if x itself is 200 (not possible here)
        # And if x combines with existing sums to make 200
        # The loop above checks s+x=200.
        # But what if x combines with another new sum? No, we add one at a time.
        # The condition is: no subset of (Current Set + {x}) sums to 200.
        # Subsets of Current Set are already safe.
        # New subsets must include x.
        # So we need x + S' = 200 for some S' subset of Current Set.
        # This is equivalent to checking if 200-x is in reachable.
        return (200 - x) not in reachable

    def get_cardinality(selected):
        return len(selected)

    def run_greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            if is_valid_addition(x, reachable):
                selected.append(x)
                # Update reachable
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable |= new_sums
        return selected

    # 1. Standard Greedy: 100 down to 1
    order1 = list(range(100, 0, -1))
    best_set = run_greedy(order1)
    best_size = len(best_set)
    
    # 2. Reverse Greedy: 1 up to 100
    order2 = list(range(1, 101))
    set2 = run_greedy(order2)
    if len(set2) > best_size:
        best_set = set2
        best_size = len(set2)

    # 3. Randomized Greedy
    # We can try several random orders if time permits
    base_order = list(range(1, 101))
    attempts = 0
    while time.time() < deadline:
        random.shuffle(base_order)
        current_set = run_greedy(base_order)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            # If we found a very good one, maybe stop?
            # Max possible is 100. 
            # If best_size is high, e.g., > 60, we might be close to optimal.
            # But let's keep going to be safe.
        attempts += 1
        
    return best_set
