# EVOLVE-BLOCK-START
def construct():
    def greedy_order(order):
        reachable = {0}
        selected = []
        for x in order:
            potential_200 = False
            new_reachable = set(reachable)
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    if ns == 200:
                        potential_200 = True
                        break
                    new_reachable.add(ns)
            if not potential_200:
                selected.append(x)
                reachable = new_reachable
        return selected

    # Strategy 1: Descending (Current best guess)
    set1 = greedy_order(range(100, 0, -1))
    
    # Strategy 2: Ascending
    set2 = greedy_order(range(1, 101))
    
    # Strategy 3: Alternating High-Low
    order3 = []
    for i in range(50):
        order3.append(100 - i)
        order3.append(1 + i)
    set3 = greedy_order(order3)
    
    # Strategy 4: Random (a few iterations)
    import random
    best_set = set1
    for _ in range(10):
        order4 = list(range(1, 101))
        random.shuffle(order4)
        set4 = greedy_order(order4)
        if len(set4) > len(best_set):
            best_set = set4
            
    # Verify and return best
    # Note: The greedy functions guarantee validity by construction.
    return best_set
# EVOLVE-BLOCK-END
