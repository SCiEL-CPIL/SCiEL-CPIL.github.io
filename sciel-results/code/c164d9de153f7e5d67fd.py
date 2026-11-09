import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(x, selected_list):
        """Check if x can be added to selected_list without creating a bad triple.
        selected_list is a list of integers."""
        for i in range(len(selected_list)):
            a = selected_list[i]
            ga = gcd_table[x][a]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                if ga == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True

    best_set = []
    best_size = 0
    
    # Randomized Greedy Restarts
    rng = random.Random(42)
    
    max_restarts = 200
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        
        ordering = list(range(1, 129))
        rng.shuffle(ordering)
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if x not in selected_set:
                if can_add(x, selected):
                    selected.append(x)
                    selected_set.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    # Final Polish: Try to add any missing elements to the best set
    if time.time() < deadline:
        best_set_list = best_set[:]
        best_set_set = set(best_set_list)
        
        # Try adding in natural order
        for x in range(1, 129):
            if time.time() > deadline:
                break
            if x not in best_set_set:
                if can_add(x, best_set_list):
                    best_set_list.append(x)
                    best_set_set.add(x)
                    
        # One more random pass
        remaining = [x for x in range(1, 129) if x not in best_set_set]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(x, best_set_list):
                best_set_list.append(x)
                best_set_set.add(x)
                
        best_set = best_set_list

    return sorted(best_set)
