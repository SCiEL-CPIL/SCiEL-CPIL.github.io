import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad triples as a set of frozensets for quick lookup
    # Actually, let's precompute for each element x, and for each pair (a,b) already selected,
    # whether adding x creates a bad triple.
    
    # Better approach: for each x, group other elements by gcd(x, other)
    # Then for a pair (a,b) in selected, check if gcd(x,a) == gcd(x,b) == gcd(a,b)
    
    def can_add(x, selected_list):
        """Fast check: group selected elements by gcd with x, then check conflicts."""
        # Build dict: gcd_value -> list of elements in selected with that gcd to x
        groups = {}
        for a in selected_list:
            g = gcd_table[x][a]
            if g not in groups:
                groups[g] = []
            groups[g].append(a)
        
        # For each group with >= 2 elements, check if any pair in the group has gcd equal to the group key
        for g_val, elems in groups.items():
            if len(elems) >= 2:
                for i in range(len(elems)):
                    for j in range(i + 1, len(elems)):
                        if gcd_table[elems[i]][elems[j]] == g_val:
                            return False
        return True
    
    def is_valid(selected_set):
        """Verify the entire set is valid."""
        lst = list(selected_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                ga = gcd_table[lst[i]][lst[j]]
                for k in range(j + 1, len(lst)):
                    if ga == gcd_table[lst[i]][lst[k]] == gcd_table[lst[j]][lst[k]]:
                        return False
        return True
    
    import random
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Greedy with multiple orderings
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 40:
            break
        
        ordering = list(range(1, 129))
        rng.shuffle(ordering)
        
        selected = set()
        selected_list = []
        for x in ordering:
            if can_add(x, selected_list):
                selected.add(x)
                selected_list.append(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Phase 2: Local search - greedy fill
    if time.time() < deadline - 15:
        current = set(best_set)
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            remaining = [x for x in range(1, 129) if x not in current]
            rng.shuffle(remaining)
            improved = False
            cur_list = list(current)
            for x in remaining:
                if can_add(x, cur_list):
                    current.add(x)
                    cur_list.append(x)
                    improved = True
            if not improved:
                break
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Phase 3: 1-for-1 swap local search
    if time.time() < deadline - 10:
        current = set(best_set)
        max_swap_iters = 500
        for it in range(max_swap_iters):
            if time.time() > deadline - 10:
                break
            
            found_swap = False
            cur_list = list(current)
            for remove in cur_list:
                if time.time() > deadline - 10:
                    break
                
                temp = set(current)
                temp.discard(remove)
                temp_list = [x for x in cur_list if x != remove]
                
                remaining = [x for x in range(1, 129) if x not in temp]
                rng.shuffle(remaining)
                
                for x in remaining:
                    if can_add(x, temp_list):
                        temp.add(x)
                        current = temp
                        cur_list = list(current)
                        found_swap = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    # Phase 4: 1-for-2 swap local search
    if time.time() < deadline - 5:
        current = set(best_set)
        max_swap_iters = 200
        for it in range(max_swap_iters):
            if time.time() > deadline - 5:
                break
            
            found_swap = False
            cur_list = list(current)
            for remove in cur_list:
                if time.time() > deadline - 5:
                    break
                
                temp = set(current)
                temp.discard(remove)
                temp_list = [x for x in cur_list if x != remove]
                
                remaining = [x for x in range(1, 129) if x not in temp and x != remove]
                rng.shuffle(remaining)
                
                added = []
                for x in remaining:
                    if len(added) == 2:
                        break
                    if can_add(x, temp_list + added):
                        added.append(x)
                        temp_list.append(x)
                
                if len(added) == 2:
                    temp.update(added)
                    current = temp
                    found_swap = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    break
            
            if not found_swap:
                break
    
    # Phase 5: Final polish
    if time.time() < deadline:
        current = set(best_set)
        remaining = [x for x in range(1, 129) if x not in current]
        cur_list = list(current)
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(x, cur_list):
                current.add(x)
                cur_list.append(x)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Final validation
    if not is_valid(best_set):
        # Fallback: remove elements until valid
        lst = list(best_set)
        for i in range(len(lst)):
            if time.time() > deadline:
                break
            test_set = set(lst[:i] + lst[i+1:])
            if is_valid(test_set):
                best_set = test_set
                break
    
    return sorted(best_set)
