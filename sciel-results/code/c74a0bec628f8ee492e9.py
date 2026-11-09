import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 256
    
    # Precompute gcd matrix for speed
    # gcd_mat[i][j] = gcd(i+1, j+1) for 0-indexed
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def is_compatible(a, b):
        return gcd(a, b) == 1
    
    def can_add_to_set(num, selected_set):
        for x in selected_set:
            if gcd(num, x) > 1:
                return False
        return True
    
    def can_add_to_list(num, selected_list):
        for x in selected_list:
            if gcd(num, x) > 1:
                return False
        return True
    
    def get_sum(lst):
        return sum(lst)
    
    def greedy_construct(order):
        """Greedy: add numbers in given order if compatible with all selected."""
        selected = []
        selected_set = set()
        for num in order:
            if can_add_to_list(num, selected):
                selected.append(num)
                selected_set.add(num)
        return selected
    
    def local_search(initial, max_iters=50000):
        """Local search with 1-for-1 and 2-for-1 swaps."""
        selected = list(initial)
        selected_set = set(selected)
        current_sum = get_sum(selected)
        
        # Build conflict sets for quick lookup
        # For each num, which other nums in [1..256] conflict with it
        conflict_with = {}
        for num in range(1, N+1):
            conflicts = set()
            for other in range(1, N+1):
                if num != other and gcd(num, other) > 1:
                    conflicts.add(other)
            conflict_with[num] = conflicts
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 2:
                break
            
            # Try 1-for-1 swaps: remove one, add one
            for i, a in enumerate(selected):
                # Try adding each num not in selected
                for b in range(1, N+1):
                    if b in selected_set:
                        continue
                    if gcd(a, b) > 1:
                        continue  # b conflicts with a, but we're removing a
                    # Check b is compatible with all others
                    compatible = True
                    for j, x in enumerate(selected):
                        if j == i:
                            continue
                        if gcd(b, x) > 1:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    new_sum = current_sum - a + b
                    if new_sum > current_sum:
                        selected[i] = b
                        selected_set.discard(a)
                        selected_set.add(b)
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try 2-for-1 swaps: remove two, add one
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    a, b = selected[i], selected[j]
                    # Try adding each num not in selected
                    for c in range(1, N+1):
                        if c in selected_set:
                            continue
                        # c must be coprime with all selected except a and b
                        compatible = True
                        for k, x in enumerate(selected):
                            if k == i or k == j:
                                continue
                            if gcd(c, x) > 1:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        new_sum = current_sum - a - b + c
                        if new_sum > current_sum:
                            # Remove a and b, add c
                            selected = [x for k, x in enumerate(selected) if k != i and k != j]
                            selected_set.discard(a)
                            selected_set.discard(b)
                            selected_set.add(c)
                            selected.append(c)
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected, current_sum
    
    # Multi-start local search
    best_solution = None
    best_sum = 0
    
    # Start 1: primes first, then composites in descending order
    primes = [a for a in range(2, N+1) if all(a % d != 0 for d in range(2, int(a**0.5)+1))]
    composites = [a for a in range(1, N+1) if a not in primes]
    order1 = [1] + primes + sorted(composites, reverse=True)
    sol1 = greedy_construct(order1)
    sol1, sum1 = local_search(sol1, max_iters=30000)
    if sum1 > best_sum:
        best_sum = sum1
        best_solution = sol1
    
    if time.time() > deadline - 10:
        return best_solution
    
    # Start 2: all numbers in descending order (greedy from largest)
    order2 = list(range(N, 0, -1))
    sol2 = greedy_construct(order2)
    sol2, sum2 = local_search(sol2, max_iters=30000)
    if sum2 > best_sum:
        best_sum = sum2
        best_solution = sol2
    
    if time.time() > deadline - 10:
        return best_solution
    
    # Start 3: random orderings
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        sol = greedy_construct(order)
        sol, s = local_search(sol, max_iters=20000)
        if s > best_sum:
            best_sum = s
            best_solution = sol
    
    # Final polish: one more local search pass on best
    if time.time() < deadline - 3 and best_solution:
        best_solution, best_sum = local_search(best_solution, max_iters=10000)
    
    return best_solution
