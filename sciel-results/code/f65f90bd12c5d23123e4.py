import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s limit
    
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    # Build candidate pool: smooth numbers and small integers
    pool = set()
    for i in range(1, 300):
        pool.add(i)
    for i in range(1, 10001):
        if is_smooth(i, [2, 3, 5]):
            pool.add(i)
        if is_smooth(i, [2, 3, 5, 7]):
            pool.add(i)
    
    # Sort by "collision potential" - numbers that appear in many factorizations
    # and small numbers that appear in many sums
    def collision_score(x):
        score = 0
        # Count how many pairs (a,b) with a*b == x have a,b in reasonable range
        for a in range(1, min(100, int(math.isqrt(x)) + 1)):
            if x % a == 0:
                b = x // a
                if b <= 100:
                    score += 1
        # Small numbers have high sum-collision potential
        if x <= 100:
            score += (100 - x) * 0.1
        return score
    
    pool_list = sorted(pool, key=lambda x: (-collision_score(x), x))
    
    # Fast evaluation using sorted lists
    def evaluate_fast(A):
        A_sorted = sorted(A)
        sums = set()
        prods = set()
        for i, a in enumerate(A_sorted):
            for j in range(i, len(A_sorted)):
                b = A_sorted[j]
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    # Greedy clique construction: build set by greedily adding element
    # that minimizes marginal increase in max(|A+A|, |A*A|)
    def greedy_construct(pool_candidates, n_select=20, deadline_check=None):
        A = []
        current_sums = set()
        current_prods = set()
        used = set()
        
        for _ in range(n_select):
            if deadline_check and time.time() > deadline_check:
                break
            best_elem = None
            best_cost = float('inf')
            
            for elem in pool_candidates:
                if elem in used:
                    continue
                
                # Compute marginal cost
                new_sums = set(current_sums)
                new_prods = set(current_prods)
                for a in A:
                    new_sums.add(a + elem)
                    new_prods.add(a * elem)
                new_sums.add(2 * elem)
                new_prods.add(elem * elem)
                
                cost = max(len(new_sums), len(new_prods))
                if cost < best_cost:
                    best_cost = cost
                    best_elem = elem
            
            if best_elem is None:
                break
            
            A.append(best_elem)
            used.add(best_elem)
            # Update current sets
            for a in A[:-1]:
                current_sums.add(a + best_elem)
                current_prods.add(a * best_elem)
            current_sums.add(2 * best_elem)
            current_prods.add(best_elem * best_elem)
        
        return A
    
    # Branch and bound for maximum weight selection
    # Weight of a subset S = -max(|S+S|, |S*S|) (we want to maximize this)
    def branch_and_bound(pool_candidates, n_select=20, deadline_check=None, 
                         initial_solution=None, max_nodes=50000):
        best_solution = initial_solution[:] if initial_solution else []
        best_cost = evaluate_fast(initial_solution) if initial_solution else float('inf')
        
        # Precompute pairwise interactions for pruning
        # For each pair (i,j), compute the additional sums and products
        n_pool = len(pool_candidates)
        
        # Sort pool by collision score for better pruning
        pool_sorted = sorted(pool_candidates, key=lambda x: (-collision_score(x), x))
        
        nodes_visited = 0
        
        def bb_recursive(selected, remaining_pool, depth):
            nonlocal best_solution, best_cost, nodes_visited
            
            if time.time() > deadline_check:
                return
            if nodes_visited > max_nodes:
                return
            
            nodes_visited += 1
            
            if depth == n_select:
                cost = evaluate_fast(selected)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = selected[:]
                return
            
            # Pruning: if we can't beat best_cost even with optimal remaining, skip
            # Lower bound on final cost: current partial cost + minimum possible increase
            if selected:
                partial_sums = set()
                partial_prods = set()
                for i, a in enumerate(selected):
                    for j in range(i, len(selected)):
                        b = selected[j]
                        partial_sums.add(a + b)
                        partial_prods.add(a * b)
                partial_cost = max(len(partial_sums), len(partial_prods))
                # Even with 0 additional elements contributing, cost >= partial_cost
                if partial_cost >= best_cost:
                    return
            
            # Greedy upper bound: assume we can add remaining elements with minimal increase
            # For each candidate in remaining_pool, compute marginal cost
            candidates = []
            for elem in remaining_pool:
                if elem in selected:
                    continue
                # Quick estimate: how many new sums/prods does this add?
                new_sums = 0
                new_prods = 0
                for a in selected:
                    s = a + elem
                    p = a * elem
                    # Rough estimate without full set tracking
                    new_sums += 1
                    new_prods += 1
                new_sums += 1  # for 2*elem
                new_prods += 1  # for elem*elem
                
                candidates.append((new_sums + new_prods, elem))
            
            candidates.sort()
            
            # Branch on top candidates
            for _, elem in candidates[:min(10, len(candidates))]:
                selected.append(elem)
                bb_recursive(selected, remaining_pool, depth + 1)
                selected.pop()
        
        bb_recursive([], pool_sorted, 0)
        return best_solution, best_cost
    
    # Local search with structured moves (not random)
    def structured_local_search(A, pool_candidates, deadline_check=None, max_iters=20000):
        A_set = set(A)
        current_cost = evaluate_fast(A_set)
        best_A = A_set.copy()
        best_cost = current_cost
        
        # Precompute all possible swaps
        no_improve = 0
        T = 5.0
        
        for iteration in range(max_iters):
            if time.time() > deadline_check:
                break
            
            # Try structured moves: replace one element with another
            # Prioritize elements that create more collisions
            elem_list = list(A_set)
            old_elem = elem_list[iteration % len(elem_list)]
            
            # Find best replacement
            best_new_elem = None
            best_new_cost = current_cost
            
            # Sample from pool, weighted by collision score
            for _ in range(20):
                candidate = random.choice(pool_candidates)
                if candidate in A_set:
                    continue
                
                A_set.remove(old_elem)
                A_set.add(candidate)
                new_cost = evaluate_fast(A_set)
                A_set.remove(candidate)
                A_set.add(old_elem)
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_elem = candidate
            
            if best_new_elem is not None:
                A_set.remove(old_elem)
                A_set.add(best_new_elem)
                current_cost = best_new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = A_set.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # Accept worse solutions with decreasing probability
            if no_improve > 50:
                # Try random perturbation
                old_elem = random.choice(list(A_set))
                candidate = random.choice(pool_candidates)
                if candidate not in A_set:
                    A_set.remove(old_elem)
                    A_set.add(candidate)
                    new_cost = evaluate_fast(A_set)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_A = A_set.copy()
                        current_cost = new_cost
                        no_improve = 0
                    else:
                        # Revert
                        A_set.remove(candidate)
                        A_set.add(old_elem)
                        no_improve += 1
            
            T *= 0.999
            
            if no_improve > 200:
                break
        
        return best_A, best_cost
    
    # Main search loop with multiple strategies
    best_overall_A = None
    best_overall_cost = float('inf')
    
    # Strategy 1: Greedy construction from different starting points
    for start_offset in range(0, min(50, len(pool_list)), 10):
        if time.time() > deadline:
            break
        
        # Take a subset of pool starting at different offsets
        subset = pool_list[start_offset:start_offset + 100]
        greedy_result = greedy_construct(subset, 20, deadline)
        if len(greedy_result) == 20:
            cost = evaluate_fast(set(greedy_result))
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = set(greedy_result)
    
    # Strategy 2: Branch and bound with greedy initialization
    if best_overall_A is None:
        initial = pool_list[:20]
    else:
        initial = list(best_overall_A)
    
    bb_result, bb_cost = branch_and_bound(pool_list[:200], 20, deadline, initial)
    if bb_cost < best_overall_cost:
        best_overall_cost = bb_cost
        best_overall_A = set(bb_result)
    
    # Strategy 3: Structured local search from best solution
    if best_overall_A is not None:
        ls_result, ls_cost = structured_local_search(list(best_overall_A), pool_list[:300], deadline)
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_A = ls_result
    
    # Strategy 4: Multiple random restarts with greedy + local search
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random subset of pool
        random_subset = random.sample(pool_list, min(100, len(pool_list)))
        greedy_result = greedy_construct(random_subset, 20, deadline)
        if len(greedy_result) == 20:
            cost = evaluate_fast(set(greedy_result))
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_A = set(greedy_result)
            
            # Refine with local search
            ls_result, ls_cost = structured_local_search(list(greedy_result), pool_list[:200], deadline, max_iters=5000)
            if ls_cost < best_overall_cost:
                best_overall_cost = ls_cost
                best_overall_A = ls_result
    
    # Final safety check
    if best_overall_A is None or len(best_overall_A) != 20:
        return list(range(1, 21))
    
    return sorted(list(best_overall_A))

# EVOLVE-BLOCK-END
