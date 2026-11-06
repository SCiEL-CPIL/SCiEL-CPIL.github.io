"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to select pairwise coprime integers a in [1, 511]
    # maximizing sum of 1/(512-a)
    # Numbers near 511 have the largest contribution
    
    # Strategy: Use branch and bound with greedy upper bound
    # Sort candidates by weight (1/(512-a)) descending, i.e., a descending
    
    # First, let's get a good initial solution with greedy
    # Then use local search / branch and bound to improve
    
    # Precompute weights
    n = 512
    candidates = list(range(511, 0, -1))  # sorted by weight descending
    weights = [1.0 / (n - a) for a in candidates]
    
    # Greedy initial solution
    def greedy_solution():
        chosen = []
        for i, a in enumerate(candidates):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_solution = greedy_solution()
    best_score = sum(1.0 / (n - a) for a in best_solution)
    
    # Try local search: try to swap in a better element
    # Or use a more systematic approach
    
    # For better results, let's try a branch-and-bound style approach
    # with memoization on the set of "forbidden" prime factors
    
    # Actually, let's use a smarter approach:
    # The key constraint is pairwise coprimality. 
    # Two numbers are incompatible if they share a prime factor.
    
    # Let's build a conflict graph and use maximum weight independent set
    # But that's NP-hard. With 511 nodes, we need heuristics.
    
    # Better approach: Use the fact that we want numbers near 511.
    # Let's do a more careful greedy with look-ahead, then local search.
    
    # Local search: try to replace one element with a better one
    def local_search(solution, max_iterations=1000):
        solution = list(solution)
        current_score = sum(1.0 / (n - a) for a in solution)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            # Try removing one element and adding a better one
            for i, a in enumerate(solution):
                # Try to replace a with some other b
                # b must be coprime with all elements except a
                remaining = [x for j, x in enumerate(solution) if j != i]
                for b in candidates:
                    if b == a or b in remaining:
                        continue
                    if 1.0 / (n - b) <= 1.0 / (n - a):
                        continue  # not an improvement
                    if all(gcd(b, x) == 1 for x in remaining):
                        # Replace
                        new_solution = remaining + [b]
                        new_score = current_score - 1.0 / (n - a) + 1.0 / (n - b)
                        if new_score > current_score:
                            solution = new_solution
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        return solution, current_score
    
    best_solution, best_score = local_search(best_solution)
    
    # Try multiple random restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle candidates slightly, but keep high-weight ones prioritized
        # Actually, let's try a different greedy: randomize the order among
        # similarly-weighted candidates
        shuffled = candidates[:]
        # Partial shuffle: only shuffle within groups of similar weight
        for i in range(len(shuffled) - 1):
            if random.random() < 0.3:  # 30% chance to swap
                j = min(i + random.randint(1, 10), len(shuffled) - 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        chosen = []
        for a in shuffled:
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        
        score = sum(1.0 / (n - a) for a in chosen)
        if score > best_score:
            best_score = score
            best_solution = chosen
        
        # Local search from this starting point
        if time.time() < deadline - 5:
            improved_sol, improved_score = local_search(chosen, max_iterations=200)
            if improved_score > best_score:
                best_score = improved_score
                best_solution = improved_sol
    
    # Final local search
    if time.time() < deadline - 2:
        final_sol, final_score = local_search(best_solution, max_iterations=500)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol
    
    return best_solution
# EVOLVE-BLOCK-END
