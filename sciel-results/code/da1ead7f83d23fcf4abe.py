import time
import random
import numpy as np

def construct():
    N = 51975
    # Get divisors
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    # We will use a list to store [m, r]
    # cov_count[x] = number of classes covering x
    # uncovered = number of x where cov_count[x] == 0
    
    best_uncovered = N
    best_classes = []
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to calculate score and update state
    # We will use numpy for speed if possible, but bytearray might be faster for simple increments/decrements in python loops if N is small?
    # N=51975 is small. Python loops over 51975 elements are fast (~ms).
    # However, doing this many times might be slow.
    # Let's use a list of ints for cov_count.
    
    # Strategy: Multiple restarts
    # 1. Greedy with ascending moduli
    # 2. Greedy with descending moduli
    # 3. Random restarts + Local Search
    
    def get_uncovered(cov):
        return sum(1 for c in cov if c == 0)
        
    def evaluate_greedy(order):
        cov = [0] * N
        classes = {} # m -> r
        uncovered = N
        
        for m in order:
            best_r = 0
            best_uncovered_after = N
            # For each r, simulate adding (m, r)
            # Optimization: 
            # Current uncovered set U.
            # Adding (m, r) covers {r, r+m, ...}.
            # New uncovered = U - {r, r+m, ...}
            # We want to maximize the number of covered elements in U.
            
            # Since N is 51975, and m >= 3, the number of residues in a class is at most 17325.
            # Iterating all r (up to m) and checking all residues in the class (up to N/m) is sum(m * N/m) = N per modulus? No.
            # Sum over r of (N/m) = N.
            # So for each modulus, we do O(N) work.
            # Total work for greedy: sum(N) over all moduli = 47 * 51975 ~ 2.4 million operations. Very fast.
            
            for r in range(m):
                # Count how many uncovered residues are in class (m, r)
                # We can do this by checking cov[x] == 0 for x in range(r, N, m)
                count_covered_uncovered = 0
                for x in range(r, N, m):
                    if cov[x] == 0:
                        count_covered_uncovered += 1
                
                if count_covered_uncovered > best_uncovered_after: # Wait, we want to minimize uncovered.
                    # Actually, we want to maximize the reduction in uncovered.
                    # Reduction = count_covered_uncovered
                    # So maximize count_covered_uncovered.
                    pass # Placeholder logic

            # Correct logic:
            best_r = 0
            max_gain = -1
            for r in range(m):
                gain = 0
                for x in range(r, N, m):
                    if cov[x] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_r = r
            
            # Apply best_r
            classes[m] = best_r
            for x in range(best_r, N, m):
                cov[x] += 1
                if cov[x] == 1:
                    uncovered -= 1
            
        return uncovered, classes, cov

    # Run a few greedy variants
    orders = [
        sorted(divisors),
        sorted(divisors, reverse=True),
        divisors[:] # original order (increasing)
    ]
    # Add some random orders
    for _ in range(10):
        random.shuffle(divisors)
        orders.append(divisors[:])
        
    for order in orders:
        if time.time() > deadline: break
        unc, classes, cov = evaluate_greedy(order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_cov = cov[:] # Store best cov for local search? Or re-derive?
            
    # Local Search on the best solution found so far
    # We need to be able to modify the solution.
    # Let's pick the best greedy result and run local search on it.
    
    # Re-run the best greedy to get cov and classes
    # Actually, let's just keep the best one found.
    # To allow local search, we need to know the current classes.
    
    # Let's structure it as:
    # 1. Find best greedy.
    # 2. Perform local search starting from that best greedy.
    
    # For local search, we need to efficiently try moves.
    # Move: change r for modulus m.
    
    # Let's assume best_classes is a dict {m: r}
    # And best_cov is the coverage array.
    
    # We'll run local search on the single best greedy solution found.
    # If time permits, we might run local search on multiple good greedy solutions.
    
    # For now, focus on one strong local search.
    
    # Identify the best greedy result
    # (The loop above already found best_uncovered and best_classes)
    # But we didn't store best_cov. Let's re-calculate it or store it.
    
    # Let's refactor to store best_cov.
    
    # ... (Refactoring logic in code)
    
    # Local Search
    # ...
    
    # Convert best_classes dict to list of lists
    result = [[m, r] for m, r in best_classes.items()]
    return result
