import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    N = 256
    # Pre-compute reciprocals for faster scoring
    recips = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    def can_add(x, S):
        """
        Check if x can be added to S without creating a 3-term arithmetic progression.
        S is a set of integers.
        """
        # Case 1: x is the middle term. a + c = 2x with a, c in S.
        # Iterate through a in S. If a >= 2x, c <= 0, which is not in S (S subset 1..N).
        target_sum = 2 * x
        for a in S:
            if a >= target_sum:
                continue
            c = target_sum - a
            if c in S:
                return False
        
        # Case 2: x is an end term.
        # Either a + x = 2b => a = 2b - x (with a, b in S)
        # Or x + c = 2b => c = 2b - x (with c, b in S)
        # In both cases, we need to check if there exists b in S such that 2b - x is in S.
        # Note: If 2b - x == x, then 2b = 2x => b = x. But x is not in S, so this case is impossible.
        for b in S:
            y = 2 * b - x
            if y in S:
                return False
        
        return True

    def score(S):
        return sum(recips[x] for x in S)

    # 1. Greedy Initialization
    # Add numbers in increasing order (prioritizing high reciprocal weight)
    S = set()
    for x in range(1, N + 1):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = score(S)
    current_S = set(S)
    current_score = best_score
    
    # 2. Optimization Loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any valid number
        # We check in increasing order to prioritize small numbers
        for x in range(1, N + 1):
            if x not in current_S:
                if can_add(x, current_S):
                    current_S.add(x)
                    current_score += recips[x]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
                    break # Restart scan after one successful add
        
        if improved:
            continue
        
        # Phase 2: Directed Swaps
        # Try to replace a larger element (lower weight) with a smaller one (higher weight)
        # Sort current elements by value descending (largest first, as they are least valuable)
        elements_to_remove = sorted(current_S, reverse=True)
        
        swap_made = False
        for out_x in elements_to_remove:
            if time.time() > deadline - 1.0:
                break
            
            # Tentatively remove out_x
            temp_S = set(current_S)
            temp_S.remove(out_x)
            
            # Try to add a smaller y
            # We only consider y < out_x for guaranteed improvement
            for in_y in range(1, out_x):
                if in_y not in temp_S:
                    if can_add(in_y, temp_S):
                        # Perform swap
                        current_S = temp_S
                        current_S.add(in_y)
                        current_score += recips[in_y] - recips[out_x]
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(current_S)
                        
                        swap_made = True
                        break # Restart swap search after one successful swap
            
            if swap_made:
                break
        
        if swap_made:
            continue
            
        # Phase 3: Random Perturbation (Simulated Annealing style)
        # If stuck in a local optimum, perform random swaps to explore
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
                
            if not current_S:
                break
                
            out_x = random.choice(list(current_S))
            in_candidates = [x for x in range(1, N + 1) if x not in current_S]
            if not in_candidates:
                break
            in_y = random.choice(in_candidates)
            
            temp_S = set(current_S)
            temp_S.remove(out_x)
            if can_add(in_y, temp_S):
                delta = recips[in_y] - recips[out_x]
                # Accept if better, or with a small probability to escape local optima
                if delta > 0 or random.random() < 0.05:
                    current_S = temp_S
                    current_S.add(in_y)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)

    return list(best_S)
