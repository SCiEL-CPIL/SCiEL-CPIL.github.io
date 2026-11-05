import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    # Initial guess for set size
    # k^2/2 >= 129 -> k >= 16
    # Let's start with k=20
    
    def get_counts(A):
        """Returns array of counts for sums in [0, 256]"""
        counts = np.zeros(max_sum + 1, dtype=int)
        A_list = list(A)
        k = len(A_list)
        for i in range(k):
            ai = A_list[i]
            # ai + ai
            s = ai + ai
            if s <= max_sum:
                counts[s] += 1
            for j in range(i + 1, k):
                aj = A_list[j]
                s = ai + aj
                if s <= max_sum:
                    counts[s] += 1
        return counts

    def get_cost(A):
        counts = get_counts(A)
        # Check coverage
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            # Penalize uncovered sums heavily
            return 1000 + np.sum(target_counts == 0)
        return np.max(target_counts)

    # Initial random set
    random.seed(42)
    A = set(random.sample(range(0, N + 1), 20))
    
    best_cost = get_cost(A)
    best_A = set(A)
    
    # Local search
    # 1. Ensure coverage
    # 2. Minimize max multiplicity
    
    # Phase 1: Cover all sums
    # While there are uncovered sums, add the best element
    while time.time() < deadline:
        counts = get_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        uncovered = [s for s in target_sums if target_counts[s - min_sum] == 0]
        
        if not uncovered:
            break
            
        # Find element x in [0, 128] \ A that covers most uncovered sums
        best_x = None
        best_gain = -1
        
        # Candidate elements: 0 to 128
        # To speed up, we can just iterate through all possible additions
        for x in range(0, N + 1):
            if x in A:
                continue
            # Calculate how many uncovered sums would be covered by adding x
            # New sums formed: x + y for y in A
            # We only care about sums in uncovered list
            gain = 0
            # Optimization: Precompute which y in A produce sums in uncovered?
            # Simpler: just iterate A
            for y in A:
                s = x + y
                if min_sum <= s <= max_sum and target_counts[s - min_sum] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain == 0:
            # Stuck? Try random addition
            candidates = [x for x in range(0, N + 1) if x not in A]
            if candidates:
                best_x = random.choice(candidates)
            else:
                break
        
        A.add(best_x)
        
        # If set gets too large, try removing elements that don't hurt coverage or reduce cost
        # But first, let's just get coverage.
        
    # Phase 2: Minimize peak multiplicity
    # Use simulated annealing or hill climbing
    # Move: Swap an element in A with one outside A, or add/remove
    # Objective: Minimize max multiplicity among covered sums.
    
    current_cost = get_cost(A)
    
    while time.time() < deadline:
        # Try a random move
        if random.random() < 0.1 and len(A) > 15:
            # Try removal
            x = random.choice(list(A))
            A.remove(x)
            new_cost = get_cost(A)
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                A.add(x)
        elif random.random() < 0.9:
            # Try swap
            if len(A) > 0:
                x_in = random.choice(list(A))
                x_out = random.choice([x for x in range(0, N + 1) if x not in A])
                A.remove(x_in)
                A.add(x_out)
                new_cost = get_cost(A)
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    # Revert
                    A.remove(x_out)
                    A.add(x_in)
        
        # If we reach cost 2 or 3, we can stop early?
        # Theoretical minimum is 1? No, sums are symmetric.
        # If cost is 2, it's very good.
        if best_cost <= 2:
            break

    return list(best_A)
