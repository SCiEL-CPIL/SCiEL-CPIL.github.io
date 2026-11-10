"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    # State representation:
    # x: binary vector of length n (indices 1..n mapped to 0..n-1)
    # We use a single vector x of length 2*n? No, that's complicated.
    # Let's use two sets A and B represented by boolean masks or lists.
    # For SA, a dense representation is better.
    # Let's use a single array 'assign' of length n, where:
    # assign[i] = 0: i is in neither A nor B
    # assign[i] = 1: i is in A
    # assign[i] = 2: i is in B
    # Note: The problem says A and B need not be disjoint, but usually in these
    # packing problems, sharing elements wastes capacity for differences.
    # However, the constraint is on differences. If a is in A and B, 
    # its differences contribute to both sets' difference sets.
    # Actually, if a is in both, diff(a, b) is in Diff(A) and Diff(B).
    # Since Diff(A) and Diff(B) must be disjoint, this implies Diff(A) ∩ Diff(B) = ∅.
    # If a is in both, any difference involving a and another element in A is in Diff(A).
    # Any difference involving a and another element in B is in Diff(B).
    # This creates complex interactions.
    # Given the small size (256), we can just maintain two sets A and B as Python sets
    # and compute costs. But SA needs fast cost updates.
    
    # Alternative representation:
    # A: set of integers
    # B: set of integers
    # Cost = #SidonViolations(A) + #SidonViolations(B) + #Intersection(Diff(A), Diff(B))
    
    # We will implement a simple SA with sets.
    # To speed up, we will cache sum counts and diff sets.
    
    def compute_costs(A, B):
        # Sidon property: all pair sums a+b (a<=b) must be unique.
        # Violation count: sum over all sums s of (count(s) - 1) if count(s) > 1?
        # Or just count of pairs that share a sum?
        # Let's use: sum_s C(count(s), 2). This counts the number of conflicting pairs.
        
        def sidon_penalty(S):
            if len(S) < 2:
                return 0
            sums = {}
            for i in range(len(S)):
                ai = S[i]
                for j in range(i, len(S)):
                    aj = S[j]
                    s = ai + aj
                    if s in sums:
                        sums[s] += 1
                    else:
                        sums[s] = 1
            penalty = 0
            for c in sums.values():
                if c > 1:
                    penalty += c * (c - 1) // 2
            return penalty
        
        def diff_set(S):
            if len(S) < 2:
                return set()
            D = set()
            for i in range(len(S)):
                ai = S[i]
                for j in range(len(S)):
                    if i != j:
                        D.add(S[j] - S[i])
            return D
        
        pen_A = sidon_penalty(A)
        pen_B = sidon_penalty(B)
        
        DA = diff_set(A)
        DB = diff_set(B)
        
        inter = len(DA & DB)
        
        return pen_A + pen_B + inter
    
    # Initialize with a random valid-ish state or empty sets
    # Let's start with the seed program's result as a warm start if possible,
    # or just random. The seed program is fast, let's run it to get a baseline.
    
    # Run seed logic to get initial A, B
    from itertools import combinations, combinations_with_replacement
    seed_sets=[[],[]]
    for x in range(1,129):
        for side in (0,1):
            candidate=seed_sets[side]+[x]
            sums=[a+b for a,b in combinations_with_replacement(candidate,2)]
            differences={b-a for a,b in combinations(candidate,2)}
            other={b-a for a,b in combinations(seed_sets[1-side],2)}
            if len(sums)==len(set(sums)) and not differences & other:
                seed_sets[side]=candidate
    
    A = set(seed_sets[0])
    B = set(seed_sets[1])
    
    current_cost = compute_costs(A, B)
    best_A = A.copy()
    best_B = B.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 1000.0
    T_min = 0.01
    alpha = 0.995
    max_iter = 50000
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor
        # Move types:
        # 1. Add element to A
        # 2. Add element to B
        # 3. Remove element from A
        # 4. Remove element from B
        # 5. Move element from A to B (if in A)
        # 6. Move element from B to A (if in B)
        # 7. Move element from A to None
        # 8. Move element from B to None
        
        r = random.random()
        
        if r < 0.3:
            # Add to A
            if len(A) < max_size:
                candidates = [x for x in range(1, n+1) if x not in A]
                if candidates:
                    x = random.choice(candidates)
                    new_A = A | {x}
                    # Quick check before full cost?
                    # Just compute cost
                    new_cost = compute_costs(new_A, B)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        A = new_A
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = A.copy()
                            best_B = B.copy()
        elif r < 0.6:
            # Add to B
            if len(B) < max_size:
                candidates = [x for x in range(1, n+1) if x not in B]
                if candidates:
                    x = random.choice(candidates)
                    new_B = B | {x}
                    new_cost = compute_costs(A, new_B)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        B = new_B
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = A.copy()
                            best_B = B.copy()
        elif r < 0.75:
            # Remove from A
            if A:
                x = random.choice(list(A))
                new_A = A - {x}
                new_cost = compute_costs(new_A, B)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                        best_B = B.copy()
        elif r < 0.9:
            # Remove from B
            if B:
                x = random.choice(list(B))
                new_B = B - {x}
                new_cost = compute_costs(A, new_B)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    B = new_B
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                        best_B = B.copy()
        else:
            # Swap/Moves
            # Try moving x from A to B
            if A:
                x = random.choice(list(A))
                # If x is in B, it's already there.
                # If we remove from A and add to B:
                new_A = A - {x}
                new_B = B | {x}
                new_cost = compute_costs(new_A, new_B)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    B = new_B
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                        best_B = B.copy()
            elif B:
                x = random.choice(list(B))
                new_B = B - {x}
                new_A = A | {x}
                new_cost = compute_costs(new_A, new_B)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = new_A
                    B = new_B
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                        best_B = B.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Return the best found
    # Ensure non-empty
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}
        
    return [list(best_A), list(best_B)]

# EVOLVE-BLOCK-END
