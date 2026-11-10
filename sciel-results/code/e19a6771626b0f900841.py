import time
import random
import numpy as np

def construct():
    """
    Construct a large B2[2] subset of {1, ..., 1000}.
    Strategy:
    1. Generate several diverse greedy seeds (ascending, descending, weighted, sparse).
    2. Perform local search (insertions and swaps) to escape local optima.
    3. Use a deadline guard to ensure completion within 120s.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 1000
    best_set = set()
    best_size = 0
    
    # Helper: Check if adding x to a set S (with counts) is valid
    def can_add(x, S, counts):
        # We need to check if adding x creates any sum > 2
        # Sums involving x: x + y for y in S, y != x (but we treat a<=b, so if x in S, 2x is involved)
        # Actually, the representation count for a sum s is the number of pairs (a,b) in S with a<=b and a+b=s.
        # When we add x, the new pairs are (x, y) for y in S where x <= y, and (x, x).
        # Wait, the problem says "a <= b from the set".
        # If we add x, the new representations are:
        # 1. (x, x) -> sum 2x
        # 2. (x, y) for y in S such that x < y -> sum x+y
        # Note: if x is already in S, we can't add it.
        
        # For each new sum s created by adding x:
        # The new count for s is counts.get(s, 0) + (1 if s == 2*x else 0) + (1 if there exists y in S, y > x, x+y==s else 0)
        # Actually, it's simpler:
        # New pairs involving x:
        # - (x, x)
        # - (x, y) for y in S, y > x
        # Let's check each potential sum.
        
        # Sum 2*x:
        # Current count for 2*x is counts.get(2*x, 0).
        # New pair (x,x) adds 1.
        # Are there other pairs? No, because x is new.
        # So new_count(2x) = counts.get(2x, 0) + 1.
        # Must be <= 2.
        
        # Sum x + y for y in S, y > x:
        # New pair (x, y) adds 1 to count of x+y.
        # So new_count(x+y) = counts.get(x+y, 0) + 1.
        # Must be <= 2.
        
        # Optimization:
        # Check 2*x first.
        if counts.get(2*x, 0) + 1 > 2:
            return False
        # Check x+y for y in S, y > x
        # We can iterate through S.
        for y in S:
            if y > x:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    return False
        return True
    
    def update_counts(S, counts, x):
        # Add x to counts
        new_counts = dict(counts)
        # Update for 2*x
        new_counts[2*x] = new_counts.get(2*x, 0) + 1
        # Update for x+y
        for y in S:
            if y > x:
                s = x + y
                new_counts[s] = new_counts.get(s, 0) + 1
        return new_counts
    
    def remove_from_counts(S, counts, x):
        # Remove x from counts
        new_counts = dict(counts)
        # Remove 2*x
        new_counts[2*x] -= 1
        if new_counts[2*x] == 0:
            del new_counts[2*x]
        # Remove x+y
        for y in S:
            if y > x:
                s = x + y
                new_counts[s] -= 1
                if new_counts[s] == 0:
                    del new_counts[s]
        return new_counts
    
    def greedy_seed(ordering):
        S = set()
        counts = {}
        for x in ordering:
            if can_add(x, S, counts):
                S.add(x)
                counts = update_counts(S, counts, x)
        return S, counts
    
    def local_search(S, counts, max_iters=1000):
        S = set(S)
        counts = dict(counts)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try insertions
            # Candidate insertion points: 1..1000 not in S
            # To speed up, we can check all, but that's 1000 checks.
            # Let's shuffle candidates to avoid bias.
            candidates = [x for x in range(1, 1001) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S, counts):
                    S.add(x)
                    counts = update_counts(S, counts, x)
                    improved = True
                    break
            # If no insertion found, try swaps
            if not improved:
                # Try swapping x in S with y not in S
                # Randomly pick x from S
                list_S = list(S)
                random.shuffle(list_S)
                for x in list_S:
                    # Try to find y not in S such that (S - {x}) U {y} is valid
                    # This is expensive. Let's limit swaps.
                    # Heuristic: try y near x or random y
                    # For simplicity, try a few random y's
                    for _ in range(10):
                        y = random.randint(1, 1000)
                        if y in S:
                            continue
                        # Check if removing x and adding y is valid
                        # 1. Remove x from counts
                        temp_counts = remove_from_counts(S, counts, x)
                        # 2. Check if y can be added to S - {x}
                        S_temp = S - {x}
                        if can_add(y, S_temp, temp_counts):
                            S = S_temp
                            S.add(y)
                            counts = update_counts(S, temp_counts, y)
                            improved = True
                            break
                    if improved:
                        break
        return S, counts

    # 1. Generate diverse seeds
    seeds = []
    
    # Ascending
    S1, C1 = greedy_seed(range(1, 1001))
    seeds.append((S1, C1))
    
    # Descending
    S2, C2 = greedy_seed(range(1000, 0, -1))
    seeds.append((S2, C2))
    
    # Weighted (prioritize middle?)
    # Let's try a few random orderings
    for i in range(5):
        order = list(range(1, 1001))
        random.shuffle(order)
        S, C = greedy_seed(order)
        seeds.append((S, C))
    
    # Sparse (step 2, step 3, etc.)
    for step in [2, 3, 4]:
        order = list(range(1, 1001, step))
        S, C = greedy_seed(order)
        seeds.append((S, C))
    
    # 2. Local search on each seed
    for S, C in seeds:
        if time.time() > deadline - 20:
            break
        S_ls, C_ls = local_search(S, C, max_iters=500)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = S_ls
    
    # 3. Final polish: try more insertions on the best set
    if time.time() < deadline - 5:
        S_final, C_final = local_search(best_set, dict(C_ls), max_iters=1000)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = S_final

    return list(best_set)

# Ensure random seed for reproducibility if needed, but random is fine for search
random.seed(42)
