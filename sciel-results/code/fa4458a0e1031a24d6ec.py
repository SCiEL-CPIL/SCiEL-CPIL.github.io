import random
import math
import time

def construct():
    max_val = 10000
    n = 20
    
    # Generate candidate pool: 7-smooth numbers + small integers
    # 7-smooth numbers are good for product collisions
    # Small integers are good for sum collisions
    candidates = set()
    # Add small integers
    for i in range(1, 101):
        candidates.add(i)
    # Add 7-smooth numbers
    # 2^a * 3^b * 5^c * 7^d <= 10000
    for a in range(14):
        p2 = 2 ** a
        if p2 > max_val: break
        for b in range(9):
            p3 = 3 ** b
            if p2 * p3 > max_val: break
            for c in range(7):
                p5 = 5 ** c
                if p2 * p3 * p5 > max_val: break
                for d in range(5):
                    p7 = 7 ** d
                    val = p2 * p3 * p5 * p7
                    if val <= max_val:
                        candidates.add(val)
    
    candidate_list = sorted(list(candidates))
    
    def evaluate(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                sum_set.add(ai + A[j])
                prod_set.add(ai * A[j])
        return max(len(sum_set), len(prod_set))
        
    def hill_climb(A, max_time=5.0):
        A = A[:]
        current_score = evaluate(A)
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            improved = False
            # Try swapping any two elements
            # To speed up, we can try random swaps first, then exhaustive if time permits
            # With n=20, there are 190 pairs.
            # Let's try random swaps to avoid getting stuck in local minima immediately
            # Or try all swaps? 190 * 20 (evaluate) = 3800 evals.
            # Evaluate is O(N^2) = 400 ops. 3800 * 400 = 1.5M ops. Very fast.
            # So we can do a full pass of all swaps.
            
            found_improvement = False
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i] == A[j]: continue
                    # Swap A[i] and A[j]
                    # Actually, swapping doesn't change the set, so it doesn't change the sum/product sets.
                    # Wait, the set A is a set of values. Swapping positions in the list doesn't change the set.
                    # So "swap" is not a valid move in terms of changing the set content.
                    # The move must be **replace** one element with another from the pool.
                    pass
            
            # Correct move: Replace A[i] with a new value from candidate_list
            # There are n * |candidate_list| moves.
            # |candidate_list| is approx 100 (small) + ~500 (smooth) = 600.
            # 20 * 600 = 12000 moves.
            # 12000 * 400 ops = 4.8M ops per iteration.
            # This might be a bit slow if we do many iterations, but 60s limit is generous.
            # Let's limit the candidate pool for the search to a smaller subset or use random sampling.
            
            # Let's use random sampling of moves for speed, but more moves per step.
            # Or, we can pre-filter candidates.
            
            # Let's try a simpler neighborhood: Replace A[i] with a random candidate.
            # If improvement, accept.
            # If no improvement after K tries, break or perturb.
            
            no_improve_count = 0
            max_no_improve = 1000
            
            while no_improve_count < max_no_improve and time.time() - start_time < max_time:
                idx = random.randint(0, n - 1)
                old_val = A[idx]
                new_val = random.choice(candidate_list)
                
                if new_val == old_val or new_val in A:
                    no_improve_count += 1
                    continue
                
                # Try the move
                A[idx] = new_val
                new_score = evaluate(A)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                else:
                    A[idx] = old_val
                    no_improve_count += 1
                    
        return A, current_score

    best_A = None
    best_score = float('inf')
    
    start_time = time.time()
    timeout = 55
    
    # Strategy: Multiple restarts with different initializations
    # 1. Arithmetic Progression 1..20
    # 2. Geometric Progression (powers of 2, capped)
    # 3. Random 7-smooth
    # 4. Random Small Ints
    
    inits = []
    
    # Init 1: 1..20
    inits.append(list(range(1, 21)))
    
    # Init 2: Powers of 2 (1, 2, 4, ..., 8192) + fill with small distinct ints
    geo = [2**i for i in range(14)] # 1 to 8192
    # Need 6 more numbers, distinct from geo, <= 10000
    # Pick small numbers not in geo
    extra = []
    for i in range(1, 100):
        if i not in geo and len(extra) < 6:
            extra.append(i)
    inits.append(geo + extra)
    
    # Init 3: Random 7-smooth
    smooths = [c for c in candidate_list if c > 1] # exclude 1? No, 1 is useful.
    random.shuffle(smooths)
    # Ensure distinct
    set_init = set()
    for c in smooths:
        if len(set_init) >= 20: break
        set_init.add(c)
    if len(set_init) < 20:
        # Fill with small ints
        for i in range(1, 100):
            if len(set_init) >= 20: break
            set_init.add(i)
    inits.append(list(set_init))
    
    # Init 4: Random small ints
    random.shuffle(candidate_list)
    set_init = set()
    for c in candidate_list:
        if len(set_init) >= 20: break
        set_init.add(c)
    inits.append(list(set_init))
    
    # Run ILS on each init
    for init_A in inits:
        if time.time() - start_time > timeout:
            break
        
        # Allocate remaining time / remaining inits
        remaining_inits = 4 - inits.index(init_A)
        time_per_init = (timeout - (time.time() - start_time)) / remaining_inits
        
        # Run Hill Climbing with time limit
        A, score = hill_climb(init_A, max_time=time_per_init * 0.8)
        
        if score < best_score:
            best_score = score
            best_A = A[:]
            
        # Perturb and restart Hill Climbing (Iterated Local Search)
        # Do this a few times if time permits
        # Let's just do one restart per init for now to keep it simple and fast
        if time.time() - start_time < timeout - 1:
            # Perturb: change 2 random elements
            perturbed_A = A[:]
            for _ in range(2):
                idx = random.randint(0, n-1)
                new_val = random.choice(candidate_list)
                if new_val not in perturbed_A:
                    perturbed_A[idx] = new_val
            
            A2, score2 = hill_climb(perturbed_A, max_time=time_per_init * 0.2)
            if score2 < best_score:
                best_score = score2
                best_A = A2[:]

    return best_A
