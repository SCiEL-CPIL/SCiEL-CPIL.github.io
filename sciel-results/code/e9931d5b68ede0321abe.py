import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 128
    target_low = 128
    target_high = 256
    
    # Initial set: [64, 128]
    A = set(range(64, 129))
    
    def compute_reps(A_set):
        reps = np.zeros(257, dtype=int)
        A_list = list(A_set)
        L = len(A_list)
        for i in range(L):
            a = A_list[i]
            for j in range(i, L):
                b = A_list[j]
                s = a + b
                if target_low <= s <= target_high:
                    reps[s] += 1
        return reps

    def is_valid(reps):
        # Check if all targets are covered
        # Using numpy slice for speed
        return np.all(reps[target_low:target_high+1] > 0)

    def get_score(reps):
        if not is_valid(reps):
            return float('inf')
        return int(np.max(reps[target_low:target_high+1]))

    current_reps = compute_reps(A)
    current_score = get_score(current_reps)
    best_A = set(A)
    best_score = current_score
    
    # Greedy removal and random swap
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations % 100 == 0 and time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try removing elements
        # Prioritize elements that contribute to high multiplicity sums
        # Simple approach: try all, pick best improvement
        best_rem_score = current_score
        best_rem_A = None
        best_rem_reps = None
        
        for a in list(A):
            if time.time() > deadline:
                break
            new_A = set(A)
            new_A.remove(a)
            new_reps = compute_reps(new_A)
            new_score = get_score(new_reps)
            
            if new_score < best_rem_score:
                best_rem_score = new_score
                best_rem_A = new_A
                best_rem_reps = new_reps
        
        if best_rem_A is not None:
            A = best_rem_A
            current_reps = best_rem_reps
            current_score = best_rem_score
            if current_score < best_score:
                best_score = current_score
                best_A = set(A)
            continue # Restart loop with new state
        
        # 2. If no removal works, try random swaps
        # Pick a random element to swap
        if not A:
            break
            
        a = random.choice(list(A))
        swap_improved = False
        
        for _ in range(20): # Try 20 random swaps per stuck point
            if time.time() > deadline:
                break
            new_val = random.randint(0, 128)
            if new_val in A:
                continue
            new_A = set(A)
            new_A.remove(a)
            new_A.add(new_val)
            new_reps = compute_reps(new_A)
            new_score = get_score(new_reps)
            
            if new_score < current_score:
                A = new_A
                current_reps = new_reps
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(A)
                swap_improved = True
                break
        
        if not swap_improved and not best_rem_A:
            # Stuck in local minimum? 
            # Could try adding an element if score is inf, but we start valid.
            # If valid but stuck, maybe try a larger random restart?
            # For now, just break or continue with more randomization.
            # Let's try a few more random swaps with different 'a'
            found_any = False
            for _ in range(5):
                if time.time() > deadline:
                    break
                a = random.choice(list(A))
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(0, 128)
                    if new_val in A:
                        continue
                    new_A = set(A)
                    new_A.remove(a)
                    new_A.add(new_val)
                    new_reps = compute_reps(new_A)
                    new_score = get_score(new_reps)
                    if new_score < current_score:
                        A = new_A
                        current_reps = new_reps
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_A = set(A)
                        found_any = True
                        break
                if found_any:
                    break
            if not found_any:
                # If we are stuck for a while, maybe restart with a slightly perturbed seed?
                # But we must maintain validity.
                # Let's just exit the loop to avoid wasting time.
                # Or we can try to add an element to reduce multiplicity? 
                # Adding an element increases multiplicity for sums it covers.
                # It might allow removal of other elements later.
                # Let's try adding a random element if size < 129
                if len(A) < 129:
                    new_val = random.randint(0, 128)
                    if new_val not in A:
                        new_A = set(A)
                        new_A.add(new_val)
                        new_reps = compute_reps(new_A)
                        # Only accept if it doesn't break validity (it won't) 
                        # and maybe we hope it enables future removals.
                        # But immediate score will be worse or equal.
                        # So we don't accept it immediately.
                        # We can store it as a "candidate" but that's complex.
                        # Let's just break.
                break

    return list(best_A)
