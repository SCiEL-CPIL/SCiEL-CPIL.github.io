import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline
    
    # Generate 3-smooth numbers <= 10000
    smooth = set()
    i = 0
    while 2**i <= 10000:
        j = 0
        while (2**i) * (3**j) <= 10000:
            smooth.add((2**i) * (3**j))
            j += 1
        i += 1
    
    smooth_list = sorted(list(smooth))
    
    # Ensure we have at least 20
    if len(smooth_list) < 20:
        # Fallback to range(1, 21) if not enough smooth numbers (unlikely)
        return list(range(1, 21))
    
    # Initial guess: smallest 20
    A = smooth_list[:20]
    
    def get_score(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        for x in A_set:
            for y in A_set:
                sum_set.add(x + y)
                prod_set.add(x * y)
        return max(len(sum_set), len(prod_set))
    
    best_score = get_score(A)
    best_A = list(A)
    current_A = set(best_A)
    current_score = best_score
    
    candidates = smooth_list
    
    # Local search
    while time.time() < deadline:
        improved = False
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        # Try swapping each x in A with each y in candidates
        for x in list(current_A):
            for y in candidates:
                if y in current_A:
                    continue
                
                # Calculate new score
                # Remove x, add y
                new_A = list(current_A)
                new_A.remove(x)
                new_A.append(y)
                
                new_score = get_score(new_A)
                
                if new_score < current_score:
                    current_A = set(new_A)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(current_A)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Perturbation: swap 2 random elements
            items = list(current_A)
            if len(items) >= 2:
                i, j = random.sample(range(len(items)), 2)
                x = items[i]
                y_candidates = [c for c in candidates if c not in current_A]
                if y_candidates:
                    y_new = random.choice(y_candidates)
                    new_A = list(current_A)
                    new_A.remove(x)
                    new_A.append(y_new)
                    new_score = get_score(new_A)
                    if new_score < current_score:
                        current_A = set(new_A)
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_A = list(current_A)

    return best_A
