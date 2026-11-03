import random
import math

def construct(q, n, radius):
    # Specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Simple Greedy
        total_words = q ** n
        neighbors = [[] for _ in range(total_words)]
        for w in range(total_words):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            # Reconstruct logic for neighbors
            nbrs = [w]
            for pos in range(n):
                div = q ** (n - 1 - pos)
                curr_digit = (w // div) % q
                for new_digit in range(q):
                    if new_digit != curr_digit:
                        new_w = w + (new_digit - curr_digit) * div
                        nbrs.append(new_w)
            neighbors[w] = nbrs
        
        covered = [False] * total_words
        code = []
        while any(not c for c in covered):
            best_count = -1
            best_candidates = []
            for w in range(total_words):
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            if best_count == 0: break
            selected = random.choice(best_candidates)
            code.append(selected)
            for nbr in neighbors[selected]:
                covered[nbr] = True
        return [[(w // q**(n-1-i)) % q for i in range(n)] for w in code]

    total_words = 729
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr_digit = (w // div) % 3
            for new_digit in range(3):
                if new_digit != curr_digit:
                    new_w = w + (new_digit - curr_digit) * div
                    nbrs.append(new_w)
        neighbors[w] = nbrs

    # 1. Greedy Initialization
    covered = [False] * total_words
    init_code = []
    while any(not c for c in covered):
        best_count = -1
        best_candidates = []
        for w in range(total_words):
            count = 0
            for nbr in neighbors[w]:
                if not covered[nbr]:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidates = [w]
            elif count == best_count and count > 0:
                best_candidates.append(w)
        if best_count == 0: break
        selected = random.Random(42 + len(init_code)).choice(best_candidates) # Deterministic-ish start? 
        # Actually, let's use a fixed seed for the greedy part to be reproducible
        # But for SA, we need randomness. Let's use a global RNG for SA.
        init_code.append(selected)
        for nbr in neighbors[selected]:
            covered[nbr] = True

    # 2. Simulated Annealing
    # State: List of centers
    # We maintain cover_count array for efficiency
    
    best_code = init_code[:]
    best_size = len(best_code)
    
    # Initialize SA state
    current_code = init_code[:]
    cover_count = [0] * total_words
    for c in current_code:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    # Cost function: size * 1000 + uncovered * 100
    def get_cost(code_size, uncov):
        return code_size * 1000 + uncov * 100
        
    current_cost = get_cost(len(current_code), uncovered_count)
    
    rng = random.Random(12345)
    
    # SA Parameters
    T_start = 100.0
    T_end = 1.0
    iterations = 50000 # Adjust based on time limit
    
    # Precompute random candidates? No, just pick random w in 0..728
    
    for i in range(iterations):
        # Cooling
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Generate Move
        move_type = rng.random()
        if move_type < 0.3:
            # Remove
            if not current_code: continue
            idx = rng.randrange(len(current_code))
            c = current_code[idx]
            # Apply removal
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
                if cover_count[nbr] == 0:
                    uncovered_count += 1
            current_code.pop(idx)
            new_cost = get_cost(len(current_code), uncovered_count)
            
            # Check acceptance
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                if len(current_code) < best_size and uncovered_count == 0:
                    best_size = len(current_code)
                    best_code = current_code[:]
            else:
                # Revert
                current_code.insert(idx, c)
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                    if cover_count[nbr] == 1: # Was 0, now 1
                        uncovered_count -= 1
                        # Wait, if cover_count was 0, it became 1, so uncovered decreases.
                        # In removal, if it became 0, uncovered increased.
                        # So in revert, if it becomes 1, uncovered decreases.
                        pass # Logic handled above? 
                        # Let's trace:
                        # Removal: count 1->0 (uncov +1), count 2->1 (uncov 0)
                        # Revert: count 0->1 (uncov -1), count 1->2 (uncov 0)
                        # My revert logic: if cover_count[nbr] == 1, it means it was 0 before increment. So uncov -= 1.
                        # Correct.
                        
        elif move_type < 0.7:
            # Add
            c = rng.randrange(total_words)
            # Apply add
            for nbr in neighbors[c]:
                if cover_count[nbr] == 0:
                    uncovered_count -= 1
                cover_count[nbr] += 1
            current_code.append(c)
            new_cost = get_cost(len(current_code), uncovered_count)
            
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                if len(current_code) < best_size and uncovered_count == 0:
                    best_size = len(current_code)
                    best_code = current_code[:]
            else:
                # Revert
                current_code.pop()
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                    if cover_count[nbr] == 0:
                        uncovered_count += 1
                        
        else:
            # Swap
            if len(current_code) < 2: continue
            idx1 = rng.randrange(len(current_code))
            c1 = current_code[idx1]
            c2 = rng.randrange(total_words)
            if c1 == c2: continue
            
            # Apply swap: Remove c1, Add c2
            # Remove c1
            for nbr in neighbors[c1]:
                cover_count[nbr] -= 1
                if cover_count[nbr] == 0:
                    uncovered_count += 1
            current_code[idx1] = c2
            # Add c2
            for nbr in neighbors[c2]:
                if cover_count[nbr] == 0:
                    uncovered_count -= 1
                cover_count[nbr] += 1
            
            new_cost = get_cost(len(current_code), uncovered_count)
            # Size is same, so cost change is only due to uncovered change
            
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                if len(current_code) < best_size and uncovered_count == 0:
                    best_size = len(current_code)
                    best_code = current_code[:]
            else:
                # Revert
                # Remove c2
                for nbr in neighbors[c2]:
                    cover_count[nbr] -= 1
                    if cover_count[nbr] == 0:
                        uncovered_count += 1
                current_code[idx1] = c1
                # Add c1
                for nbr in neighbors[c1]:
                    if cover_count[nbr] == 0:
                        uncovered_count -= 1
                    cover_count[nbr] += 1

    # Convert best_code to list of lists
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
