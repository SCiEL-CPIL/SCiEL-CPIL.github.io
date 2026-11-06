"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    N = 512
    # Precompute score contribution
    score_contrib = [0.0] * N
    for a in range(1, N):
        score_contrib[a] = 1.0 / (N - a)
    
    def is_coprime_to_all(a, chosen_set, chosen_list):
        for b in chosen_list:
            if gcd(a, b) != 1:
                return False
        return True
    
    # Phase 1: Greedy initialization - pick from high to low
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    def current_score():
        return sum(score_contrib[a] for a in chosen)
    
    best_score = current_score()
    best_chosen = chosen[:]
    
    # Phase 2: Local search - try to improve via swaps and additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any unchosen element
        for a in range(511, 0, -1):
            if a not in chosen_set:
                if is_coprime_to_all(a, chosen_set, chosen):
                    new_score = current_score() + score_contrib[a]
                    if new_score > best_score + 1e-15:
                        chosen.append(a)
                        chosen_set.add(a)
                        best_score = new_score
                        best_chosen = chosen[:]
                        improved = True
                    break  # Only try to add one at a time, re-evaluate
        
        # Try swaps: replace one chosen element with one unchosen element
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            old_a = chosen[i]
            for new_a in range(511, 0, -1):
                if new_a in chosen_set:
                    continue
                # Check if new_a is coprime to all other chosen elements
                coprime = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    if gcd(new_a, chosen[j]) != 1:
                        coprime = False
                        break
                if not coprime:
                    continue
                delta = score_contrib[new_a] - score_contrib[old_a]
                if delta > 1e-15:
                    # Perform swap
                    chosen[i] = new_a
                    chosen_set.remove(old_a)
                    chosen_set.add(new_a)
                    best_score += delta
                    best_chosen = chosen[:]
                    improved = True
                    break  # Restart search after a swap
    
    # Return the best solution found
    return best_chosen
# EVOLVE-BLOCK-END
