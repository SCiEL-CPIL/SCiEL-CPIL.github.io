import random
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    
    def is_valid_addition(selected, candidate):
        """Check if adding candidate to selected maintains validity."""
        sel_set = set(selected)
        # For the new candidate: check if it divides sum of any 3 others in selected
        if len(selected) >= 3:
            for a, b, c in combinations(selected, 3):
                if (a + b + c) % candidate == 0:
                    return False
        # For each existing element a: check triples that include candidate
        # i.e., a | (b + c + candidate) for b,c in selected, b!=a, c!=a, b!=c
        for a in selected:
            others = [x for x in selected if x != a]
            if len(others) >= 2:
                for b, c in combinations(others, 2):
                    if (b + c + candidate) % a == 0:
                        return False
        return True

    def is_valid(selected):
        """Check if the entire set is valid."""
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def greedy_top_k_random(k=10):
        selected = []
        candidates = list(range(n, 0, -1))
        while len(selected) < max_elements and candidates:
            # Find valid candidates in the top k of the remaining sorted list
            # We maintain candidates sorted descending
            valid_candidates = []
            for i in range(min(k, len(candidates))):
                x = candidates[i]
                if is_valid_addition(selected, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                # If no valid in top k, try the rest (or break if none)
                # To be safe, scan all remaining
                for x in candidates[k:]:
                    if is_valid_addition(selected, x):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= k:
                            break
                
                if not valid_candidates:
                    break

            # Pick randomly from valid candidates
            x = random.choice(valid_candidates)
            selected.append(x)
            candidates.remove(x)
            
        return selected

    def local_search(selected):
        selected = list(selected)
        # Try to improve by swapping
        for _ in range(50):  # More iterations
            improved = False
            # Try adding
            if len(selected) < max_elements:
                for x in range(1, n + 1):
                    if x not in selected:
                        if is_valid_addition(selected, x):
                            selected.append(x)
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try single swap
            for i in range(len(selected)):
                removed = selected[i]
                temp = selected[:i] + selected[i+1:]
                # Try to replace removed with something better/larger
                # Sort candidates descending to prefer large
                for x in range(n, 0, -1):
                    if x not in temp:
                        if is_valid_addition(temp, x):
                            new_set = temp + [x]
                            # Accept if it allows future growth? 
                            # For now, just accept any valid swap to explore
                            # But we want to maximize size. If size is same, 
                            # we prefer larger numbers? 
                            # Actually, if we are stuck at size K, any swap 
                            # that keeps size K but allows further additions is good.
                            # We don't know if it allows further additions.
                            # Heuristic: Accept swap if x > removed?
                            if x > removed:
                                selected = new_set
                                improved = True
                                break
                if improved:
                    break
            
            # Try 1-for-2 swap (remove 1, add 2)
            if not improved and len(selected) < max_elements - 1:
                for i in range(len(selected)):
                    removed = selected[i]
                    temp = selected[:i] + selected[i+1:]
                    added_count = 0
                    new_set = list(temp)
                    for x in range(n, 0, -1):
                        if x not in new_set and added_count < 2:
                            if is_valid_addition(new_set, x):
                                new_set.append(x)
                                added_count += 1
                    if added_count == 2 and len(new_set) > len(selected):
                        selected = new_set
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break
        return selected

    best = []
    
    # Run multiple trials with different random seeds and k values
    for trial in range(50):
        random.seed(2000 + trial)
        k = 5 + (trial % 10) # Vary k
        s = greedy_top_k_random(k)
        s = local_search(s)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
    
    # Fallback to large first if best is empty or small
    if len(best) < 40:
        selected = []
        for x in range(n, 0, -1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        if len(selected) > len(best):
            best = selected
            best = local_search(best)

    if not is_valid(best):
        # Emergency fallback
        best = []
        for x in range(n, 0, -1):
            if is_valid_addition(best, x):
                best.append(x)
                if len(best) >= max_elements:
                    break

    return best[:max_elements]
