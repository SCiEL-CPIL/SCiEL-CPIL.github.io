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
    
    def greedy_large_first():
        selected = []
        for x in range(n, 0, -1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected
    
    def greedy_random():
        selected = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected
    
    def local_search(selected):
        selected = list(selected)
        for _ in range(20):  # Limited iterations
            improved = False
            # Try adding
            for x in range(1, n + 1):
                if x not in selected and len(selected) < max_elements:
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
                for x in range(1, n + 1):
                    if x not in temp:
                        if is_valid_addition(temp, x):
                            new_set = temp + [x]
                            if len(new_set) > len(selected):
                                selected = new_set
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        return selected
    
    best = greedy_large_first()
    
    for trial in range(30):
        random.seed(1000 + trial)
        s = greedy_random()
        s = local_search(s)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
    
    if not is_valid(best):
        best = greedy_large_first()
    
    return best[:max_elements]
