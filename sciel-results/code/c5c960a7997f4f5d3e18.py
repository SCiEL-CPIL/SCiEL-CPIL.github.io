import time
import random
import math

def _precompute_conflicts(N=180):
    """
    Precompute all forbidden triples (a, b, c) with distinct elements
    where 1/a = 1/b + 1/c.
    Return: list of (a, b, c) tuples, and for each element, the set of pairs it conflicts with.
    """
    triples = []
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if b == a:
                continue
            # c = a*b / (a+b), need a+b | a*b
            denom = a + b
            numer = a * b
            if numer % denom == 0:
                c = numer // denom
                if c != a and c != b and 1 <= c <= N:
                    # Store sorted triple for uniqueness
                    t = tuple(sorted([a, b, c]))
                    triples.append(t)
    
    # Remove duplicates
    triples = list(set(triples))
    
    # For each element, build conflict map: element -> set of (other1, other2) pairs
    # Actually, for checking if adding x to S creates a conflict:
    # For each triple containing x, if the other two are both in S, conflict.
    # Precompute: for each x, list of (y, z) such that (x,y,z) is a forbidden triple
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    return triples, conflict_pairs

def _is_valid(S, conflict_pairs):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for x in S:
        for (y, z) in conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
    return True

def _count_conflicts(S, conflict_pairs):
    """Count number of forbidden triples within S."""
    S_set = set(S)
    count = 0
    for x in S:
        for (y, z) in conflict_pairs[x]:
            if y in S_set and z in S_set:
                count += 1
    return count // 3  # each triple counted 3 times (once per element)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 180
    triples, conflict_pairs = _precompute_conflicts(N)
    
    # Greedy initialization: start with large numbers, try to add smaller ones
    # Strategy: process elements in descending order, add if no conflict
    S = set()
    for x in range(N, 0, -1):
        # Check if adding x creates any conflict
        can_add = True
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Simulated annealing
    if time.time() < deadline - 5:
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        current_S = set(S)
        current_conflicts = _count_conflicts(current_S, conflict_pairs)
        
        while time.time() < deadline - 5:
            if T < T_min:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element
                candidates = [x for x in range(1, N + 1) if x not in current_S]
                if not candidates:
                    break
                x = random.choice(candidates)
                # Check if adding x creates conflicts
                new_conflicts = 0
                for (y, z) in conflict_pairs[x]:
                    if y in current_S and z in current_S:
                        new_conflicts += 1
                delta_conflicts = new_conflicts
                delta_size = 1
                
                if delta_conflicts == 0 or random.random() < math.exp(-delta_conflicts / T):
                    current_S.add(x)
                    current_conflicts += delta_conflicts
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_size = len(current_S)
                        
            elif move_type < 0.7:
                # Try to remove a random element
                if not current_S:
                    break
                x = random.choice(list(current_S))
                # Count how many conflicts involve x
                old_conflicts = 0
                for (y, z) in conflict_pairs[x]:
                    if y in current_S and z in current_S:
                        old_conflicts += 1
                # Each triple with x is counted once in conflict_pairs[x]
                # But _count_conflicts divides by 3, so...
                # Actually let's just recompute for simplicity if set is small
                delta_conflicts = -old_conflicts
                delta_size = -1
                
                if delta_conflicts >= 0 or random.random() < math.exp(-delta_conflicts / T):
                    current_S.remove(x)
                    current_conflicts += delta_conflicts
                    
            else:
                # Swap: remove one, add another
                if len(current_S) < 2:
                    continue
                x = random.choice(list(current_S))
                candidates = [y for y in range(1, N + 1) if y not in current_S]
                if not candidates:
                    continue
                z = random.choice(candidates)
                
                # Compute change in conflicts
                # Remove x: removes conflicts involving x
                remove_conflicts = 0
                for (y, w) in conflict_pairs[x]:
                    if y in current_S and w in current_S and y != z and w != z:
                        remove_conflicts += 1
                    elif (y == z or w == z) and (y in current_S or w in current_S):
                        # This is tricky - if z is being added, these might become new conflicts
                        pass
                
                # This is getting complex. Let's just do add/remove separately.
                # Try add z
                add_conflicts = 0
                for (y, w) in conflict_pairs[z]:
                    if y in current_S and w in current_S:
                        add_conflicts += 1
                
                # Try remove x  
                rem_conflicts = 0
                for (y, w) in conflict_pairs[x]:
                    if y in current_S and w in current_S:
                        rem_conflicts += 1
                
                delta_conflicts = add_conflicts - rem_conflicts
                delta_size = 0  # swap keeps size same
                
                if delta_conflicts <= 0 or random.random() < math.exp(-delta_conflicts / T):
                    current_S.remove(x)
                    current_S.add(z)
                    current_conflicts += delta_conflicts
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_size = len(current_S)
            
            T *= cooling
        
        # Use best found
        if len(best_S) > len(S):
            S = best_S
    
    # Local search polish: try to add more elements
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            can_add = True
            for (y, z) in conflict_pairs[x]:
                if y in S and z in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                changed = True
    
    return sorted(S)
