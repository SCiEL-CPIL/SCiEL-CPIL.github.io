"""Rewritten program: greedy + local search for large reciprocal-Schur-free subset of [1,180]."""
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a,b,c) with distinct elements where 1/a = 1/b + 1/c
    # a = bc/(b+c), a is always the minimum
    forbidden = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    forbidden.add(triple)
    
    forbidden_list = list(forbidden)
    num_forbidden = len(forbidden_list)
    
    # Precompute: for each element, list of (other1, other2) pairs that form a forbidden triple
    # i.e., if element x is in S, then (y,z) is a conflict pair if {x,y,z} is forbidden
    conflict_pairs = {i: [] for i in range(1, N+1)}
    for triple in forbidden_list:
        elems = list(triple)
        for i in range(3):
            x = elems[i]
            y = elems[(i+1) % 3]
            z = elems[(i+2) % 3]
            conflict_pairs[x].append((y, z))
    
    # Also precompute: for each pair (y,z), which x values form a forbidden triple
    # This helps for checking if adding x creates a conflict
    pair_to_third = {}
    for triple in forbidden_list:
        elems = list(triple)
        for i in range(3):
            x = elems[i]
            y = elems[(i+1) % 3]
            z = elems[(i+2) % 3]
            pair_key = (min(y,z), max(y,z))
            if pair_key not in pair_to_third:
                pair_to_third[pair_key] = []
            pair_to_third[pair_key].append(x)
    
    def check_valid(S):
        """Check if set S has no forbidden triple using conflict pairs."""
        S_set = set(S)
        for x in S:
            for (y, z) in conflict_pairs[x]:
                if y in S_set and z in S_set:
                    return False
        return True
    
    def count_conflicts(elem, S):
        """Count how many forbidden triples would be created if elem is added to S."""
        S_set = set(S)
        count = 0
        for (y, z) in conflict_pairs[elem]:
            if y in S_set and z in S_set:
                count += 1
        return count
    
    def can_add(elem, S):
        """Check if elem can be added to S without creating a forbidden triple."""
        S_set = set(S)
        for (y, z) in conflict_pairs[elem]:
            if y in S_set and z in S_set:
                return False
        return True
    
    # Greedy construction: try multiple orderings, keep best
    best_S = set()
    best_size = 0
    
    # Ordering 1: descending (large numbers first - they appear in fewer triples as the large element)
    for order_seed in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if order_seed == 0:
            order = list(range(N, 0, -1))
        elif order_seed == 1:
            order = list(range(1, N+1))
        elif order_seed == 2:
            order = list(range(1, N+1))
            random.seed(order_seed * 1000)
            random.shuffle(order)
        elif order_seed == 3:
            # Sort by degree (number of forbidden triples containing element)
            degree = {i: len(conflict_pairs[i]) for i in range(1, N+1)}
            order = sorted(range(1, N+1), key=lambda x: degree[x])
        else:
            # Sort by degree descending
            degree = {i: len(conflict_pairs[i]) for i in range(1, N+1)}
            order = sorted(range(1, N+1), key=lambda x: -degree[x])
        
        S = set()
        for elem in order:
            if time.time() - start_time > DEADLINE - 30:
                break
            if can_add(elem, S):
                S.add(elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    S = best_S.copy()
    removed = set(range(1, N+1)) - S
    
    # Local search with simulated annealing style
    # Move types: add an element, remove an element, swap
    temperature = 10.0
    cooling = 0.995
    best_S_local = S.copy()
    best_size_local = len(S)
    
    current_size = len(S)
    
    max_steps = 50000
    step = 0
    while step < max_steps and (time.time() - start_time) < DEADLINE - 10:
        step += 1
        if step % 1000 == 0:
            temperature *= cooling
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and removed:
            # Try to add a random removed element
            elem = random.choice(list(removed))
            if can_add(elem, S):
                S.add(elem)
                removed.discard(elem)
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_S_local = S.copy()
            else:
                # Don't accept (would create conflict)
                pass
        elif move_type < 0.7 and S:
            # Try to remove a random element and add another
            if removed:
                remove_elem = random.choice(list(S))
                add_elem = random.choice(list(removed))
                if remove_elem == add_elem:
                    continue
                # Check if swap is valid
                S_test = (S - {remove_elem}) | {add_elem}
                if check_valid(S_test):
                    S = S_test
                    removed.discard(add_elem)
                    removed.add(remove_elem)
                    # current_size unchanged
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_S_local = S.copy()
                else:
                    # Accept with probability based on temperature (shouldn't happen since size same)
                    pass
            else:
                # Nothing to add, just remove
                remove_elem = random.choice(list(S))
                S.discard(remove_elem)
                removed.add(remove_elem)
                current_size -= 1
        else:
            # Try to add multiple elements
            if removed:
                # Try adding up to 3 random removed elements
                candidates = random.sample(list(removed), min(3, len(removed)))
                added_any = False
                for elem in candidates:
                    if can_add(elem, S):
                        S.add(elem)
                        removed.discard(elem)
                        current_size += 1
                        added_any = True
                if added_any and current_size > best_size_local:
                    best_size_local = current_size
                    best_S_local = S.copy()
    
    S = best_S_local.copy()
    removed = set(range(1, N+1)) - S
    
    # Final polish: try to add any remaining elements
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        for elem in list(removed):
            if can_add(elem, S):
                S.add(elem)
                removed.discard(elem)
                improved = True
    
    # Try swaps one more time
    if (time.time() - start_time) < DEADLINE - 5:
        for _ in range(3):
            improved = False
            for add_elem in list(removed):
                if time.time() - start_time > DEADLINE - 5:
                    break
                for remove_elem in list(S):
                    S_test = (S - {remove_elem}) | {add_elem}
                    if check_valid(S_test):
                        S = S_test
                        removed.discard(add_elem)
                        removed.add(remove_elem)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    result = sorted(S)
    if not result:
        result = [91]  # fallback
    
    # Final validation
    result_set = set(result)
    for t in forbidden:
        if all(x in result_set for x in t):
            for x in t:
                if x in result_set:
                    result_set.discard(x)
                    break
            result = sorted(result_set)
            break
    
    return result
