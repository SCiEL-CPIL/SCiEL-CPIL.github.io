import time
import random
import sys

def _check_can_add(x, selected):
    """
    Check if x can be added to selected without violating the condition.
    Condition: No element divides the sum of any 3 distinct OTHER selected elements.
    
    When adding x to selected (S), we need to check:
    1. x does not divide sum of any 3 elements in S.
    2. For each a in S, a does not divide sum of x and any 2 other elements in S \ {a}.
    """
    n = len(selected)
    if n < 2:
        return True
    
    # 1. Check if x divides sum of any 3 elements in S
    if n >= 3:
        for i in range(n - 2):
            si = selected[i]
            for j in range(i + 1, n - 1):
                sij = si + selected[j]
                for k in range(j + 1, n):
                    if (sij + selected[k]) % x == 0:
                        return False

    # 2. Check if any a in S divides sum of x and 2 other elements in S
    # For a fixed a at index i, we need to pick 2 other indices j, k from S \ {i}.
    if n >= 2:
        for i, a in enumerate(selected):
            # We need to pick 2 elements from S excluding index i.
            # Iterate j, k over all indices, skipping i.
            # j < k to avoid duplicates and ensure distinctness.
            for j in range(n):
                if j == i:
                    continue
                sj = selected[j]
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    s = x + sj + selected[k]
                    if s % a == 0:
                        return False

    return True

def _is_valid(selected):
    """
    Full check for validity of the set.
    """
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        # Check if a divides sum of any 3 others
        # Others are indices j, k, l != i
        for j in range(n):
            if j == i: continue
            sj = selected[j]
            for k in range(j + 1, n):
                if k == i: continue
                sk = selected[k]
                for l in range(k + 1, n):
                    if l == i: continue
                    if (sj + sk + selected[l]) % a == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial Greedy Construction
    # Large numbers are safer divisors.
    # We start with all numbers 1..192 and try to build a set.
    # A good heuristic is to start with the largest numbers.
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy from high to low
    # This usually produces a large valid set quickly.
    current = []
    for x in range(192, 0, -1):
        if _check_can_add(x, current):
            current.append(x)
    
    best_set = current
    best_size = len(current)
    
    # Strategy 2: Local Search / Randomized Restart
    # We will run multiple iterations of local search.
    # Each iteration starts from a perturbed version of the best set or a new greedy run.
    
    all_numbers = set(range(1, 193))
    
    for iter in range(500): # Max iterations, will stop by time
        if time.time() > deadline:
            break
            
        # 1. Get a starting set
        # Sometimes restart from scratch with random order
        # Sometimes continue from best_set
        if iter % 10 == 0:
            # Randomized Greedy
            order = list(range(192, 0, -1))
            random.shuffle(order)
            # Bias towards large numbers? 
            # Let's just shuffle.
            current = []
            for x in order:
                if _check_can_add(x, current):
                    current.append(x)
        else:
            # Start from best found so far
            current = list(best_set)
            # Perturb: remove a few random elements
            if len(current) > 5:
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(current) - 1)
                    current.pop(idx)
        
        current_set = set(current)
        
        # 2. Local Search Loop
        no_improve_count = 0
        while time.time() < deadline - 1.0 and no_improve_count < 20:
            improved = False
            
            # A. Try to add an element
            # Candidates are numbers not in current_set
            # We prioritize large numbers as they are less likely to be divisors
            candidates = [x for x in range(192, 0, -1) if x not in current_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if _check_can_add(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    added = True
                    break # Restart search loop with new state
            
            if added:
                no_improve_count = 0
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
                continue # Restart the while loop
            
            # B. If no addition, try swaps
            # B1. Swap 1-for-1 (maintain size, change composition)
            # Pick a random element to remove and try to replace it
            if len(current) > 0:
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 1.0:
                        break
                    idx = random.randint(0, len(current) - 1)
                    removed = current[idx]
                    temp_current = [e for i, e in enumerate(current) if i != idx]
                    temp_set = set(temp_current)
                    
                    # Try to add a replacement
                    # We want to find x not in temp_set such that temp_current + [x] is valid
                    # And ideally, this state allows future additions.
                    # For now, just check validity.
                    # To make it effective, we should prefer x that is "large" or "compatible".
                    # Let's try a few candidates.
                    repl_candidates = [x for x in range(192, 0, -1) if x not in temp_set]
                    random.shuffle(repl_candidates)
                    
                    for x in repl_candidates[:20]: # Check top 20 candidates
                        if _check_can_add(x, temp_current):
                            # Accept the swap if it doesn't reduce size (it doesn't)
                            # We accept it to explore the space
                            current = temp_current + [x]
                            current_set = temp_set | {x}
                            improved = True
                            break
                    if improved:
                        no_improve_count = 0
                        break # Restart while loop
            
            if improved:
                continue
                
            # B2. Swap 1-for-2 (Increase size by 1)
            # Remove one element, try to add two
            if len(current) > 0:
                for _ in range(10):
                    if time.time() > deadline - 1.0:
                        break
                    idx = random.randint(0, len(current) - 1)
                    removed = current[idx]
                    temp_current = [e for i, e in enumerate(current) if i != idx]
                    temp_set = set(temp_current)
                    
                    # Try to add two elements
                    # Greedily add one, then try to add another
                    add_candidates = [x for x in range(192, 0, -1) if x not in temp_set]
                    random.shuffle(add_candidates)
                    
                    new_current = list(temp_current)
                    new_set = set(temp_current)
                    added_count = 0
                    
                    for x in add_candidates:
                        if added_count == 2:
                            break
                        if _check_can_add(x, new_current):
                            new_current.append(x)
                            new_set.add(x)
                            added_count += 1
                    
                    if len(new_current) > len(current):
                        current = new_current
                        current_set = new_set
                        improved = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = list(current)
                        break # Restart while loop
            
            if improved:
                continue
                
            # No improvement in this pass
            no_improve_count += 1

    # Final Validation and Return
    if not _is_valid(best_set):
        # Fallback if something went wrong
        # Find a valid subset of best_set
        valid_set = []
        for x in best_set:
            if _check_can_add(x, valid_set):
                valid_set.append(x)
        best_set = valid_set
        
    if len(best_set) == 0:
        best_set = [192]

    return best_set
