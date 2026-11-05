import random
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    N = 1000
    max_sum = 2 * N

    def is_valid_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            s = x + y
            if sum_counts[s] >= 2:
                return False
        return True

    def try_add(A, in_A, sum_counts, x):
        if not is_valid_add(A, in_A, sum_counts, x):
            return False
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def check_swap_validity(A, in_A, sum_counts, y, x):
        if x in in_A or y not in in_A:
            return False
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
        return True

    def perform_swap(A, in_A, sum_counts, y, x):
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1

    def greedy_with_priority_queue():
        """Greedy initialization using priority queue with conflict-cost scoring."""
        A = []
        in_A = set()
        sum_counts = [0] * (max_sum + 1)
        
        # Priority queue: (conflict_cost, x)
        # We'll rebuild the heap as elements are added
        candidates = list(range(1, N + 1))
        
        while candidates:
            # Score each candidate by its future conflict cost
            best_x = None
            best_cost = float('inf')
            best_idx = -1
            
            for i, x in enumerate(candidates):
                # Check if x can be added
                if x in in_A:
                    continue
                if sum_counts[2*x] >= 2:
                    continue
                valid = True
                for y in A:
                    s = x + y
                    if sum_counts[s] >= 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate conflict cost: how many sums will become saturated (reach count 2)
                # or close to saturation (reach count 1 when they were 0, meaning we're using up budget)
                cost = 0
                for y in A:
                    s = x + y
                    if sum_counts[s] == 1:
                        cost += 10  # High cost: will saturate this sum
                    elif sum_counts[s] == 0:
                        cost += 1   # Low cost: uses one slot
                if sum_counts[2*x] == 1:
                    cost += 10
                elif sum_counts[2*x] == 0:
                    cost += 1
                
                # Add a small random tiebreaker
                cost += random.random() * 0.1
                
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_idx = i
            
            if best_x is None:
                break
            
            # Add the best candidate
            in_A.add(best_x)
            A.append(best_x)
            sum_counts[2*best_x] += 1
            for y in A[:-1]:
                sum_counts[best_x+y] += 1
            
            # Remove from candidates
            candidates.pop(best_idx)
            
            # Early exit if time is running low
            if time.time() > deadline - 30:
                break
        
        return A, in_A, sum_counts

    # Try multiple different initialization strategies
    best_set = []
    best_size = 0
    
    # Strategy 1: Standard greedy (smallest first)
    A1 = []
    in_A1 = set()
    sum_counts1 = [0] * (max_sum + 1)
    for x in range(1, N + 1):
        if try_add(A1, in_A1, sum_counts1, x):
            pass
    if len(A1) > best_size:
        best_size = len(A1)
        best_set = A1[:]
    
    # Strategy 2: Priority queue greedy
    if time.time() < deadline - 40:
        A2, in_A2, sum_counts2 = greedy_with_priority_queue()
        if len(A2) > best_size:
            best_size = len(A2)
            best_set = A2[:]
    
    # Strategy 3: Randomized greedy with multiple trials
    if time.time() < deadline - 30:
        for trial in range(5):
            if time.time() > deadline - 25:
                break
            A3 = []
            in_A3 = set()
            sum_counts3 = [0] * (max_sum + 1)
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if try_add(A3, in_A3, sum_counts3, x):
                    pass
            if len(A3) > best_size:
                best_size = len(A3)
                best_set = A3[:]
    
    # Local search on the best set found
    A = best_set[:]
    in_A = set(A)
    sum_counts = [0] * (max_sum + 1)
    for y in A:
        sum_counts[2*y] += 1
        for z in A:
            if z > y:
                sum_counts[y+z] += 1
            elif z == y:
                pass  # Already counted 2*y
    
    # Actually, let's rebuild sum_counts properly
    sum_counts = [0] * (max_sum + 1)
    for i, y in enumerate(A):
        sum_counts[2*y] += 1
        for z in A[i+1:]:
            sum_counts[y+z] += 1
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5:
        # 1. Try to add
        added = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                if try_add(A, in_A, sum_counts, x):
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                        no_improve_count = 0
                    break
        
        if added:
            continue
            
        # 2. Try Swap - more aggressive
        swapped = False
        if A and candidates:
            # Try multiple swap attempts
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                y = random.choice(A)
                x = random.choice(candidates)
                if check_swap_validity(A, in_A, sum_counts, y, x):
                    perform_swap(A, in_A, sum_counts, y, x)
                    swapped = True
                    break
        
        if swapped:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Remove most constrained element to break deadlock
                if A:
                    scores = {}
                    for y in A:
                        score = 0
                        for z in A:
                            if sum_counts[y+z] >= 2:
                                score += 1
                        if sum_counts[2*y] >= 2:
                            score += 1
                        scores[y] = score
                    max_score = max(scores.values())
                    to_remove = [y for y in A if scores[y] == max_score]
                    y = random.choice(to_remove)
                    remove_element(A, in_A, sum_counts, y)
                    no_improve_count = 0
                continue
        
        # 3. If no swap, try removing an element
        if A:
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
            no_improve_count += 1
            if no_improve_count > 5000:
                # Restart local search with fresh greedy
                if time.time() < deadline - 10:
                    A, in_A, sum_counts = greedy_with_priority_queue()
                    best_size = max(best_size, len(A))
                    best_set = A[:]
                    no_improve_count = 0

    return best_set
