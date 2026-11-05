import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    best_set = []
    best_size = 0
    
    def try_add(x):
        """Try to add x to A. Returns True if successful."""
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid - perform add
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(x):
        """Remove x from A. Returns True if successful."""
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def check_swap_valid(y, x):
        """Check if swapping y (in A) with x (not in A) is valid."""
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
        
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
        
        return True

    def perform_swap(y, x):
        """Perform swap of y with x."""
        remove_element(y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1

    # Phase 1: Greedy initialization with multiple random orderings
    for trial in range(10):
        A.clear()
        in_A.clear()
        sum_counts = [0] * (2 * N + 1)
        
        # Random ordering with some bias toward smaller numbers
        candidates = list(range(1, N + 1))
        if trial % 2 == 0:
            random.shuffle(candidates)
        else:
            # Mix of sorted and random
            candidates = sorted(candidates, key=lambda x: random.random() * x)
        
        for x in candidates:
            if try_add(x):
                pass
        
        if len(A) > best_size:
            best_size = len(A)
            best_set = A[:]
    
    # Restore best state
    A.clear()
    in_A.clear()
    sum_counts = [0] * (2 * N + 1)
    for x in best_set:
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
    
    # Phase 2: Local search with smart strategies
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Strategy 1: Try to add elements (most beneficial)
        added = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        
        # Sort candidates by "promisingness" - prefer those with fewer conflicts
        if candidates:
            def conflict_score(x):
                score = 0
                if sum_counts[2*x] >= 2:
                    score += 10
                for y in A:
                    if sum_counts[x+y] >= 2:
                        score += 1
                return score
            
            candidates.sort(key=conflict_score)
            random.shuffle(candidates[:min(50, len(candidates))])  # Shuffle top candidates
            
            for x in candidates:
                if try_add(x):
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                        no_improve_count = 0
                    break
        
        if added:
            continue
        
        # Strategy 2: Targeted swaps - try to find valid swaps
        swapped = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        
        if candidates and A:
            # Try targeted swaps: for each x not in A, find a y in A that can be swapped
            # Prioritize x with lower conflict scores
            def x_conflict(x):
                score = 0
                if sum_counts[2*x] >= 2:
                    score += 10
                for y in A:
                    if sum_counts[x+y] >= 2:
                        score += 1
                return score
            
            candidates.sort(key=x_conflict)
            
            # Try top candidates first
            for x in candidates[:50]:
                # Find a y in A that can be swapped with x
                # Prioritize y with high "constraint" (many sums at capacity)
                def y_constraint(y):
                    score = 0
                    for z in A:
                        s = y + z
                        if sum_counts[s] >= 2:
                            score += 1
                    if sum_counts[2*y] >= 2:
                        score += 1
                    return score
                
                # Sort A by constraint (descending)
                A_sorted = sorted(A, key=y_constraint, reverse=True)
                
                for y in A_sorted[:20]:
                    if check_swap_valid(y, x):
                        perform_swap(y, x)
                        swapped = True
                        no_improve_count = 0
                        break
                
                if swapped:
                    break
        
        if swapped:
            continue
        
        # Strategy 3: Random swaps for exploration
        if A and candidates:
            for _ in range(100):
                y = random.choice(A)
                x = random.choice(candidates)
                if check_swap_valid(y, x):
                    perform_swap(y, x)
                    swapped = True
                    break
        
        if swapped:
            continue
        
        # Strategy 4: Removal to break deadlock
        if A:
            # Score each element by how many sums it contributes to that are at capacity
            removal_scores = {}
            for y in A:
                score = 0
                for z in A:
                    s = y + z
                    if sum_counts[s] >= 2:
                        score += 1
                if sum_counts[2*y] >= 2:
                    score += 1
                removal_scores[y] = score
            
            max_score = max(removal_scores.values())
            candidates_to_remove = [y for y in A if removal_scores[y] >= max_score - 1]
            if not candidates_to_remove:
                candidates_to_remove = A
            y = random.choice(candidates_to_remove)
            remove_element(y)
            no_improve_count += 1
        
        # If too many iterations without improvement, restart with perturbation
        if no_improve_count >= max_no_improve:
            # Perturb: remove 5-10 random elements and restart local search
            num_remove = random.randint(5, 10)
            for _ in range(min(num_remove, len(A))):
                if A:
                    y = random.choice(A)
                    remove_element(y)
            no_improve_count = 0
    
    return best_set
