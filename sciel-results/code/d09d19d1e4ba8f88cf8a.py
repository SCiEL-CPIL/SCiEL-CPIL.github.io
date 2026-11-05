import random
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0

    # State: A, B, sums_A, sums_B, diffs_A, diffs_B
    # We will maintain these explicitly for fast updates.
    
    def initialize_greedy():
        """Create a decent starting point using a simple greedy approach."""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Use a random order for initialization
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            # Check if x can be added to A
            valid_A = True
            for a in A:
                if (x + a) in sums_A or (x - a) in diffs_B or (a - x) in diffs_B:
                    valid_A = False
                    break
            if valid_A and (2 * x) in sums_A:
                valid_A = False
            
            # Check if x can be added to B
            valid_B = True
            for b in B:
                if (x + b) in sums_B or (x - b) in diffs_A or (b - x) in diffs_A:
                    valid_B = False
                    break
            if valid_B and (2 * x) in sums_B:
                valid_B = False
            
            if valid_A and not valid_B:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                sums_A.add(2 * x)
            elif valid_B and not valid_A:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                sums_B.add(2 * x)
            elif valid_A and valid_B:
                # Randomly choose
                if random.random() < 0.5:
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(2 * x)
                else:
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(2 * x)
            
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        """Check if x can be added to A."""
        if x in A: return False
        if (2 * x) in sums_A: return False
        for a in A:
            if (x + a) in sums_A: return False
            if (x - a) in diffs_B: return False
            if (a - x) in diffs_B: return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        """Check if x can be added to B."""
        if x in B: return False
        if (2 * x) in sums_B: return False
        for b in B:
            if (x + b) in sums_B: return False
            if (x - b) in diffs_A: return False
            if (b - x) in diffs_A: return False
        return True

    def remove_from_A(x, A, sums_A, diffs_A):
        """Remove x from A and update sets. Returns new A, sums_A, diffs_A."""
        A_new = [a for a in A if a != x]
        sums_A_new = set()
        diffs_A_new = set()
        # Rebuild is O(k^2) which is fine for k~64
        for i, a in enumerate(A_new):
            sums_A_new.add(2 * a)
            for b in A_new[i+1:]:
                sums_A_new.add(a + b)
                diffs_A_new.add(a - b)
                diffs_A_new.add(b - a)
        return A_new, sums_A_new, diffs_A_new

    def remove_from_B(x, B, sums_B, diffs_B):
        """Remove x from B and update sets. Returns new B, sums_B, diffs_B."""
        B_new = [b for b in B if b != x]
        sums_B_new = set()
        diffs_B_new = set()
        for i, b in enumerate(B_new):
            sums_B_new.add(2 * b)
            for c in B_new[i+1:]:
                sums_B_new.add(b + c)
                diffs_B_new.add(b - c)
                diffs_B_new.add(c - b)
        return B_new, sums_B_new, diffs_B_new

    # Initialize
    A, B, sums_A, sums_B, diffs_A, diffs_B = initialize_greedy()
    best_score = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    
    # Precompute list of all numbers for random selection
    all_numbers = list(range(1, n + 1))
    
    while time.time() - start_time < time_limit and T > min_T:
        current_score = len(A) + len(B)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random element
            x = random.choice(all_numbers)
            if x not in A and x not in B:
                # Check A
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    # Apply
                    A_new = A + [x]
                    sums_A_new = sums_A.copy()
                    diffs_A_new = diffs_A.copy()
                    for a in A:
                        sums_A_new.add(x + a)
                        diffs_A_new.add(x - a)
                        diffs_A_new.add(a - x)
                    sums_A_new.add(2 * x)
                    
                    new_score = len(A_new) + len(B)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        A, sums_A, diffs_A = A_new, sums_A_new, diffs_A_new
                        if new_score > best_score:
                            best_score = new_score
                            best_A, best_B = list(A), list(B)
                elif can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                    # Apply to B
                    B_new = B + [x]
                    sums_B_new = sums_B.copy()
                    diffs_B_new = diffs_B.copy()
                    for b in B:
                        sums_B_new.add(x + b)
                        diffs_B_new.add(x - b)
                        diffs_B_new.add(b - x)
                    sums_B_new.add(2 * x)
                    
                    new_score = len(A) + len(B_new)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        B, sums_B, diffs_B = B_new, sums_B_new, diffs_B_new
                        if new_score > best_score:
                            best_score = new_score
                            best_A, best_B = list(A), list(B)
            elif x in A and x not in B:
                 # Move from A to B
                 if can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                     # Remove from A, Add to B
                     A_old, sums_A_old, diffs_A_old = A, sums_A, diffs_A
                     A_temp, sums_A_temp, diffs_A_temp = remove_from_A(x, A, sums_A, diffs_A)
                     
                     # Check if valid to add to B given new diffs_A
                     if can_add_to_B(x, B, sums_B, diffs_A_temp, diffs_B):
                         B_new = B + [x]
                         sums_B_new = sums_B.copy()
                         diffs_B_new = diffs_B.copy()
                         for b in B:
                             sums_B_new.add(x + b)
                             diffs_B_new.add(x - b)
                             diffs_B_new.add(b - x)
                         sums_B_new.add(2 * x)
                         
                         new_score = len(A_temp) + len(B_new)
                         delta = new_score - current_score
                         if delta > 0 or random.random() < pow(2.718281828, delta / T):
                             A, sums_A, diffs_A = A_temp, sums_A_temp, diffs_A_temp
                             B, sums_B, diffs_B = B_new, sums_B_new, diffs_B_new
                             if new_score > best_score:
                                 best_score = new_score
                                 best_A, best_B = list(A), list(B)
            elif x in B and x not in A:
                 # Move from B to A
                 if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                     B_old, sums_B_old, diffs_B_old = B, sums_B, diffs_B
                     B_temp, sums_B_temp, diffs_B_temp = remove_from_B(x, B, sums_B, diffs_B)
                     
                     if can_add_to_A(x, A, sums_A, diffs_A, diffs_B_temp):
                         A_new = A + [x]
                         sums_A_new = sums_A.copy()
                         diffs_A_new = diffs_A.copy()
                         for a in A:
                             sums_A_new.add(x + a)
                             diffs_A_new.add(x - a)
                             diffs_A_new.add(a - x)
                         sums_A_new.add(2 * x)
                         
                         new_score = len(A_new) + len(B_temp)
                         delta = new_score - current_score
                         if delta > 0 or random.random() < pow(2.718281828, delta / T):
                             A, sums_A, diffs_A = A_new, sums_A_new, diffs_A_new
                             B, sums_B, diffs_B = B_temp, sums_B_temp, diffs_B_temp
                             if new_score > best_score:
                                 best_score = new_score
                                 best_A, best_B = list(A), list(B)

        else:
            # Try to remove a random element (to allow reshuffling)
            if A:
                x = random.choice(A)
                A_new, sums_A_new, diffs_A_new = remove_from_A(x, A, sums_A, diffs_A)
                new_score = len(A_new) + len(B)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    A, sums_A, diffs_A = A_new, sums_A_new, diffs_A_new
            elif B:
                x = random.choice(B)
                B_new, sums_B_new, diffs_B_new = remove_from_B(x, B, sums_B, diffs_B)
                new_score = len(A) + len(B_new)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    B, sums_B, diffs_B = B_new, sums_B_new, diffs_B_new

        T *= cooling_rate

    # Final check and ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
