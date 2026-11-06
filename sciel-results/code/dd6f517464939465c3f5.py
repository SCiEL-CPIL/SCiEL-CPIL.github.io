import random
import time

def construct():
    # Time limit management
    start_time = time.time()
    time_limit = 50  # seconds, leave buffer for 60s limit
    
    best_A = []
    best_B = []
    best_size = 0
    
    def is_valid_addition(x, set_A, set_B, sums_A, sums_B, diffs_A, diffs_B):
        # Check Sidon condition for set_A
        if 2 * x in sums_A:
            return False
        for a in set_A:
            if x + a in sums_A:
                return False
        # Check difference disjointness
        # New diffs in A: x-a and a-x
        # These must not be in diffs_B
        for a in set_A:
            if (x - a) in diffs_B or (a - x) in diffs_B:
                return False
        return True

    def add_to_set(x, set_A, set_B, sums_A, sums_B, diffs_A, diffs_B, target):
        if target == 'A':
            set_A.append(x)
            sums_A.add(2 * x)
            for a in set_A[:-1]:
                sums_A.add(x + a)
                diffs_A.add(x - a)
                diffs_A.add(a - x)
        else:
            set_B.append(x)
            sums_B.add(2 * x)
            for b in set_B[:-1]:
                sums_B.add(x + b)
                diffs_B.add(x - b)
                diffs_B.add(b - x)

    def remove_from_set(x, set_A, set_B, sums_A, sums_B, diffs_A, diffs_B, target):
        if target == 'A':
            set_A.remove(x)
            sums_A.remove(2 * x)
            for a in set_A:
                sums_A.remove(x + a)
                diffs_A.remove(x - a)
                diffs_A.remove(a - x)
        else:
            set_B.remove(x)
            sums_B.remove(2 * x)
            for b in set_B:
                sums_B.remove(x + b)
                diffs_B.remove(x - b)
                diffs_B.remove(b - x)

    # Initial construction using greedy
    def initial_greedy():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try A
            if is_valid_addition(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
                # Check B validity for x? No, just add to A if valid for A.
                # But we might want to balance.
                # For now, just add to A if possible, else B.
                add_to_set(x, A, B, sums_A, sums_B, diffs_A, diffs_B, 'A')
            else:
                # Try B
                # Need a separate check for B
                # is_valid_addition checks against set_A's sums and set_B's diffs?
                # My is_valid_addition(x, set_A, ...) checks x against set_A's sums and set_B's diffs.
                # If we want to add to B, we need to check x against set_B's sums and set_A's diffs.
                
                # Let's rewrite is_valid_addition to be target-specific.
                pass
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    # Refactored validity check
    def can_add_to_A(x, A, sums_A, diffs_B):
        if 2 * x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if (x - a) in diffs_B or (a - x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if 2 * x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if (x - b) in diffs_A or (b - x) in diffs_A:
                return False
        return True

    def do_add_to_A(x, A, sums_A, diffs_A):
        A.append(x)
        sums_A.add(2 * x)
        for a in A[:-1]:
            sums_A.add(x + a)
            diffs_A.add(x - a)
            diffs_A.add(a - x)

    def do_add_to_B(x, B, sums_B, diffs_B):
        B.append(x)
        sums_B.add(2 * x)
        for b in B[:-1]:
            sums_B.add(x + b)
            diffs_B.add(x - b)
            diffs_B.add(b - x)

    def do_remove_from_A(x, A, sums_A, diffs_A):
        A.remove(x)
        sums_A.remove(2 * x)
        for a in A:
            sums_A.remove(x + a)
            diffs_A.remove(x - a)
            diffs_A.remove(a - x)

    def do_remove_from_B(x, B, sums_B, diffs_B):
        B.remove(x)
        sums_B.remove(2 * x)
        for b in B:
            sums_B.remove(x + b)
            diffs_B.remove(x - b)
            diffs_B.remove(b - x)

    # Main loop
    for restart in range(50): # Limit restarts to ensure time
        if time.time() - start_time > time_limit:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Greedy fill
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Prefer adding to smaller set
            if len(A) <= len(B):
                if can_add_to_A(x, A, sums_A, diffs_B):
                    do_add_to_A(x, A, sums_A, diffs_A)
                elif can_add_to_B(x, B, sums_B, diffs_A):
                    do_add_to_B(x, B, sums_B, diffs_B)
            else:
                if can_add_to_B(x, B, sums_B, diffs_A):
                    do_add_to_B(x, B, sums_B, diffs_B)
                elif can_add_to_A(x, A, sums_A, diffs_B):
                    do_add_to_A(x, A, sums_A, diffs_A)
            
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A = A[:]
                best_B = B[:]
                
        # Local Search / Hill Climbing
        # Try to add more elements
        # We might have missed some because of the greedy order.
        # Try random additions from remaining candidates.
        remaining = [x for x in range(1, 257) if x not in A and x not in B]
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for x in remaining:
                if len(A) <= len(B):
                    if can_add_to_A(x, A, sums_A, diffs_B):
                        do_add_to_A(x, A, sums_A, diffs_A)
                        remaining.remove(x)
                        improved = True
                        if len(A) + len(B) > best_size:
                            best_size = len(A) + len(B)
                            best_A = A[:]
                            best_B = B[:]
                    elif can_add_to_B(x, B, sums_B, diffs_A):
                        do_add_to_B(x, B, sums_B, diffs_B)
                        remaining.remove(x)
                        improved = True
                        if len(A) + len(B) > best_size:
                            best_size = len(A) + len(B)
                            best_A = A[:]
                            best_B = B[:]
                else:
                    if can_add_to_B(x, B, sums_B, diffs_A):
                        do_add_to_B(x, B, sums_B, diffs_B)
                        remaining.remove(x)
                        improved = True
                        if len(A) + len(B) > best_size:
                            best_size = len(A) + len(B)
                            best_A = A[:]
                            best_B = B[:]
                    elif can_add_to_A(x, A, sums_A, diffs_B):
                        do_add_to_A(x, A, sums_A, diffs_A)
                        remaining.remove(x)
                        improved = True
                        if len(A) + len(B) > best_size:
                            best_size = len(A) + len(B)
                            best_A = A[:]
                            best_B = B[:]
                
                if time.time() - start_time > time_limit:
                    break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
