# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        for a in A:
            if (x + a) in sums_A:
                return False
        if (2 * x) in sums_A:
            return False
        for a in A:
            d = abs(x - a)
            if d in diffs_B:
                return False
        return True
    
    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        for b in B:
            if (x + b) in sums_B:
                return False
        if (2 * x) in sums_B:
                return False
        for b in B:
            d = abs(x - b)
            if d in diffs_A:
                return False
        return True

    def rebuild_state(A, B):
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        for i, a in enumerate(A):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(abs(A[i] - A[j]))
        for i, b in enumerate(B):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(abs(B[i] - B[j]))
        return sums_A, sums_B, diffs_A, diffs_B

    def run_greedy(order, prefer_smaller=False, prefer_A_first=True):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            valid_A = can_add_to_A(x, A, sums_A, diffs_A, diffs_B)
            valid_B = can_add_to_B(x, B, sums_B, diffs_A, diffs_B)
            
            if valid_A and valid_B:
                if prefer_smaller:
                    if len(A) < len(B):
                        add_to_A = True
                    elif len(B) < len(A):
                        add_to_A = False
                    else:
                        add_to_A = prefer_A_first if prefer_A_first is not None else random.choice([True, False])
                else:
                    add_to_A = True
            elif valid_A:
                add_to_A = True
            elif valid_B:
                add_to_A = False
            else:
                continue
            
            if add_to_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2 * x)
            else:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                sums_B.add(2 * x)
        
        return A, B

    def local_search(A, B, max_rounds=5):
        """Improve by adding, swapping, or moving elements"""
        A = list(A)
        B = list(B)
        
        for _ in range(max_rounds):
            improved = False
            sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
            
            # Try adding unassigned elements
            assigned = set(A) | set(B)
            for x in candidates:
                if x in assigned:
                    continue
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    improved = True
                    break
                elif can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                    B.append(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: move element from A to B or vice versa
            swapped = False
            for i, a in enumerate(A):
                # Try moving a from A to B
                A_test = A[:i] + A[i+1:]
                if can_add_to_B(a, B, sums_B, diffs_A, diffs_B):
                    # Check that removing a from A doesn't break anything (it won't, we're removing)
                    B_test = B + [a]
                    sums_A2, sums_B2, diffs_A2, diffs_B2 = rebuild_state(A_test, B_test)
                    # Verify all constraints still hold
                    if _validate(A_test, B_test, sums_A2, sums_B2, diffs_A2, diffs_B2):
                        A = A_test
                        B = B_test
                        swapped = True
                        improved = True
                        break
            if swapped:
                continue
            
            for i, b in enumerate(B):
                # Try moving b from B to A
                B_test = B[:i] + B[i+1:]
                if can_add_to_A(b, A, sums_A, diffs_A, diffs_B):
                    A_test = A + [b]
                    sums_A2, sums_B2, diffs_A2, diffs_B2 = rebuild_state(A_test, B_test)
                    if _validate(A_test, B_test, sums_A2, sums_B2, diffs_A2, diffs_B2):
                        A = A_test
                        B = B_test
                        swapped = True
                        improved = True
                        break
            if swapped:
                continue
            
            # Try direct swap: exchange a from A with b from B
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    A_test = A[:i] + [b] + A[i+1:]
                    B_test = B[:j] + [a] + B[j+1:]
                    sums_A2, sums_B2, diffs_A2, diffs_B2 = rebuild_state(A_test, B_test)
                    if _validate(A_test, B_test, sums_A2, sums_B2, diffs_A2, diffs_B2):
                        A = A_test
                        B = B_test
                        swapped = True
                        improved = True
                        break
                if swapped:
                    break
            if swapped:
                continue
            
            if not improved:
                break
        
        return A, B

    def _validate(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Verify all constraints are satisfied"""
        # Check Sidon property for A
        seen_sums = set()
        for i, a in enumerate(A):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in seen_sums:
                    return False
                seen_sums.add(s)
        
        # Check Sidon property for B
        seen_sums = set()
        for i, b in enumerate(B):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in seen_sums:
                    return False
                seen_sums.add(s)
        
        # Check disjoint differences
        common = diffs_A & diffs_B
        if common:
            return False
        
        return True

    # Strategy 1: Deterministic natural order
    A, B = run_greedy(candidates, prefer_smaller=True, prefer_A_first=True)
    A, B = local_search(A, B)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
        
    # Strategy 2: Deterministic reverse order
    A, B = run_greedy(list(reversed(candidates)), prefer_smaller=True, prefer_A_first=False)
    A, B = local_search(A, B)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
        
    # Strategy 3: Random shuffles with varied heuristics
    random.seed(42)
    
    trial_count = 0
    while time.time() < deadline:
        order = list(candidates)
        random.shuffle(order)
        
        trial_count += 1
        if trial_count % 3 == 0:
            A, B = run_greedy(order, prefer_smaller=True, prefer_A_first=True)
        elif trial_count % 3 == 1:
            A, B = run_greedy(order, prefer_smaller=True, prefer_A_first=False)
        else:
            A, B = run_greedy(order, prefer_smaller=False, prefer_A_first=True)
        
        A, B = local_search(A, B)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        if time.time() > deadline - 5:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
