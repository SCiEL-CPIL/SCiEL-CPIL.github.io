import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    def rebuild_state(A, B):
        """Rebuild sums and diffs sets from A and B."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
                if i != j:
                    diffs_A.add(abs(A[i] - A[j]))
        
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
                if i != j:
                    diffs_B.add(abs(B[i] - B[j]))
        
        return sums_A, sums_B, diffs_A, diffs_B
    
    def can_add_to_set(x, S, sums_S, diffs_other):
        """Check if x can be added to set S given its sums and other set's diffs."""
        if 2 * x in sums_S:
            return False
        for a in S:
            if x + a in sums_S:
                return False
        for a in S:
            d = abs(x - a)
            if d in diffs_other:
                return False
        return True
    
    def add_to_set(x, S, sums_S, diffs_S):
        """Add x to set S and update sums and diffs."""
        sums_S.add(2 * x)
        for a in S:
            sums_S.add(x + a)
            diffs_S.add(abs(x - a))
        S.append(x)
    
    def greedy_construct(order, prefer_A_first=True):
        """Greedy construction with given order."""
        A = []
        B = []
        sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            if prefer_A_first:
                if can_add_to_set(x, A, sums_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif can_add_to_set(x, B, sums_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
            else:
                if can_add_to_set(x, B, sums_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                elif can_add_to_set(x, A, sums_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
        
        return A, B
    
    def local_improve(A, B):
        """Improve by trying to add candidates and performing swaps."""
        A = A[:]
        B = B[:]
        
        # Phase 1: Try to add any candidate that fits
        for _ in range(3):  # Multiple passes
            if time.time() - start_time > time_limit:
                break
            sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
            improved = False
            for x in range(1, n + 1):
                if time.time() - start_time > time_limit:
                    break
                if x not in A and x not in B:
                    if can_add_to_set(x, A, sums_A, diffs_B):
                        add_to_set(x, A, sums_A, diffs_A)
                        improved = True
                    elif can_add_to_set(x, B, sums_B, diffs_A):
                        add_to_set(x, B, sums_B, diffs_B)
                        improved = True
            if not improved:
                break
        
        # Phase 2: Try swaps to potentially improve
        if time.time() - start_time < time_limit:
            for _ in range(2):  # Multiple swap passes
                if time.time() - start_time > time_limit:
                    break
                sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
                improved = False
                
                # Try moving an element from A to B or vice versa
                for a in A[:]:
                    if time.time() - start_time > time_limit:
                        break
                    # Remove a from A
                    A.remove(a)
                    new_sums_A, new_sums_B, new_diffs_A, new_diffs_B = rebuild_state(A, B)
                    
                    # Try adding a to B
                    if can_add_to_set(a, B, new_sums_B, new_diffs_A):
                        add_to_set(a, B, new_sums_B, new_diffs_B)
                        improved = True
                    else:
                        # Put a back in A
                        add_to_set(a, A, new_sums_A, new_diffs_A)
                
                if not improved:
                    break
        
        return A, B
    
    def create_biased_order(bias_strength=0.3):
        """Create a biased ordering with smaller numbers preferred."""
        order = list(range(1, n + 1))
        for i in range(len(order) - 1):
            if random.random() < bias_strength * (1 - i / len(order)):
                j = random.randint(i, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order
    
    # Main search loop
    restarts = 0
    while time.time() - start_time < time_limit:
        # Vary bias strength for diversity
        bias = 0.2 + random.random() * 0.3
        order = create_biased_order(bias)
        
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(order, prefer_A)
        
        # Local improvement pass
        A, B = local_improve(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
        
        restarts += 1
        
        # If we've found a good solution, do more careful refinement
        if best_size >= 30 and restarts % 10 == 0:
            # Try a more deterministic order for refinement
            A, B = local_improve(best_A, best_B)
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A = A[:]
                best_B = B[:]

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
