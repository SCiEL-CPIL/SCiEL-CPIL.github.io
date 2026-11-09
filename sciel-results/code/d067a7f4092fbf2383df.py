import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit
    
    best_size = float('inf')
    best_set = []
    
    # Precompute all pairs for faster sum checking
    # We'll use a set of sums for the current set
    
    def build_sum_set(A):
        """Build the set of all unordered sums a+b with a<=b for a,b in A."""
        S = set()
        for i, a in enumerate(A):
            S.add(2 * a)
            for b in A[i+1:]:
                S.add(a + b)
        return S
    
    def is_sidon(A):
        """Check if A is a Sidon set."""
        S = set()
        for i, a in enumerate(A):
            if 2 * a in S:
                return False
            S.add(2 * a)
            for b in A[i+1:]:
                s = a + b
                if s in S:
                    return False
                S.add(s)
        return True
    
    def is_maximal(A):
        """Check if A is inclusion-maximal Sidon in [1,N]."""
        if not A:
            return False
        A_set = set(A)
        S = build_sum_set(A)
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if x can be added
            if 2 * x in S:
                continue
            ok = True
            for a in A:
                if x + a in S:
                    ok = False
                    break
            if ok:
                return False
        return True
    
    def greedy_construct(rng, strategy='random'):
        """Greedy construction of a Sidon set."""
        A = []
        S = set()
        
        while True:
            addable = []
            for x in range(1, N + 1):
                if x in A:
                    continue
                if 2 * x in S:
                    continue
                ok = True
                for a in A:
                    if x + a in S:
                        ok = False
                        break
                if ok:
                    addable.append(x)
            
            if not addable:
                break
            
            if strategy == 'smallest':
                x = min(addable)
            elif strategy == 'largest':
                x = max(addable)
            elif strategy == 'central':
                x = addable[len(addable) // 2]
            else:  # random
                x = rng.choice(addable)
            
            A.append(x)
            S.add(2 * x)
            for a in A[:-1]:
                S.add(x + a)
        
        return A
    
    def local_search_remove(A, rng, time_limit=2.0):
        """Try to remove elements while maintaining maximality."""
        A = list(A)
        local_start = time.time()
        
        while time.time() - local_start < time_limit:
            improved = False
            # Shuffle for randomization
            indices = list(range(len(A)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() - local_start > time_limit:
                    break
                if idx >= len(A):
                    continue
                
                candidate = A[idx]
                A_new = A[:idx] + A[idx+1:]
                
                if not A_new:
                    continue
                
                # Check if A_new is still maximal
                if is_maximal(A_new):
                    A = A_new
                    improved = True
                    break  # Restart after improvement
            
            if not improved:
                break
        
        return A
    
    def swap_search(A, rng, time_limit=2.0):
        """Try swapping one element for another to reduce size."""
        A = list(A)
        local_start = time.time()
        
        while time.time() - local_start < time_limit:
            improved = False
            indices = list(range(len(A)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() - local_start > time_limit:
                    break
                if idx >= len(A):
                    continue
                
                candidate = A[idx]
                A_without = A[:idx] + A[idx+1:]
                
                # Try adding each missing element
                A_set = set(A_without)
                for x in range(1, N + 1):
                    if x in A_set:
                        continue
                    A_swap = A_without + [x]
                    
                    # Check if it's a valid Sidon set
                    if not is_sidon(A_swap):
                        continue
                    
                    # Check if it's maximal
                    if not is_maximal(A_swap):
                        continue
                    
                    # If we can swap, do it (same size, might help future removals)
                    # Actually, swaps don't reduce size. Let's focus on removals.
                    pass
            
            if not improved:
                break
        
        return A
    
    # Main search loop
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Try different greedy strategies
        strategies = ['random', 'smallest', 'largest', 'central']
        strategy = strategies[seed % len(strategies)]
        
        A = greedy_construct(rng, strategy)
        
        if not A:
            continue
        
        # Local search: try to remove elements
        A = local_search_remove(A, rng, time_limit=1.0)
        
        # Update best
        if len(A) < best_size:
            best_size = len(A)
            best_set = list(A)
            # If we found a very small set, we can be more aggressive
            if best_size <= 12:
                # Try more local search
                A = local_search_remove(A, rng, time_limit=3.0)
                if len(A) < best_size:
                    best_size = len(A)
                    best_set = list(A)
    
    # Final polish: try to improve the best set found
    if best_set:
        rng = random.Random(42)
        best_set = local_search_remove(best_set, rng, time_limit=5.0)
        if len(best_set) < best_size:
            best_size = len(best_set)
    
    return best_set
