import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 75  # seconds, safety margin before 90s kill

def construct():
    """Find subset A of [0,128] covering [128,256] by pair sums, minimizing peak multiplicity."""
    n = 128
    target_lo, target_hi = 128, 256
    
    # Start with a valid baseline: A = [64, 65, ..., 128]
    # This covers [128, 256] since min sum = 64+64=128, max sum = 128+128=256
    A = np.zeros(n + 1, dtype=bool)
    A[64:129] = True
    
    def compute_sums(A_bool):
        """Compute pair sum multiplicities for unordered pairs including equal summands."""
        # For unordered pairs with equal summands allowed:
        # count[s] = sum over a <= b, a+b=s of 1
        # Using convolution: if we let c = A_bool (as 0/1), then
        # the ordered convolution gives ordered pairs. For unordered with self-pairs:
        # unordered_count[s] = (ordered_conv[s] + (s even and A[s//2])) / 2
        # But actually, let's think carefully:
        # Ordered pairs: (a,b) and (b,a) are distinct if a!=b, same if a==b
        # Unordered pairs with self: each pair {a,b} with a<=b counts once
        # Number of unordered representations of s = sum_{a=0}^{s} A[a]*A[s-a] but divide by 2 for a!=b, keep for a==b
        # = (sum_{a} A[a]*A[s-a] + A[s/2] if s even) / 2
        # Let's use numpy convolution for the ordered count
        A_int = A_bool.astype(np.int32)
        ordered = np.convolve(A_int, A_int, mode='full')
        # ordered[s] = sum_{a+b=s} A[a]*A[b] (ordered, counts both (a,b) and (b,a) when a!=b)
        # For unordered with self: 
        #   if s is even and A[s//2] is True, the term (s//2, s//2) is counted once in ordered
        #   all other terms come in pairs
        # So unordered[s] = (ordered[s] + (1 if s%2==0 and A[s//2] else 0)) // 2
        # Wait, that's not right either. Let me reconsider.
        # ordered[s] = sum_{a=0}^{s} A[a]*A[s-a]
        # This counts (a,b) for all a,b with a+b=s.
        # For unordered pairs {a,b} with a<=b:
        #   If a < b: contributes 2 to ordered (once as (a,b), once as (b,a))
        #   If a == b: contributes 1 to ordered
        # So unordered[s] = (ordered[s] + self_count[s]) / 2 where self_count[s] = 1 if s even and A[s//2] else 0
        # Because: ordered = 2*(#pairs with a<b) + (#pairs with a==b)
        #          unordered = (#pairs with a<b) + (#pairs with a==b)
        #          So unordered = (ordered + self_count) / 2
        self_count = np.zeros(len(ordered), dtype=np.int32)
        for s in range(target_lo, target_hi + 1):
            if s % 2 == 0 and s // 2 <= n and A_bool[s // 2]:
                self_count[s] = 1
        unordered = (ordered + self_count) // 2
        return unordered
    
    def get_cost(A_bool):
        """Return (peak_mult, sum_of_squares) for the target interval."""
        sums = compute_sums(A_bool)
        target_sums = sums[target_lo:target_hi + 1]
        # Check validity: all must be >= 1
        if np.any(target_sums < 1):
            return (float('inf'), float('inf'))
        peak = int(np.max(target_sums))
        sq = int(np.sum(target_sums ** 2))
        return (peak, sq)
    
    # Initial cost
    init_cost = get_cost(A)
    best_A = A.copy()
    best_cost = init_cost
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - _START_TIME < _DEADLINE and T > T_min:
        # Random swap: remove one element, add another
        current_elements = np.where(A)[0]
        if len(current_elements) < 2:
            break
        
        # Try a few swaps per temperature step
        for _ in range(50):
            if time.time() - _START_TIME > _DEADLINE:
                break
            
            # Pick a random element to remove
            rem_idx = np.random.randint(0, len(current_elements))
            rem_val = current_elements[rem_idx]
            
            # Pick a random value to add (not currently in A)
            candidates = np.where(~A)[0]
            if len(candidates) == 0:
                break
            add_val = candidates[np.random.randint(0, len(candidates))]
            
            # Apply swap
            A[rem_val] = False
            A[add_val] = True
            
            new_cost = get_cost(A)
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if new_cost <= best_cost or (new_cost[0] < float('inf') and 
               np.random.random() < np.exp(-(new_cost[0] - init_cost[0]) / T)):
                # Accept
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = A.copy()
                init_cost = new_cost
            else:
                # Reject, revert
                A[add_val] = False
                A[rem_val] = True
        
        T *= cooling
    
    # Return the best found as a list
    return list(np.where(best_A)[0])
