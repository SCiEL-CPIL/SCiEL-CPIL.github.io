import time
import random
import numpy as np
from collections import Counter

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _fast_delta_score(A, idx, new_val, sum_counter, prod_counter, sum_set, prod_set):
    """
    Calculate the new score if A[idx] is replaced by new_val.
    Uses pre-computed counters of other elements to update set sizes in O(n).
    Returns: (new_max_size, updated_sum_counter, updated_prod_counter, updated_sum_set, updated_prod_set)
    """
    old_val = A[idx]
    if new_val == old_val:
        return max(len(sum_set), len(prod_set)), sum_counter, prod_counter, sum_set, prod_set

    n = len(A)
    
    # --- Update Sum Set ---
    # Elements that will be removed from sum_set:
    # old_val + A[j] for all j != idx
    # Note: If A[j] == old_val, we only have one such j (since distinct), so we remove it once.
    # But wait, A is a set of distinct integers. So A[j] != old_val for all j != idx.
    # So we remove {old_val + A[j] | j != idx}.
    
    # Elements that will be added to sum_set:
    # new_val + A[j] for all j != idx
    # Plus new_val + new_val (the diagonal for the new element)
    # Wait, the original sum_set contains old_val + old_val.
    # The new sum_set should contain new_val + new_val.
    
    # Let's do it carefully.
    # Current sum_set contains all a_i + a_j for i,j in [0, n).
    # We remove pairs involving old_val at index idx.
    # Pairs: (idx, j) and (j, idx) for all j.
    # Since addition is commutative, the set of sums involving old_val is {old_val + A[j] for j in [0, n)}.
    # But we are replacing A[idx].
    # The sums that disappear are those where one operand is old_val.
    # Specifically: {old_val + A[j] for j in [0, n)}.
    # However, if there are other elements equal to old_val? No, distinct.
    # So we decrement counts for these n sums.
    
    new_sum_counter = sum_counter.copy()
    for j in range(n):
        s = old_val + A[j]
        new_sum_counter[s] -= 1
        if new_sum_counter[s] == 0:
            del new_sum_counter[s]

    # Add new sums involving new_val.
    # {new_val + A[j] for j in [0, n)}
    # Note: A[idx] is still old_val in the array A during this calculation?
    # We are simulating the change. The other elements A[j] for j!=idx remain.
    # The element at idx becomes new_val.
    # So the new sums are {new_val + A[j] for j != idx} U {new_val + new_val}.
    # Which is exactly {new_val + A[j] for j in [0, n)} if we treat A[idx] as new_val?
    # No, A[j] for j!=idx are unchanged.
    # So we add {new_val + A[j] for j in [0, n)}?
    # Wait. The set of sums in the new state is:
    # {A[i] + A[j] for i,j in [0, n)} with A[idx]=new_val.
    # This set is composed of:
    # 1. Sums not involving index idx: {A[i] + A[j] for i,j != idx}. These were in the old set and are not removed.
    # 2. Sums involving index idx: {new_val + A[j] for j != idx} U {new_val + new_val}.
    #    This is {new_val + A[j] for j in [0, n)} if we consider that A[idx] in the "other" slot is new_val?
    #    Actually, {new_val + A[j] for j in [0, n)} includes new_val + new_val (when j=idx, if we imagine A[idx] updated) 
    #    and new_val + A[j] for j!=idx.
    #    So yes, we simply add {new_val + A[j] for j in [0, n)}.
    
    for j in range(n):
        s = new_val + A[j]
        if s not in new_sum_counter:
            new_sum_counter[s] = 0
        new_sum_counter[s] += 1

    # --- Update Product Set ---
    # Similar logic.
    new_prod_counter = prod_counter.copy()
    for j in range(n):
        p = old_val * A[j]
        new_prod_counter[p] -= 1
        if new_prod_counter[p] == 0:
            del new_prod_counter[p]

    for j in range(n):
        p = new_val * A[j]
        if p not in new_prod_counter:
            new_prod_counter[p] = 0
        new_prod_counter[p] += 1

    new_sum_size = len(new_sum_counter)
    new_prod_size = len(new_prod_counter)
    new_max = max(new_sum_size, new_prod_size)
    
    # We don't need to update the sets themselves for the score, just the counters.
    # But for the next step, we need the counters to be accurate.
    # The sets sum_set and prod_set are not used in the delta calculation, only their lengths were.
    # We can just return the new counters. The "sets" can be reconstructed if needed, but we only track counters.
    # Actually, the function signature asked for sets, but we can just return counters and let the caller manage.
    # Let's return the new counters.
    
    return new_max, new_sum_counter, new_prod_counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Initialize with a "Logarithmic Lattice" structure.
    #    A = { floor(base^k) } or similar.
    #    This minimizes product set size because products are roughly powers of base.
    #    Then use SA to optimize sum set and break product symmetries if beneficial.
    
    # Try several different "bases" or structures for initialization.
    # Structure 1: Powers of a base b.
    # Structure 2: Geometric progression with small integer ratio.
    # Structure 3: Random but log-spaced.
    
    structures = []
    
    # 1. Pure GP
    for r in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
        A = []
        cur = 1.0
        for _ in range(20):
            v = int(round(cur))
            if v in A:
                v += 1
            if v > 10000:
                break
            A.append(v)
            cur *= r
        if len(A) < 20:
            # Fill remaining
            last = A[-1]
            while len(A) < 20:
                v = int(last * r) + rng.randint(0, 5)
                if v not in A and v <= 10000:
                    A.append(v)
                else:
                    v += 1
                    if v > 10000:
                        break
            if len(A) < 20:
                continue
        structures.append(A[:20])
        
    # 2. Log-spaced integers
    # Map [0, 19] to [log(1), log(10000)]
    for scale in [0.9, 1.0, 1.1]:
        A = []
        for i in range(20):
            # t in [0, 1]
            t = i / 19.0
            # log value
            log_v = np.log(10000) * t * scale
            v = int(np.exp(log_v))
            if v < 1: v = 1
            if v > 10000: v = 10000
            if v in A:
                v += 1
            A.append(v)
        if len(set(A)) == 20:
            structures.append(sorted(A))

    # 3. Random Log-Uniform
    for _ in range(10):
        A = []
        for _ in range(20):
            v = int(10 ** rng.uniform(0, 4))
            if v < 1: v = 1
            if v > 10000: v = 10000
            while v in A:
                v += 1
                if v > 10000:
                    v = 1
            A.append(v)
        structures.append(sorted(A))

    # Run SA on each structure
    for A_init in structures:
        if time.time() > deadline:
            break
            
        A = A_init[:]
        A.sort()
        
        # Pre-compute counters for initial state
        sum_counter = Counter()
        prod_counter = Counter()
        for i in range(20):
            for j in range(20):
                sum_counter[A[i] + A[j]] += 1
                prod_counter[A[i] * A[j]] += 1
        
        current_score = max(len(sum_counter), len(prod_counter))
        
        if current_score < best_score:
            best_score = current_score
            best_A = A[:]
            
        # Simulated Annealing
        T = 100.0
        T_min = 1.0
        cooling = 0.9995
        
        # Determine number of steps based on time remaining
        # We want to use most of the time.
        # Estimate time per step:
        # Delta score is O(n) = O(20) operations. Very fast.
        # Let's run until time limit.
        
        while time.time() < deadline - 1.0:
            if T < T_min:
                break
                
            i = rng.randint(0, 19)
            old_val = A[i]
            
            # Generate new value
            # Strategy: Mix of local perturbations and global jumps
            if rng.random() < 0.7:
                # Local move
                # Bias towards keeping log-structure?
                # Just random nearby value
                delta = rng.randint(-max(1, old_val//10), max(1, old_val//10))
                new_val = old_val + delta
            else:
                # Global jump: random value in [1, 10000]
                # Or jump to a value that preserves log-spacing roughly?
                new_val = rng.randint(1, 10000)
            
            if new_val < 1: new_val = 1
            if new_val > 10000: new_val = 10000
            if new_val == old_val:
                continue
            if new_val in A:
                # Check if it's the same index
                if A.index(new_val) != i:
                    continue
            
            # Calculate delta score
            new_score, new_sum_counter, new_prod_counter = _fast_delta_score(
                A, i, new_val, sum_counter, prod_counter, None, None
            )
            
            delta_s = new_score - current_score
            
            if delta_s < 0 or rng.random() < np.exp(-delta_s / T):
                # Accept
                A[i] = new_val
                A.sort() # Keep sorted for consistency, though not strictly required for set logic
                sum_counter = new_sum_counter
                prod_counter = new_prod_counter
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            
            T *= cooling
            
            # If T is too low, we might get stuck. 
            # Simple restart if stuck for a while? 
            # For now, just cool down.
            
    if best_A is None:
        best_A = list(range(1, 21))
        
    return sorted(best_A)

# Test/Run
if __name__ == "__main__":
    # Quick self-test
    A = construct()
    print(f"Result: {A}")
    print(f"Score: {_score(A)}")
