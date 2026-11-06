import time
import random
import numpy as np
import sys

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

def construct():
    start_time = time.time()
    # Use 85s limit to be safe against the 90s kill
    deadline = start_time + 85.0
    
    n = 20
    max_val = 10000
    
    # State representation for fast incremental updates
    # A: list of 20 distinct integers
    # sum_counts: dict of sum -> count
    # prod_counts: dict of prod -> count
    # sum_size: number of unique sums
    # prod_size: number of unique products
    # objective: max(sum_size, prod_size)
    
    # Helper to initialize state
    def init_state(A):
        sum_counts = {}
        prod_counts = {}
        for i in range(n):
            ai = A[i]
            for j in range(n):
                s = ai + A[j]
                p = ai * A[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        sum_size = len(sum_counts)
        prod_size = len(prod_counts)
        obj = max(sum_size, prod_size)
        return sum_counts, prod_counts, sum_size, prod_size, obj

    # Helper to compute delta for swapping A[i] = old_val to new_val
    def compute_delta(A, i, old_val, new_val, sum_counts, prod_counts, sum_size, prod_size):
        # We need to update counts for pairs involving i.
        # Pairs: (i, j) and (j, i) for all j.
        # Note: if j == i, the pair is (i, i).
        
        # 1. Remove contributions of old_val at index i
        # 2. Add contributions of new_val at index i
        
        # We'll track changes in sum_size and prod_size
        # and changes in counts.
        
        # To avoid modifying the original dicts during calculation, 
        # we can simulate the changes or just calculate the new sizes directly?
        # Direct calculation of new sizes requires knowing which keys hit 0 or became 1.
        
        # Let's simulate the changes on temporary copies of the relevant parts? 
        # No, n=20, so 400 pairs. Updating 40 pairs (20 in, 20 out) is cheap.
        
        # Let's just do the update on the dicts and then rollback if needed? 
        # Or better: calculate the exact change in size.
        
        # Changes to sums:
        # For each j in 0..n-1:
        #   old_sum = old_val + A[j]
        #   new_sum = new_val + A[j]
        #   If j == i, we are replacing (old+old) with (new+new).
        #   If j != i, we are replacing (old+A[j]) with (new+A[j]) AND (A[j]+old) with (A[j]+new).
        #   Wait, the loop `for i, for j` adds both (i,j) and (j,i) if we iterate all pairs.
        #   My init_state iterates all i,j. So (i,j) and (j,i) are distinct entries in the loop, 
        #   but they contribute to the same sum value if addition is commutative.
        #   sum_counts[s] counts the number of pairs (i,j) such that A[i]+A[j]=s.
        
        # So, removing old_val at index i:
        #   For each j:
        #     s_old = old_val + A[j]
        #     sum_counts[s_old] -= 1
        #     if sum_counts[s_old] == 0: sum_size -= 1
        #     p_old = old_val * A[j]
        #     prod_counts[p_old] -= 1
        #     if prod_counts[p_old] == 0: prod_size -= 1
        
        # Adding new_val at index i:
        #   For each j:
        #     s_new = new_val + A[j]
        #     if sum_counts[s_new] == 0: sum_size += 1
        #     sum_counts[s_new] += 1
        #     p_new = new_val * A[j]
        #     if prod_counts[p_new] == 0: prod_size += 1
        #     prod_counts[p_new] += 1
        
        # We need to be careful with j=i.
        # When j=i:
        #   old: s = 2*old_val, p = old_val^2
        #   new: s = 2*new_val, p = new_val^2
        # This is handled correctly by the generic loop if A[i] is updated before the second phase?
        # No, we must use the OLD A[i] for the "remove" phase and NEW A[i] for the "add" phase?
        # Actually, for j=i, A[j] is the element being changed.
        # So in "remove", we use old_val. In "add", we use new_val.
        # For j != i, A[j] is constant.
        
        # Let's implement the simulation without modifying the global dicts, just tracking the net change in sizes.
        # We need to know the current counts of the specific sums/prods involved.
        
        # To be efficient and safe, let's just apply the changes to the dicts, record the size changes, 
        # and if the move is rejected, we'd have to rollback. 
        # Given the speed requirement, rollback is risky if we don't do it perfectly.
        # Alternative: Calculate new_sum_size and new_prod_size by checking the specific keys involved.
        
        # Keys involved in sums:
        # Old keys: {old_val + A[j] for j in range(n)}
        # New keys: {new_val + A[j] for j in range(n)}
        # Union of keys: K_s = Old_keys | New_keys
        # For each key in K_s, we can calculate its new count.
        # new_count(k) = old_count(k) - removed(k) + added(k)
        
        # This is O(n) per move to identify keys, and O(n) to update counts. Total O(n).
        # This is much faster than O(n^2) full recalculation.
        
        old_keys_s = set()
        new_keys_s = set()
        old_keys_p = set()
        new_keys_p = set()
        
        for j in range(n):
            aj = A[j]
            # Note: if j==i, aj is old_val in current A, but logically it's the element at i.
            # We treat A as the current state.
            
            s_old = old_val + aj
            s_new = new_val + aj
            old_keys_s.add(s_old)
            new_keys_s.add(s_new)
            
            p_old = old_val * aj
            p_new = new_val * aj
            old_keys_p.add(p_old)
            new_keys_p.add(p_new)
            
        # Calculate new sum_size
        # We need to know which keys in old_keys_s drop to 0, and which in new_keys_s rise from 0.
        # Actually, we can just compute the delta.
        
        # Let's just do the dict update and track size. It's only 40 operations.
        # To allow rollback, we'd need to store the old counts of the affected keys.
        # Affected keys: old_keys_s, new_keys_s.
        
        # Let's try a different approach: 
        # Since n=20, full recalculation is 400 additions and 400 multiplications + set updates.
        # Python sets are fast. 400 ops is ~0.01ms?
        # 10,000 moves * 0.01ms = 100ms. 
        # 100,000 moves * 0.01ms = 1s.
        # We have 85s. We can do ~100,000 - 1,000,000 moves.
        # Full recalculation is simpler and less error-prone.
        # Let's stick to full recalculation but optimize the Python loop.
        
        # Actually, let's use the incremental method properly. It's not that hard.
        pass

    # Let's go with the full recalculation but using a faster method if possible.
    # Or just standard Python. 20x20=400 pairs.
    # Let's benchmark mentally:
    # _score(A) takes ~1-2 ms in Python.
    # 85s / 2ms = 42,500 moves.
    # That's a decent number of SA steps.
    
    # Strategy:
    # 1. Generate diverse initial states.
    #    - Geometric progressions
    #    - Arithmetic progressions
    #    - Random clusters
    # 2. Run SA for each, keeping the global best.
    
    best_A = None
    best_score = float('inf')
    
    # Pre-define some good starting structures
    def generate_starts():
        starts = []
        
        # 1. Geometric Progressions
        for r in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
            A = []
            curr = 1.0
            for _ in range(n):
                v = int(round(curr))
                if v < 1: v = 1
                if v > max_val: break
                if v not in A:
                    A.append(v)
                curr *= r
            if len(A) == n:
                starts.append(A[:])
            else:
                # Fill gaps
                while len(A) < n:
                    A.append(random.randint(A[-1]+1, max_val))
                # Ensure distinct
                A = list(set(A))
                if len(A) == n:
                    starts.append(A[:])
        
        # 2. Arithmetic Progressions
        for d in [2, 3, 5, 10, 20, 50]:
            A = [1 + i*d for i in range(n)]
            if A[-1] <= max_val:
                starts.append(A[:])
        
        # 3. Random
        for _ in range(10):
            A = random.sample(range(1, max_val+1), n)
            starts.append(A[:])
            
        # 4. "Log-uniform" random
        for _ in range(10):
            logs = np.sort(np.random.uniform(0, np.log(max_val), n))
            A = list(np.exp(logs).astype(int))
            # Ensure distinct
            A = list(set(A))
            if len(A) < n:
                 # Fill
                 existing = set(A)
                 while len(A) < n:
                     v = random.randint(1, max_val)
                     if v not in existing:
                         A.append(v)
                         existing.add(v)
            starts.append(sorted(A)[:n])
            
        return starts

    starts = generate_starts()
    random.shuffle(starts)
    
    # SA Parameters
    # We will run SA on each start until time runs out or convergence.
    # Since we have many starts, we should allocate time evenly or use a global time check.
    
    for start_A in starts:
        if time.time() > deadline - 1.0:
            break
            
        current_A = start_A[:]
        current_score = _score(current_A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
        # SA Loop
        T = 100.0  # Initial temperature
        T_min = 1.0
        cooling = 0.995
        
        # We want to use up to ~1-2 seconds per start if we have many, 
        # or longer if few. 
        # Let's say we run for 5000 steps or 1 second, whichever comes first.
        steps_this_run = 0
        run_start = time.time()
        
        while steps_this_run < 5000 and (time.time() - run_start) < 1.5:
            if time.time() > deadline - 0.5:
                break
                
            i = random.randint(0, n-1)
            old_val = current_A[i]
            
            # Choose new value
            # Mix of local (small delta) and global (random) moves
            if random.random() < 0.7:
                # Local move: +/- 10% or +/- 10
                delta_range = max(10, int(old_val * 0.1))
                new_val = old_val + random.randint(-delta_range, delta_range)
            else:
                # Global move
                new_val = random.randint(1, max_val)
                
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in current_A and new_val != old_val:
                continue
                
            new_A = current_A[:]
            new_A[i] = new_val
            new_A.sort()
            
            new_score = _score(new_A)
            delta = new_score - current_score
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                current_A = new_A
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
                    
            T *= cooling
            steps_this_run += 1
            
    # Final check
    if best_A is None:
        best_A = list(range(1, n+1))
        
    return sorted(best_A)

if __name__ == "__main__":
    # Simple self-test
    A = construct()
    print(f"Result: {A}")
    print(f"Score: {_score(A)}")
