"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def fast_check(x, S_set):
        """Check if x can be added to S_set without violating sum-free."""
        # We need to ensure no a, b in S_set such that a+b=x, and no a in S_set such that a+x in S_set.
        # Also 2x not in S_set.
        # 1. a+b=x => b = x-a. Check if (x-a) in S_set for any a in S_set?
        #    Actually, we can iterate over S_set.
        #    If x-a in S_set, then a + (x-a) = x. Conflict.
        # 2. a+x=c => c = a+x. Check if (a+x) in S_set for any a in S_set.
        # 3. 2x=c => Check if 2x in S_set.
        
        # Optimization: 
        # For condition 1: We need to check if there exists a in S_set such that (x-a) in S_set.
        # This is equivalent to checking if S_set intersects (x - S_set).
        # For condition 2: We need to check if there exists a in S_set such that (a+x) in S_set.
        # This is equivalent to checking if S_set intersects (-x + S_set) ? No.
        # a+x in S_set => a in S_set - x. So check if S_set intersects (S_set - x).
        
        # Let's just iterate. n is small (<=240).
        for a in S_set:
            if (x - a) in S_set:
                return False
            if (x + a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def local_search(elements, max_time=30.0, rng=None):
        """
        Local search to maximize size of sum-free subset.
        Uses a conflict graph representation for efficient swaps.
        """
        if rng is None:
            rng = Random(42)
            
        n = len(elements)
        if n == 0:
            return []
        
        # Precompute conflict relationships?
        # Two elements x, y are "compatible" if they can coexist in a sum-free set?
        # No, compatibility is not pairwise. {1, 2, 3} is not sum-free, but {1,2} is and {2,3} is.
        # So we cannot simply use a pairwise conflict graph for the set itself.
        # However, we can use the following:
        # A set S is sum-free iff for all a, b in S, a+b not in S.
        
        # Strategy: 
        # 1. Initialize with a random or greedy set.
        # 2. Try to add elements from the remaining pool.
        # 3. If stuck, try to remove one element from S to allow adding two from the pool.
        #    Or remove one to allow adding one (swap).
        
        # Greedy Initialization
        # Sort by absolute value descending (heuristic: larger numbers are less likely to be sums)
        # Or sort by "degree" in a hypothetical conflict graph.
        # Let's try sorting by absolute value descending.
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        
        S = []
        S_set = set()
        
        for x in sorted_elems:
            if fast_check(x, S_set):
                S.append(x)
                S_set.add(x)
                
        remaining = [x for x in elements if x not in S_set]
        
        # Local Search Phase
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # 1. Try to add any remaining element
            # Shuffle remaining to avoid bias
            rng.shuffle(remaining)
            for i, x in enumerate(remaining):
                if time.time() > max_time:
                    break
                if fast_check(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    remaining.pop(i)
                    improved = True
                    break # Restart loop to re-shuffle or continue? Breaking to restart is safer for time check.
            
            if not improved and time.time() < max_time - 1.0:
                # 2. Try swaps: remove one from S, add one from remaining
                # This is expensive. Let's try a limited number of random swaps.
                # Or try to find a "bottleneck" element in S that blocks many elements in remaining.
                
                # Heuristic: Calculate "blocking score" for each element in S.
                # An element s in S blocks x in remaining if x cannot be added because of s.
                # Specifically, if x-s in S_set or x+s in S_set or 2x in S_set (and 2x==s? no).
                # Actually, x is blocked if fast_check(x, S_set) is False.
                # We can check which s in S is responsible.
                
                # Let's just do random restarts of the local search phase if we get stuck?
                # Or try 1-for-2 swaps.
                
                # Try 1-for-2: Remove s from S. Can we add two elements from remaining?
                # This is O(|S| * |remaining|^2 * check_cost).
                # |S| ~ 80, |remaining| ~ 160. 80 * 160^2 = 2M. Too slow per iteration.
                
                # Let's try a simpler swap: 1-for-1.
                # Remove s, add x.
                # Check if x can be added to S \ {s}.
                # If yes, and x was previously blocked by s, this is a valid swap (size unchanged, but might enable future adds).
                # But we want to INCREASE size. So 1-for-1 doesn't help immediately unless it enables a 2nd add.
                
                # Let's stick to the "add if possible" loop. If we are stuck, it means no single element can be added.
                # This is a local optimum for single-add moves.
                # To escape, we need multi-step moves.
                
                # Let's try a "kick" move: Remove a random element from S, and then greedily add as many as possible.
                # If the resulting set is larger, accept it.
                
                if S and remaining:
                    # Pick a random element to remove
                    idx_to_remove = rng.randint(0, len(S)-1)
                    s_removed = S[idx_to_remove]
                    
                    # Temporarily remove
                    S.pop(idx_to_remove)
                    S_set.remove(s_removed)
                    
                    # Greedily add from remaining
                    # We need to check all remaining elements
                    added_any = False
                    new_remaining = list(remaining)
                    rng.shuffle(new_remaining)
                    
                    for x in new_remaining:
                        if fast_check(x, S_set):
                            S.append(x)
                            S_set.add(x)
                            added_any = True
                            # Note: we don't remove from 'remaining' list permanently yet, 
                            # we just want to see if this kick leads to a larger set.
                            # Actually, if we add x, it might block others.
                            # This is a local search step.
                    
                    # Compare size
                    # We don't easily know the size of the set after greedy add without tracking it.
                    # Let's track it.
                    
                    # This approach is messy. Let's use a cleaner local search.
                    
                    # Revert
                    S.append(s_removed)
                    S_set.add(s_removed)
                    
                    # Try a proper 1-for-1 swap that increases size? Impossible.
                    # 1-for-2 swap:
                    # Pick s in S. Remove it.
                    # Find two elements x, y in remaining that can be added to S\{s}.
                    # And x, y must be compatible with each other and with S\{s}.
                    
                    # This is hard to do efficiently.
                    
                    # Let's try a different initialization strategy: Randomized Greedy.
                    # Run the greedy algorithm multiple times with different orderings and keep the best.
                    
                    pass

        return S

    def multi_start_greedy(elements, max_time=30.0, rng=None):
        """
        Run greedy construction multiple times with different orderings and return the best result.
        """
        if rng is None:
            rng = Random(42)
            
        n = len(elements)
        if n == 0:
            return []
        
        best_S = []
        best_size = 0
        
        # Strategy 1: Sort by abs value descending
        # Strategy 2: Sort by abs value ascending
        # Strategy 3: Random order
        # Strategy 4: Sort by "degree" in sum-graph (how many pairs sum to it, or how many it sums to with others)
        
        # Precompute a heuristic score for each element
        # Score: number of conflicts if added to an empty set? No.
        # Score: number of elements x such that x + y is in the set?
        
        elem_set = set(elements)
        scores = {}
        for x in elements:
            count = 0
            for y in elements:
                if x != y and (x + y) in elem_set:
                    count += 1
            scores[x] = count
            
        # Elements with low scores are "safer" to pick early?
        # Or elements with high scores are "more central" and should be picked or avoided?
        # If we pick an element with high score, it might block many others.
        # So maybe sort by score ascending (pick safe ones first)?
        
        strategies = [
            lambda: sorted(elements, key=lambda x: abs(x), reverse=True),
            lambda: sorted(elements, key=lambda x: abs(x)),
            lambda: sorted(elements, key=lambda x: scores[x]),
            lambda: sorted(elements, key=lambda x: scores[x], reverse=True),
            lambda: list(rng.sample(elements, n)),
            lambda: list(rng.sample(elements, n)),
            lambda: list(rng.sample(elements, n)),
            lambda: list(rng.sample(elements, n)),
        ]
        
        for i, strat in enumerate(strategies):
            if time.time() > max_time - 1.0:
                break
                
            order = strat()
            S = []
            S_set = set()
            
            for x in order:
                if fast_check(x, S_set):
                    S.append(x)
                    S_set.add(x)
            
            # Local improvement: try to add remaining elements
            remaining = [x for x in elements if x not in S_set]
            rng.shuffle(remaining)
            
            improved = True
            while improved and time.time() < max_time - 1.0:
                improved = False
                for i_rem, x in enumerate(remaining):
                    if time.time() > max_time - 1.0:
                        break
                    if fast_check(x, S_set):
                        S.append(x)
                        S_set.add(x)
                        remaining.pop(i_rem)
                        improved = True
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
                
        return best_S

    def spectral_greedy(elements, max_time=30.0):
        """
        Use spectral methods to order elements for greedy construction.
        """
        n = len(elements)
        if n < 5:
            return multi_start_greedy(elements, max_time)
        
        elem_set = set(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Build adjacency matrix A where A[i][j] = 1 if elements[i] + elements[j] in elem_set
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi = elements[i]
            for j in range(i, n):
                xj = elements[j]
                if xi + xj in elem_set:
                    A[i][j] = 1
                    if i != j:
                        A[j][i] = 1
        
        # Compute Laplacian
        D = np.sum(A, axis=1)
        L = np.diag(D) - A
        
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Find Fiedler vector (smallest non-zero eigenvalue)
            tol = 1e-8
            fiedler_idx = 1
            if abs(eigvals[0]) < tol:
                if abs(eigvals[1]) < tol:
                    for k in range(len(eigvals)):
                        if abs(eigvals[k]) > tol:
                            fiedler_idx = k
                            break
                else:
                    fiedler_idx = 1
            else:
                fiedler_idx = 0
                
            v = eigvecs[:, fiedler_idx]
            
            # Sort elements by spectral value descending
            sorted_elems = sorted(elements, key=lambda x: v[val_to_idx[x]], reverse=True)
            
            # Run greedy on this order
            S = []
            S_set = set()
            for x in sorted_elems:
                if fast_check(x, S_set):
                    S.append(x)
                    S_set.add(x)
            
            # Local improvement
            remaining = [x for x in elements if x not in S_set]
            rng_local = Random(42)
            rng_local.shuffle(remaining)
            
            improved = True
            while improved and time.time() < max_time - 1.0:
                improved = False
                for i_rem, x in enumerate(remaining):
                    if time.time() > max_time - 1.0:
                        break
                    if fast_check(x, S_set):
                        S.append(x)
                        S_set.add(x)
                        remaining.pop(i_rem)
                        improved = True
                        break
                        
            return S
            
        except Exception:
            return multi_start_greedy(elements, max_time)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # Allocate time
        # Larger instances need more time for local search
        time_budget = (size / 420.0) * 100.0
        remaining_time = deadline - time.time()
        if time_budget > remaining_time:
            time_budget = max(1.0, remaining_time - 1.0)
            
        # Use multi-start greedy as it is robust and fast
        # Spectral is also good, let's try both and pick the best
        t_start = time.time()
        
        # Run multi-start greedy with a portion of the time
        t_multi = time_budget * 0.5
        S1 = multi_start_greedy(elems, max_time=t_start + t_multi)
        
        # Run spectral greedy with the rest
        t_spec = time_budget * 0.5
        S2 = spectral_greedy(elems, max_time=t_start + t_spec)
        
        # Pick the best
        if len(S1) >= len(S2):
            subset = S1
        else:
            subset = S2
            
        # Final verification
        if not is_sum_free(subset):
             # Fallback
             subset = multi_start_greedy(elems, max_time=time.time() + 5.0)
             
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
