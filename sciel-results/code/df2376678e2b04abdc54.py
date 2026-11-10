import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s

    N = 51975

    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()
    
    # Precompute indices for each divisor? 
    # No, indices depend on residue. But we can precompute the "step" arrays?
    # Just using range/r::m is fine for N=51975.

    def greedy_init(moduli_list):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                # Count uncovered in this class
                # covered[r::m] gives the slice
                cnt = int(np.sum(covered[r::m] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
            
        uncovered = int(np.sum(covered == 0))
        return classes, uncovered

    def local_search(classes, score, time_limit):
        """
        Local search using cover_count for efficient updates.
        """
        classes = [c[:] for c in classes]
        score = score
        
        # Initialize cover_count
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
            
        # Verify score
        # score should match sum(cover_count == 0)
        # We trust the input score but can verify in debug.
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = score
                
                # We need to evaluate all new_r
                # To do this efficiently, we can't just recompute the whole array every time.
                # We will simulate the change.
                
                # Get the indices for the current class
                # Note: r::m is a view, but we need to modify cover_count.
                # We will calculate the delta for each candidate new_r.
                
                # Optimization:
                # Let S_old be indices of current class.
                # Let S_new be indices of candidate class.
                # We want to know the change in number of zeros in cover_count
                # if we do: cover_count[S_old] -= 1, cover_count[S_new] += 1.
                
                # Let's just iterate candidates.
                # For each candidate new_r:
                #   1. Identify S_new = np.arange(new_r, N, m)
                #   2. Identify S_old = np.arange(r, N, m)
                #   3. The union of indices affected is S_old ∪ S_new.
                #   4. We can compute the new score by looking at cover_count at these indices.
                
                # Since m can be up to N, S_old size is ~N/m.
                # Iterating m candidates means O(N) work per class per pass.
                # With ~48 classes, one pass is ~48*N ~ 2.5M ops.
                # This is fast enough.
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    # We need to calculate the new uncovered count.
                    # Instead of modifying cover_count in place and reverting (which is error prone with overlaps),
                    # we can calculate the change.
                    
                    # Let's use a helper to compute score change.
                    # This is tricky because of overlaps between S_old and S_new.
                    
                    # Alternative: Just update cover_count, compute score, revert.
                    # Update:
                    #   cover_count[r::m] -= 1
                    #   cover_count[new_r::m] += 1
                    #   new_score = np.sum(cover_count == 0)
                    #   Revert:
                    #   cover_count[new_r::m] -= 1
                    #   cover_count[r::m] += 1
                    
                    # This is O(N) for the sum. Doing it m times is O(N*m) = O(N^2/m).
                    # For small m (e.g., 3), this is large.
                    # For large m (e.g., 51975), this is small.
                    # Total cost is sum(N^2/m_i) for all classes.
                    # Sum of 1/m_i for divisors of N is not huge.
                    # However, np.sum(cover_count == 0) is O(N).
                    # So total cost per pass is sum(m_i * N) = N * sum(m_i).
                    # Sum of divisors of 51975.
                    # 51975 = 3^3 * 5^2 * 7 * 11.
                    # Sum of divisors sigma(N) = (1+3+9+27)(1+5+25)(1+7)(1+11)
                    # = 40 * 31 * 8 * 12 = 121600.
                    # So one pass is ~ N * 121600 ~ 51975 * 121600 ~ 6.3 * 10^9 operations?
                    # No, that's too slow.
                    
                    # We need a better way to compute the change in score.
                    
                    # Let's stick to the "delta" method.
                    # Let U be the set of indices where cover_count == 0.
                    # We modify cover_count at S_old and S_new.
                    # Let A = S_old, B = S_new.
                    # Indices in A \ B: cover_count decreases by 1.
                    #   If it was 1, it becomes 0 (U increases by 1).
                    #   If it was > 1, it stays > 0 (U unchanged).
                    # Indices in B \ A: cover_count increases by 1.
                    #   If it was 0, it becomes 1 (U decreases by 1).
                    #   If it was >= 1, it stays >= 1 (U unchanged).
                    # Indices in A ∩ B: cover_count unchanged.
                    
                    # So, delta_U = (count of i in A\B where cover_count[i]==1) - (count of i in B\A where cover_count[i]==0)
                    
                    # We can compute this using numpy.
                    # A_indices = np.arange(r, N, m)
                    # B_indices = np.arange(new_r, N, m)
                    
                    # We need to be careful with memory and speed.
                    # Creating A_indices and B_indices is O(N/m).
                    
                    # Let's try this approach.
                    
                    A_indices = np.arange(r, N, m)
                    B_indices = np.arange(new_r, N, m)
                    
                    # Find A \ B and B \ A
                    # Since they are arithmetic progressions with same step, 
                    # A \ B are those in A not in B.
                    # It's easier to just check membership? No, that's slow.
                    
                    # Alternative:
                    # Just update cover_count, compute sum, revert.
                    # If we optimize the sum calculation:
                    # We only care about indices in A ∪ B.
                    # Let Union = np.unique(np.concatenate([A_indices, B_indices]))
                    # Old values at Union: old_vals = cover_count[Union]
                    # New values:
                    #   For i in A: new_val = old_val - 1 (unless also in B, then -1+1=0)
                    #   For i in B: new_val = old_val + 1 (unless also in A, then +1-1=0)
                    
                    # This is getting complicated to vectorize perfectly without explicit sets.
                    
                    # Given N is small (51k), let's just do the explicit update/revert and use a fast sum.
                    # np.sum(cover_count == 0) is the bottleneck.
                    # Can we maintain a count of zeros?
                    # Yes!
                    
                    # Let's refactor to maintain `uncovered_count` explicitly.
                    
                    # Re-initializing cover_count and uncovered_count for local search
                    # (Already done above)
                    
                    # Let's implement the delta logic properly.
                    
                    # We will compute the change in uncovered count.
                    # We need to know cover_count at A and B.
                    
                    # Let's just do the update/revert and use np.sum.
                    # 6.3 * 10^9 simple ops in Python is too slow.
                    # But numpy vectorized ops are fast.
                    # cover_count[r::m] -= 1 is vectorized.
                    # np.sum(cover_count == 0) is vectorized.
                    # The overhead is Python loop over m.
                    # For m=3, 3 iterations. For m=51975, 51975 iterations.
                    # The large m cases have small N/m.
                    # The small m cases have few iterations.
                    # Total numpy operations: sum(m_i) * (2 updates + 1 sum).
                    # sum(m_i) ~ 121600.
                    # 121600 * 3 numpy calls.
                    # Each numpy call on an array of size ~50k is fast (~microseconds).
                    # 360,000 * 10us = 3.6 seconds per pass.
                    # This is feasible for a few passes.
                    
                    # Let's try it.
                    
                    cover_count[r::m] -= 1
                    cover_count[new_r::m] += 1
                    new_score = int(np.sum(cover_count == 0))
                    cover_count[new_r::m] -= 1
                    cover_count[r::m] += 1
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Apply the best change permanently
                    cover_count[r::m] -= 1
                    cover_count[best_r::m] += 1
                    classes[i] = [m, best_r]
                    score = best_score
                    improved = True

        return classes, score

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score = greedy_init(divisors)

    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 5)

    # Strategy 2: Try different orderings
    random.seed(42)
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        # Try greedy with shuffled order
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2 = greedy_init(shuffled)
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3)

    if (time.time() - start_time) < deadline - 15:
        # Try with larger moduli first
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3 = greedy_init(divs_desc)
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3)

    # Strategy 3: Simulated Annealing
    # We need a fast way to evaluate perturbations.
    # We will reuse the cover_count array from the last local search?
    # No, let's initialize it from best_classes.
    
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in current_classes:
            cover_count[r::m] += 1
            
        temperature = max(1, best_score // 5)
        
        for iteration in range(500):
            if (time.time() - start_time) > deadline - 5:
                break

            # Perturb: change one residue
            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            if new_r == old_r:
                continue

            # Evaluate
            cover_count[old_r::m] -= 1
            cover_count[new_r::m] += 1
            new_score = int(np.sum(cover_count == 0))
            
            delta = new_score - current_score

            if delta < 0 or random.random() < (1.0 / (1.0 + delta / max(1, temperature))):
                current_classes[idx] = [m, new_r]
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
                    # Run local search from new best
                    if (time.time() - start_time) < deadline - 3:
                        best_classes, best_score = local_search(best_classes, best_score, deadline - 2)
                        # Re-sync cover_count with best_classes if we returned from local search
                        # local_search returns classes and score, but doesn't return cover_count.
                        # We should re-initialize cover_count if we do local search.
                        cover_count = np.zeros(N, dtype=np.int32)
                        for mm, rr in best_classes:
                            cover_count[rr::mm] += 1
                        current_classes = [c[:] for c in best_classes]
                        current_score = best_score
            else:
                # Revert
                cover_count[new_r::m] -= 1
                cover_count[old_r::m] += 1

            # Cool down
            temperature = max(1, temperature * 0.95)

    # Final local search if time permits
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 1)

    return best_classes
