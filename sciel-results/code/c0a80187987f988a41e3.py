"""Ternary covering code K_3(7,1) - Robust Greedy + Local Search."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy for other parameters to ensure validity
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # For general case, return a valid (though large) code: all words
        # This ensures validity. For specific small cases we can do better.
        # But since the task is specific to 3,7,1, we focus there.
        # For safety, return full space if we don't have a specific heuristic.
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    # For q=3, n=7, radius=1:
    # Each word has 1 + 7*2 = 15 neighbors (including itself)
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set Cover ----------
    # Start with no codewords.
    # At each step, pick the codeword that covers the most uncovered points.
    # This guarantees a valid covering code.
    
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # To speed up greedy, we can use a heuristic ordering or just standard greedy.
    # Standard greedy is O(N * K * N) which is 2187 * 186 * 2187 ~ 1e9 ops, too slow in Python.
    # We need a faster greedy.
    
    # Optimization: 
    # 1. Precompute for each point, the list of codewords that cover it (inverse neighbors).
    #    Actually, we just need to pick the best codeword from the set of candidates 
    #    that cover at least one uncovered point.
    
    # Let's use a slightly smarter greedy:
    # Maintain a list of uncovered points.
    # For each uncovered point, we can add any of its neighbors.
    # We pick the neighbor that covers the most other uncovered points.
    
    # To avoid O(N^2) per step, we can limit the search or use randomization.
    # However, for N=2187, even O(N^2) is ~5M ops. 186 steps * 5M = 1e9. Still too slow.
    
    # Let's try a different approach:
    # Randomized Greedy with Restart?
    # Or a structured initial code + Repair?
    
    # The previous attempt failed because the structured code wasn't valid and repair was weak.
    # Let's use a "Random Greedy" that is fast enough.
    # We can speed up the "max coverage" calculation by using numpy vectorization.
    
    # But 186 iterations of vectorized operations over 2187 elements is fast.
    # For each step:
    # 1. Identify all candidates (neighbors of all uncovered points).
    # 2. For each candidate, count how many uncovered points it covers.
    # 3. Pick the best.
    
    # This is still potentially slow if we do it naively.
    # Let's try a "Chunked" Greedy or just a simple greedy with numpy.
    
    # Actually, let's try a different initial construction:
    # The "Tetracode" product construction gives 186 words?
    # Van Laarhoven et al. found 186.
    # The product of Tetracode (length 4, size 9, radius 1) and something else?
    # Tetracode covers 3^4.
    # If we use Tetracode for first 4 coords and something for last 3?
    # No, that's not how product codes work for covering.
    
    # Let's stick to a robust Greedy Set Cover but optimized.
    # We will maintain `uncovered` as a numpy array.
    # To find the best codeword, we can't iterate all N candidates efficiently in pure Python.
    # However, we can restrict candidates to neighbors of a *random* uncovered point?
    # No, that's not guaranteed to be optimal.
    
    # Let's use a "Greedy with Local Repair" approach.
    # 1. Start with a random valid code (e.g., all words, then remove redundant ones).
    #    This is too big (2187 words).
    # 2. Start with a smaller heuristic code (e.g., 1 per plane, 243 words) and repair.
    #    The previous plane heuristic gave 243 words initially.
    #    Repairing 243 words down to ~180 is feasible.
    
    # Let's restart with the Plane Heuristic but ensure validity first.
    
    # Plane Heuristic:
    # Partition into 243 planes (fix 5 coords, vary 2).
    # In each plane (3x3 grid), 1 word covers all 9.
    # So 243 words cover all intra-plane points.
    # Inter-plane points: A word at (a0,a1,a2,v3,a4,a5,v6) covers points differing in one fixed coord.
    # Specifically, it covers 4 "adjacent" points in each of the 5 fixed dimensions?
    # No, it covers points in planes that differ in exactly one fixed coordinate.
    # Let's check coverage of the plane heuristic.
    # If we pick one word per plane, do we cover everything?
    # Let P be a plane. We picked w in P.
    # Points in P are covered by w.
    # Points in adjacent planes P' (differ in 1 fixed coord):
    # A point x in P' is covered by w if dist(x, w) <= 1.
    # Since x and w differ in at least 1 fixed coord, they differ in that coord.
    # For dist to be 1, they must be identical in all other 6 coords.
    # So w covers exactly 5 points in adjacent planes (one for each fixed coord direction? No, just the specific point).
    # Actually, for each fixed coord i, there are 2 adjacent values.
    # So w covers 2 points in the plane differing in coord i (if the other coords match).
    # Total 5 * 2 = 10 points in adjacent planes?
    # Wait, the plane is defined by fixing 5 coords.
    # Changing one fixed coord moves to a different plane.
    # The word w has specific values for the 5 fixed coords.
    # For each fixed coord j, if we change w[j] to w[j]+1 or w[j]-1, we get a new word.
    # This new word is in a different plane.
    # Is it covered by w? Yes, dist=1.
    # So w covers 10 points outside its plane.
    # But do these 10 points get covered by the words in those *other* planes?
    # Not necessarily.
    
    # So the plane heuristic leaves gaps.
    # We will use the plane heuristic to generate an initial set of 243 words.
    # Then we will add words to cover any uncovered points (Repair).
    # Then we will remove redundant words (Minimize).
    
    # Generate Initial Code (243 words)
    code = []
    code_set = set()
    
    # Iterate through all plane keys
    for a0 in range(3):
        for a1 in range(3):
            for a2 in range(3):
                for a4 in range(3):
                    for a5 in range(3):
                        # Pick a representative word for this plane.
                        # Let's pick (a0, a1, a2, 0, a4, a5, 0)
                        # Or better, pick the one that maximizes coverage of *currently* uncovered points?
                        # Since we haven't started covering, just pick a fixed one to be fast.
                        # We can refine this in local search.
                        w = [a0, a1, a2, 0, a4, a5, 0]
                        idx = tuple_to_idx(w)
                        code.append(idx)
                        code_set.add(idx)
    
    # Calculate initial coverage
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
    
    # Repair Phase: Add words to cover any uncovered points
    # This ensures validity.
    uncovered_mask = (counts == 0)
    
    while np.any(uncovered_mask):
        if time.time() > deadline - 30:
            break
        # Find an uncovered point
        uncovered_indices = np.where(uncovered_mask)[0]
        p = int(uncovered_indices[0])
        
        # Find a codeword that covers p and maximizes coverage of other uncovered points
        # Candidates are neighbors of p
        cands = neighbor_sets[p]
        best_c = -1
        best_gain = -1
        
        for c in cands:
            if c in code_set:
                continue
            gain = int(np.sum(uncovered_mask[neighbors_list[c]]))
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1:
            # Should not happen if we check all neighbors, but just in case
            best_c = p
        
        # Add best_c
        code.append(best_c)
        code_set.add(best_c)
        counts[neighbors_list[best_c]] += 1
        uncovered_mask = (counts == 0)
        
    # Now we have a valid code.
    # Minimize Phase: Local Search to reduce size.
    
    def try_remove():
        """Remove redundant codewords."""
        removed = False
        for c in list(code):
            # Check if c is the only codeword covering any point
            # If min count in its neighborhood is > 1, it's redundant
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Build candidate pool from neighbors of uncovered words
            # Optimization: only consider neighbors of the *first* uncovered word?
            # No, that might miss a better global choice.
            # But for speed, let's try a subset or just the first one if slow.
            # Given N=2187, checking all neighbors of all uncovered is expensive if many are uncovered.
            # However, in kick moves, only a few points become uncovered.
            
            # Let's just iterate through all N candidates? No, too slow.
            # Iterate through neighbors of the first uncovered point?
            # This is a heuristic. It might not find the best, but it's fast.
            # Let's use a random sample of uncovered points to generate candidates.
            
            if len(temp_uncovered) > 10:
                sample = random.sample(list(temp_uncovered), 10)
            else:
                sample = list(temp_uncovered)
                
            candidate_pool = set()
            for w in sample:
                candidate_pool.update(neighbor_sets[w])
                
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    
            if best_c == -1:
                # Fallback: pick neighbor of first uncovered
                p = next(iter(temp_uncovered))
                best_c = next(iter(neighbor_sets[p]))
                
            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)
                
        return added

    def kick():
        """Remove a random codeword and try to repair. Accept if size decreases."""
        if not code:
            return False

        old_size = len(code)
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    # Main Local Search Loop
    best_size = len(code)
    best_code = list(code)
    
    while time.time() < deadline - 5:
        # 1. Remove redundant
        while try_remove():
            pass
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        # 2. Kick
        improved = False
        for _ in range(20): # Limit kicks per iteration to avoid getting stuck
            if kick():
                improved = True
                break # Restart loop if improved? Or continue?
                # Let's restart the outer loop to re-evaluate redundancy
        
        if not improved:
            # Try a few more kicks or break?
            # Let's try a 2-kick if 1-kick fails
            if len(code) > 2:
                old_size = len(code)
                i1 = random.randint(0, len(code) - 1)
                i2 = random.randint(0, len(code) - 1)
                if i1 == i2: i2 = (i2+1) % len(code)
                c1, c2 = code[i1], code[i2]
                
                temp_cov = counts.copy()
                temp_cov[neighbors_list[c1]] -= 1
                temp_cov[neighbors_list[c2]] -= 1
                unc = np.where(temp_cov == 0)[0]
                if len(unc) == 0:
                    code.remove(c1); code_set.remove(c1); counts[neighbors_list[c1]] -= 1
                    code.remove(c2); code_set.remove(c2); counts[neighbors_list[c2]] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    improved = True
                else:
                    unc_set = set(unc.tolist())
                    added = greedy_repair(unc_set)
                    new_size = old_size - 2 + len(added)
                    if new_size < old_size:
                        code.remove(c1); code_set.remove(c1); counts[neighbors_list[c1]] -= 1
                        code.remove(c2); code_set.remove(c2); counts[neighbors_list[c2]] -= 1
                        for c in added:
                            code.append(c); code_set.add(c); counts[neighbors_list[c]] += 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                        improved = True
            
            if not improved:
                # No improvement in this round, break to save time?
                # Or continue?
                # Let's break if we've been stuck for a while.
                # For now, just break to be safe.
                break

    # Restore best code
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        
    # Final validity check and repair (just in case)
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
        
    uncovered_mask = (counts == 0)
    while np.any(uncovered_mask):
        if time.time() > deadline - 1:
            break
        unc_idx = np.where(uncovered_mask)[0]
        p = int(unc_idx[0])
        cands = neighbor_sets[p]
        best_c = -1
        best_gain = -1
        for c in cands:
            if c in code_set: continue
            gain = int(np.sum(uncovered_mask[neighbors_list[c]]))
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1: best_c = p
        code.append(best_c)
        code_set.add(best_c)
        counts[neighbors_list[best_c]] += 1
        uncovered_mask = (counts == 0)
        
    result = [idx_to_tuple(cw) for cw in code]
    return result
