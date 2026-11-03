"""
Optimized ternary covering code K_3(6,1) search.
Strategy:
1. Fast greedy set-cover to get a valid initial code.
2. Aggressive local search:
   - Redundancy removal (drop centers whose sphere is fully covered by others).
   - 1-swap moves: try replacing one center with another to reduce size or maintain coverage with smaller "cost".
   - 1-drop + repair: drop a center, repair uncovered words greedily, then local search.
3. Perturbation (re-heating) to escape local minima.
4. Deadline guard at 145s to ensure safe return.
"""
import itertools
import random
import time
import sys

import numpy as np

def _build_space(q, n, radius):
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute spheres as lists of indices
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        if radius >= 1:
            for pos in range(n):
                cur = w[pos]
                for v in range(q):
                    if v != cur:
                        w2 = list(w)
                        w2[pos] = v
                        s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
        
    # NumPy adjacency matrix: sphere_np[i, j] = 1 if word j is in sphere of center i
    sphere_np = np.zeros((total, total), dtype=np.int8)
    for i, s in enumerate(sphere):
        sphere_np[i, s] = 1
        
    return words, sphere, sphere_np, total

def _verify(code_idx, sphere_np):
    if not code_idx:
        return sphere_np.shape[0] == 0
    total = sphere_np.shape[0]
    counts = sphere_np[code_idx].sum(axis=0)
    return bool((counts > 0).all())

def _coverage_counts(code_idx, sphere_np):
    if not code_idx:
        return np.zeros(sphere_np.shape[0], dtype=np.int32)
    return sphere_np[code_idx].sum(axis=0)

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 145.0  # Safety margin under 150s limit

    words, sphere, sphere_np, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # --- Initial Seed: Greedy Set Cover ---
    # This is fast and provides a valid starting point.
    def greedy_cover():
        uncov = set(range(total))
        code = []
        # Precompute sphere sets for faster intersection? 
        # For 729 words, list intersection is fine.
        while uncov:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                # Count how many uncovered words are in sphere[c]
                # Optimization: if sphere[c] is subset of already covered, skip? 
                # Simple count is robust.
                cnt = 0
                # Use numpy for speed if possible, but set intersection is O(|sphere|)
                # sphere[c] is a list of ~19 indices.
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        return code

    initial_code = greedy_cover()
    best_code = initial_code
    best_size = len(best_code)
    
    # --- Local Search Functions ---
    
    def local_search(code, max_iters=500):
        """
        Attempts to reduce the size of the code.
        1. Remove redundant centers.
        2. Try 1-swap moves (replace one center with another to potentially allow further removals, 
           or directly if it doesn't break coverage and we can remove another? No, swap keeps size same.
           Actually, standard local search for covering codes:
           - Try to remove a center if possible.
           - Try to swap a center with another if it allows a subsequent removal (2-opt).
           - Or simpler: just redundancy removal is very powerful for this size.
        """
        code = sorted(set(code))
        changed = True
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            new_code = []
            # Compute coverage counts
            counts = _coverage_counts(code, sphere_np)
            
            # First pass: remove redundant centers
            for ci in code:
                # A center ci is redundant if every word in its sphere is covered by at least 2 centers.
                # i.e., min(counts[sphere[ci]]) >= 2
                if counts[sphere[ci]].min() >= 2:
                    # Remove ci
                    counts -= sphere_np[ci]
                    changed = True
                else:
                    new_code.append(ci)
            
            if not changed:
                # Try 1-swap moves to enable removals?
                # If we can't remove, can we swap?
                # A swap (remove a, add b) is useful if it allows a subsequent removal.
                # This is expensive. Let's try a simpler heuristic:
                # Try swapping each center 'a' with each center 'b' (not in code).
                # If the new code is still valid, check if any center in the new code is redundant.
                # If so, remove it -> size decreases by 1.
                
                # Optimization: Only try swaps for a subset if time is tight, but 729 is small.
                # Let's try swaps only if we haven't improved in a while or randomly.
                # To keep it fast, we'll do a limited number of swap checks per iteration.
                
                # Actually, for K_3(6,1), simple redundancy removal from a good greedy start often gets to 73-75.
                # To get to 73, we might need swaps.
                
                # Let's implement a "swap and remove" check.
                # For each a in code:
                #   For each b in range(total) not in code:
                #      new_code = (code - {a}) + {b}
                #      if valid(new_code):
                #          try to remove redundant centers from new_code.
                #          if size < len(code): return new_code
                # This is O(N^2 * N) which is 729^3 ~ 380M. Too slow for Python loops.
                
                # Alternative: Simulated Annealing style?
                # Or just rely on multiple restarts with different greedy orders.
                
                # Let's try a lighter swap: 
                # If we can't remove, break.
                break
            
            code = new_code
        
        return code

    def perturbation(code):
        """
        Randomly modify the code to escape local minima.
        1. Remove k random centers.
        2. Repair uncovered words with greedy.
        3. Local search.
        """
        if len(code) < 2:
            return code
            
        code = list(code)
        rng = random.Random()
        
        # Remove 1 or 2 centers
        k = rng.choice([1, 2, 3])
        if k >= len(code):
            k = max(1, len(code) - 1)
            
        to_remove = rng.sample(code, k)
        new_code = [c for c in code if c not in to_remove]
        
        # Repair
        counts = _coverage_counts(new_code, sphere_np)
        uncov = set(np.where(counts == 0)[0])
        
        if uncov:
            # Greedy repair
            while uncov:
                best_c = -1
                best_cnt = -1
                for c in range(total):
                    if c in new_code:
                        continue
                    cnt = 0
                    for w in sphere[c]:
                        if w in uncov:
                            cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                if best_c == -1 or best_cnt <= 0:
                    break
                new_code.append(best_c)
                for w in sphere[best_c]:
                    uncov.discard(w)
        
        # Local search on the repaired code
        return local_search(new_code)

    # --- Main Search Loop ---
    
    # First, local search on the initial greedy code
    best_code = local_search(initial_code)
    best_size = len(best_code)
    
    rng = random.Random(42)
    iterations = 0
    
    while time.time() < DEADLINE - 5.0:
        iterations += 1
        
        # Perturbation
        perturbed = perturbation(best_code)
        
        # If perturbation resulted in a smaller or equal code, keep it
        # If it's larger, discard.
        if len(perturbed) < best_size:
            best_code = perturbed
            best_size = len(best_code)
        elif len(perturbed) == best_size:
            # Accept equal size to allow diversity, but only sometimes
            if rng.random() < 0.5:
                best_code = perturbed
        
        # Additional local search on the current best
        # Sometimes run a more aggressive local search
        if iterations % 10 == 0:
            # Try a "shake" and local search
            # Randomly swap a few centers
            if len(best_code) > 5:
                code_copy = list(best_code)
                for _ in range(rng.randint(2, 5)):
                    idx = rng.randrange(len(code_copy))
                    new_c = rng.randrange(total)
                    if new_c not in code_copy:
                        code_copy[idx] = new_c
                # Check validity, if invalid, repair
                counts = _coverage_counts(code_copy, sphere_np)
                uncov = set(np.where(counts == 0)[0])
                if uncov:
                    # Repair
                    while uncov:
                        best_c = -1
                        best_cnt = -1
                        for c in range(total):
                            if c in code_copy:
                                continue
                            cnt = 0
                            for w in sphere[c]:
                                if w in uncov:
                                    cnt += 1
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_c = c
                        if best_c == -1 or best_cnt <= 0:
                            break
                        code_copy.append(best_c)
                        for w in sphere[best_c]:
                            uncov.discard(w)
                
                improved = local_search(code_copy)
                if len(improved) < best_size:
                    best_code = improved
                    best_size = len(best_code)

    # Final verification
    if not _verify(best_code, sphere_np):
        # Fallback to initial greedy code (which was valid)
        # Re-verify initial
        if not _verify(initial_code, sphere_np):
            # Should not happen, but create a trivial valid code (all words)
            best_code = list(range(total))
        else:
            best_code = initial_code

    return [list(words[i]) for i in sorted(set(best_code))]

if __name__ == "__main__":
    # Quick self-test
    code = construct(3, 6, 1)
    print(f"Found code size: {len(code)}")
    # Verify
    words, sphere, sphere_np, total = _build_space(3, 6, 1)
    word_to_idx = {w: i for i, w in enumerate(words)}
    code_idx = [word_to_idx[tuple(c)] for c in code]
    print(f"Valid: {_verify(code_idx, sphere_np)}")
