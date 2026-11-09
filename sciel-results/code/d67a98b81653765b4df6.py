import numpy as np
import random

def construct(q, n, radius):
    # Precomputation
    N = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Precompute balls: ball[i] contains i and all neighbors at dist <= radius
    # For radius=1, this is i + 2*n neighbors (for q=3)
    balls = []
    for i in range(N):
        b = [i]
        for p in range(n):
            cur = (i // powers[p]) % q
            base = i - cur * powers[p]
            for nd in range(q):
                if nd != cur:
                    b.append(base + nd * powers[p])
        balls.append(b)
    
    best_code = None
    best_size = float('inf')
    
    # Number of random restarts
    num_restarts = 50
    
    for seed in range(num_restarts):
        random.seed(seed)
        np.random.seed(seed)
        
        # 1. Randomized Greedy Construction
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        # Precompute ball arrays for faster numpy ops if needed, 
        # but list of lists is fine for small N.
        # To speed up greedy, we can maintain a coverage count array?
        # No, standard greedy picks max coverage.
        # We can optimize the "find max" step.
        
        while np.any(uncovered):
            # Find candidate with max coverage
            # For speed, we can sample a few or just do full scan.
            # Full scan is 6561 * 500 = 3.2M checks. Each check is sum of 17 bools.
            # 3.2M * 17 ~ 50M ops. In Python, this might be slow (1-2s per restart?).
            # 50 restarts * 2s = 100s. Risky.
            
            # Optimization: Use numpy for the sum.
            # We need to compute coverage for all i at once?
            # cov[i] = sum(uncovered[j] for j in balls[i])
            
            # Let's try to vectorize or use a smarter heuristic.
            # Alternative: Pick a random uncovered point, find all codewords covering it, pick one randomly?
            # This is "Randomized Greedy by Point".
            # 1. Pick random uncovered point u.
            # 2. Find all i such that u in balls[i].
            # 3. Pick i randomly from these candidates (or weighted by coverage?).
            # This is much faster: O(N) to find candidates covering u, then pick.
            # Finding candidates covering u: We need reverse balls (who covers u?).
            # Since the graph is symmetric, balls[u] contains all i that cover u?
            # Yes, dist(i,u) <= 1 iff dist(u,i) <= 1. So balls[u] is the set of candidates.
            
            # This "Point-Based Randomized Greedy" is very fast.
            
            # Get a random uncovered point
            unc_indices = np.where(uncovered)[0]
            if len(unc_indices) == 0:
                break
            u = unc_indices[np.random.randint(len(unc_indices))]
            
            # Candidates are all i in balls[u]
            candidates = balls[u]
            
            # To be smarter, we can pick the candidate that covers the most uncovered points.
            # But that requires computing coverage for ~17 candidates.
            # Let's compute coverage for candidates in balls[u] only.
            best_cand = -1
            best_cov = -1
            for i in candidates:
                # Compute coverage of i
                # cov = sum(uncovered[j] for j in balls[i])
                # Using numpy sum on a small array is fast.
                # But we don't have balls[i] as numpy array precomputed for every i?
                # We can convert on the fly or precompute numpy arrays.
                
                # Let's precompute numpy arrays for balls for faster access.
                # Actually, let's just use the list and loop, 17 items is small.
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_cand = i
            
            # Add best_cand to code
            code.append(best_cand)
            # Update uncovered
            for j in balls[best_cand]:
                uncovered[j] = False
        
        # 2. Pruning Phase
        # Convert code to list of unique indices (greedy might add duplicates? No, once covered, uncovered[j]=False, so i won't be picked again if it only covers covered points? 
        # Wait, if i is picked, we mark its ball as covered. If we pick i again, cov will be 0 (if all its ball is covered). 
        # But if it has some uncovered points, it might be picked again? 
        # No, if i is in code, its ball is fully covered. So any future pick will have cov >= 1 for some other point.
        # However, the same index i could be picked again if the logic allows? 
        # My logic: pick u (uncovered). i must cover u. If i is already in code, its ball is covered, so u would be covered. Contradiction. 
        # So no duplicates.
        
        # Pruning
        # cover_count[j] = number of codewords covering j
        cover_count = np.zeros(N, dtype=int)
        for c in code:
            for j in balls[c]:
                cover_count[j] += 1
        
        # Try to remove each word
        # Randomize order of code for pruning to escape local minima
        random.shuffle(code)
        
        for c in code:
            # Check if c is redundant
            # c is redundant if for all j in balls[c], cover_count[j] > 1
            redundant = True
            for j in balls[c]:
                if cover_count[j] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                for j in balls[c]:
                    cover_count[j] -= 1
                code.remove(c)
        
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert best_code to words
    result = []
    for idx in best_code:
        word = []
        rem = idx
        for p in range(n):
            digit = rem % q
            word.append(digit)
            rem //= q
        word.reverse()
        result.append(word)
        
    return result
