import time
import itertools
import numpy as np
import random

def _ball_indices(q, n, radius):
    """
    Precompute, for each word index, the list of indices within Hamming distance <= radius.
    Returns a list of lists (or numpy arrays) of indices.
    """
    total = q ** n
    
    # Precompute powers of q
    powers = [1] * n
    for j in range(1, n):
        powers[j] = powers[j-1] * q
        
    # Precompute all neighbors for radius 1 (since that's the main case)
    # For efficiency, we store them as lists of integers.
    # If radius > 1, we would need to expand, but for q=3, n=8, r=1 is standard.
    # Let's handle general radius by BFS if needed, but r=1 is fast.
    
    if radius == 1:
        balls = []
        # Precompute the "steps" for each position
        # For position j, changing digit by delta changes index by delta * powers[j]
        # But we need to be careful with boundaries? No, base-q representation handles it.
        # Wait, changing digit d to d' in position j:
        # index' = index - d*powers[j] + d'*powers[j] = index + (d'-d)*powers[j]
        # This works perfectly for base-q.
        
        # We can precompute the list of offsets for each position?
        # offsets[j] = [(d'-d)*powers[j] for d in range(q) for d' in range(q) if d'!=d]
        
        for i in range(total):
            neighbors = [i]
            # Extract digits of i
            # Or just loop j and compute digit
            for j in range(n):
                d = (i // powers[j]) % q
                for d_new in range(q):
                    if d_new != d:
                        new_idx = i + (d_new - d) * powers[j]
                        neighbors.append(new_idx)
            balls.append(neighbors)
        return balls
    else:
        # General radius: BFS or precompute all pairs?
        # For small n, precomputing all distances is possible but O(N^2).
        # N=6561, N^2=43M. Storing lists is heavy.
        # Let's do BFS for each i? No, that's O(N * BallSize).
        # For radius > 1, let's just do a simple check on the fly if needed?
        # But the interface asks for balls to be precomputed for the solver.
        # If radius > 1, N is likely smaller or radius is small.
        # Let's implement a generic precomputation using digit vectors.
        
        words = np.zeros((total, n), dtype=np.int8)
        for i in range(total):
            x = i
            for j in range(n - 1, -1, -1):
                words[i, j] = x % q
                x //= q
                
        balls = []
        for i in range(total):
            neighbors = set()
            # Use product of deltas?
            # For radius r, we change up to r positions.
            # This is expensive to generate neighbors on the fly for each i.
            # Alternative: Check all j if dist <= radius.
            # O(N^2 * n). For N=6561, this is 43M * 8 = 344M ops. Might be too slow in Python.
            # However, the problem is specifically for radius 1 usually.
            # If radius > 1, I'll fall back to a slower precompute or assume it's handled.
            # Given the 120s limit, if radius > 1, N is probably smaller.
            
            # Let's just use the O(N^2) check for general radius if N is small enough,
            # otherwise we might TLE.
            # But for the specific task (q=3, n=8, r=1), the above block is used.
            
            # Fallback for radius > 1:
            for j in range(total):
                if np.sum(words[i] != words[j]) <= radius:
                    neighbors.add(j)
            balls.append(list(neighbors))
            
        return balls

def _tetracode_solution(q, n):
    """
    Construct a valid code using the tetracode direct sum (fallback).
    """
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    _TETRA = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    result = []
    for h in _TETRA:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(h + list(tail))
    return result

def _prune(chosen, balls):
    """
    Remove redundant codewords.
    chosen: list of indices
    balls: list of lists of indices (neighbors)
    Returns pruned list of indices.
    """
    if not chosen:
        return []
        
    total = len(balls)
    # Count how many times each point is covered
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        for j in balls[c]:
            coverage_count[j] += 1
            
    keep = [True] * len(chosen)
    
    # Iterate until no more changes
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(chosen):
            if not keep[i]:
                continue
            # Check if c is redundant
            # Redundant if all points in balls[c] are covered by other chosen words
            # i.e., coverage_count[j] > 1 for all j in balls[c]
            redundant = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                changed = True
                # Update counts
                for j in balls[c]:
                    coverage_count[j] -= 1
                    
    return [chosen[i] for i in range(len(chosen)) if keep[i]]

def _local_search(chosen, balls, deadline):
    """
    Local search: try to swap a codeword with another to allow pruning.
    Uses a strategy of identifying "critical" words and trying to replace them.
    """
    chosen = _prune(chosen, balls)
    total = len(balls)
    if not chosen:
        return []
        
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        
        # Calculate coverage counts
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen:
            for j in balls[c]:
                coverage_count[j] += 1
                
        # Find candidates for removal or swap
        # A word is "removable" if it's redundant (handled by prune).
        # A word is "swappable" if we can find a replacement that doesn't break coverage
        # and allows further pruning.
        
        # Strategy:
        # 1. Identify words with many "unique" points (points covered only by them).
        # 2. Try to replace such a word with one that covers those unique points.
        
        # To speed up, we can randomly pick a word to process.
        # Or iterate through all.
        
        # Let's iterate through all, but with a time check.
        
        for i, c in enumerate(chosen):
            if time.time() > deadline:
                break
                
            # Points uniquely covered by c
            unique_points = []
            for j in balls[c]:
                if coverage_count[j] == 1:
                    unique_points.append(j)
                    
            if not unique_points:
                # Redundant? Should have been pruned.
                # If prune missed it, remove it.
                chosen.pop(i)
                improved = True
                break
                
            # Find candidates that cover ALL unique points
            # Intersection of balls[uj] for uj in unique_points
            candidates = set(balls[unique_points[0]])
            for uj in unique_points[1:]:
                candidates &= set(balls[uj])
                
            # Filter candidates:
            # Must not be in chosen (except c itself, but we want a different one)
            chosen_set = set(chosen)
            valid_candidates = [cp for cp in candidates if cp not in chosen_set]
            
            if not valid_candidates:
                continue
                
            # Try a few random valid candidates
            rng.shuffle(valid_candidates)
            
            for cp in valid_candidates[:5]: # Limit tries per word
                # Tentative swap
                tentative = list(chosen)
                tentative[i] = cp
                
                # Check validity of tentative set
                # We can do this by checking if union of balls covers all points.
                # But it's faster to check if the unique points of the OLD word are now covered
                # by the NEW word, AND no other points become uncovered.
                # Actually, simpler: just check if the tentative set is valid.
                # O(N * ball_size).
                
                covered = np.zeros(total, dtype=bool)
                valid = True
                for cc in tentative:
                    for j in balls[cc]:
                        covered[j] = True
                if not np.all(covered):
                    continue
                    
                # If valid, prune and see if size reduces
                pruned = _prune(tentative, balls)
                if len(pruned) < len(tentative):
                    chosen = pruned
                    improved = True
                    break
                    
            if improved:
                break
                
        if not improved:
            # Try random moves to escape local minima?
            # Or just break.
            # Let's try a few random swaps that are valid (even if size doesn't reduce immediately)
            # and then prune?
            
            random_improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                if not chosen:
                    break
                i = rng.randrange(len(chosen))
                c = chosen[i]
                cp = rng.randrange(total)
                if cp in chosen and cp != c:
                    continue
                
                tentative = list(chosen)
                tentative[i] = cp
                
                # Quick validity check
                covered = np.zeros(total, dtype=bool)
                valid = True
                for cc in tentative:
                    for j in balls[cc]:
                        covered[j] = True
                if not np.all(covered):
                    continue
                    
                pruned = _prune(tentative, balls)
                if len(pruned) < len(tentative):
                    chosen = pruned
                    random_improved = True
                    break
                    
            if not random_improved:
                break
                
    return chosen

def construct(q, n, radius):
    deadline = time.time() + 115.0
    
    try:
        # Precompute balls
        balls = _ball_indices(q, n, radius)
        total = q ** n
        
        # 1. Initial construction: Greedy Set Cover
        # Start with an empty set.
        # Uncovered set = all points.
        
        uncovered = np.ones(total, dtype=bool)
        covered_count = 0
        chosen = []
        
        # Greedy step
        while covered_count < total and time.time() < deadline:
            best_idx = -1
            best_gain = 0
            
            # Find word with max gain
            for i in range(total):
                # Gain = number of uncovered points in balls[i]
                gain = 0
                for j in balls[i]:
                    if uncovered[j]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_gain == 0:
                break
                
            chosen.append(best_idx)
            for j in balls[best_idx]:
                if uncovered[j]:
                    uncovered[j] = False
                    covered_count += 1
                    
        # 2. Prune
        chosen = _prune(chosen, balls)
        
        # 3. Local Search
        chosen = _local_search(chosen, balls, deadline)
        
        # 4. Final Prune
        chosen = _prune(chosen, balls)
        
        # Convert indices to words
        result = []
        for idx in chosen:
            x = idx
            word = [0] * n
            for j in range(n - 1, -1, -1):
                word[j] = x % q
                x //= q
            result.append(word)
            
        # Verify validity
        if len(result) > 0:
            covered = np.zeros(total, dtype=bool)
            for w_idx in chosen:
                for j in balls[w_idx]:
                    covered[j] = True
            if not np.all(covered):
                return _tetracode_solution(q, n)
                
        return result
        
    except Exception:
        return _tetracode_solution(q, n)
