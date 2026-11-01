import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies 3,6,1
        # Simple greedy for general case
        pass

    # Specific optimization for q=3, n=6, r=1
    N = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Generate all words
    words = []
    for i in range(N):
        w = []
        tmp = i
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        words.append(tuple(w))
        
    # Precompute spheres (masks)
    # Sphere of c: c and all words at dist 1
    sphere_masks = [0] * N
    for c_idx, c in enumerate(words):
        mask = 1 << c_idx
        for i in range(n):
            for delta in range(1, q):
                neighbor = list(c)
                neighbor[i] = (neighbor[i] + delta) % q
                n_idx = word_to_idx(neighbor)
                mask |= (1 << n_idx)
        sphere_masks[c_idx] = mask
        
    all_mask = (1 << N) - 1
    
    best_solution = None
    best_size = N # worst case
    
    # Helper to check validity and get size
    def get_solution_size(codeword_indices):
        return len(codeword_indices)

    def is_valid(codeword_indices):
        # Check if union of spheres covers all_mask
        cover = 0
        for idx in codeword_indices:
            cover |= sphere_masks[idx]
        return (cover == all_mask)

    def reduce_solution(codeword_indices):
        # Greedy reduction: remove redundant codewords
        # We maintain counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in codeword_indices:
            # Increment counts for all w in sphere
            # This is slow if done naively with bit masks
            # But N=729, and sphere size=13.
            # So 13 increments per codeword.
            c = words[idx]
            counts[idx] += 1
            for i in range(n):
                for delta in range(1, q):
                    neighbor = list(c)
                    neighbor[i] = (neighbor[i] + delta) % q
                    n_idx = word_to_idx(neighbor)
                    counts[n_idx] += 1
                    
        current_set = list(codeword_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_set)):
                idx = current_set[i]
                # Check if removable
                c = words[idx]
                # Check min count in sphere
                min_count = counts[idx]
                for j in range(n):
                    for delta in range(1, q):
                        neighbor = list(c)
                        neighbor[j] = (neighbor[j] + delta) % q
                        n_idx = word_to_idx(neighbor)
                        if counts[n_idx] < min_count:
                            min_count = counts[n_idx]
                
                if min_count >= 2:
                    # Remove idx
                    # Update counts
                    counts[idx] -= 1
                    for j in range(n):
                        for delta in range(1, q):
                            neighbor = list(c)
                            neighbor[j] = (neighbor[j] + delta) % q
                            n_idx = word_to_idx(neighbor)
                            counts[n_idx] -= 1
                    current_set.pop(i)
                    changed = True
                    break # Restart loop as indices shifted
        return current_set

    def greedy_construct():
        uncovered = all_mask
        selected = []
        # Candidates are all indices
        # To speed up, we can just iterate all N candidates each step
        # N=729, steps ~50-70. 729*70 = ~50k popcounts. Fast.
        while uncovered:
            best_idx = -1
            best_gain = -1
            for i in range(N):
                gain = (uncovered & sphere_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~sphere_masks[best_idx]
        
        # Reduce
        reduced = reduce_solution(selected)
        return reduced

    # Run multiple random restarts of greedy
    # Note: Greedy is deterministic if we pick the first max.
    # To randomize, we can shuffle candidates or pick randomly among ties.
    # Let's just run it a few times with slight perturbations or just once if fast.
    
    # Actually, let's try the 81-word code first as a baseline for reduction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    baseline_indices = []
    for h in tetracode:
        for tail in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]:
            w = h + list(tail)
            baseline_indices.append(word_to_idx(w))
            
    current_best = reduce_solution(baseline_indices)
    best_size = len(current_best)
    
    # Try Greedy
    greedy_sol = greedy_construct()
    if len(greedy_sol) < best_size:
        current_best = greedy_sol
        best_size = len(current_best)
        
    # Local Search on best_solution
    # Try to swap a codeword with another to allow further reduction
    # This is complex. Let's just do more random greedy restarts with random tie-breaking.
    
    import sys
    sys.setrecursionlimit(10000)

    while time.time() < deadline:
        # Randomized Greedy
        uncovered = all_mask
        selected = []
        while uncovered:
            best_gain = -1
            best_candidates = []
            for i in range(N):
                gain = (uncovered & sphere_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            # Pick random from best
            idx = random.choice(best_candidates)
            selected.append(idx)
            uncovered &= ~sphere_masks[idx]
        
        reduced = reduce_solution(selected)
        if len(reduced) < best_size:
            current_best = reduced
            best_size = len(reduced)
            # If we found 73 or less, we can try to polish further
            # But 73 is the target.
            if best_size <= 73:
                # Try to improve with local search
                # Simple hill climbing: try swapping each codeword with a random one
                # If valid and allows reduction, accept
                pass # Keep it simple for now, greedy restarts are strong

    # Convert indices back to words
    return [list(words[i]) for i in current_best]
