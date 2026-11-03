import itertools
import time
import numpy as np

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute suffixes and prefixes for fast overlap calculation
    # suffixes[i][k] is the string of length k ending at perm i
    # prefixes[j][k] is the string of length k starting at perm j
    # We only need k up to n-1
    
    # To save memory and time, we can compute overlap on the fly for NN,
    # but for Local Search, we need the matrix.
    
    # Let's build the overlap matrix using NumPy if possible, or efficient Python.
    # N=5040. N^2 = 25,401,600.
    # dtype uint8 is 25MB.
    
    ov_matrix = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: Group permutations by their suffixes/prefixes?
    # Or just loop.
    
    # Let's try a vectorized approach for overlap calculation?
    # It's hard to vectorize string comparison easily.
    # We will use a Python loop but optimized.
    
    # Precompute prefixes and suffixes as lists of strings
    prefixes = []
    suffixes = []
    for s in perm_strs:
        pre = [s[:k] for k in range(1, n)]
        suf = [s[n-k:] for k in range(1, n)]
        prefixes.append(pre)
        suffixes.append(suf)
        
    # Fill matrix
    # We iterate i, j.
    # Find max k such that suffixes[i][k-1] == prefixes[j][k-1]
    
    # This is 25M iterations.
    # Inner loop: check k from n-1 down to 1.
    # Most will fail at k=n-1.
    
    # To speed this up, we can use a dictionary mapping suffixes to lists of indices?
    # No, we need specific i, j pairs.
    
    # Let's just run the loop. It might take 10-20s.
    
    start_time = time.time()
    
    for i in range(N):
        si = suffixes[i]
        for j in range(N):
            sj = prefixes[j]
            max_ov = 0
            # Check from largest k
            # si[k-1] corresponds to length k
            for k in range(n - 1, 0, -1):
                if si[k - 1] == sj[k - 1]:
                    max_ov = k
                    break
            ov_matrix[i, j] = max_ov
            
    # Check time
    # If it takes too long, we might need to abort.
    # But 120s limit is generous for 25M simple ops.
    
    # 1. Greedy Nearest Neighbor
    unvisited = set(range(N))
    unvisited.remove(0)
    path = [0]
    current = 0
    
    for _ in range(N - 1):
        best_next = -1
        best_overlap = -1
        # Find max overlap
        # We can use numpy to speed up the search?
        # row = ov_matrix[current]
        # candidates = list(unvisited)
        # overlaps = row[candidates]
        # best_idx = np.argmax(overlaps)
        # This is O(N) but vectorized.
        
        candidates = list(unvisited)
        overlaps = ov_matrix[current, candidates]
        max_ov_val = np.max(overlaps)
        # Get all indices with max overlap
        best_indices = np.where(overlaps == max_ov_val)[0]
        
        # Tie breaking: Pick the one with lowest index? Or random?
        # Let's pick the one that has the lowest "out-degree" in the remaining graph?
        # That requires calculating degrees.
        # For now, pick the first one.
        best_next = candidates[best_indices[0]]
        
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # 2. Local Search: 1-Opt (Adjacent Swap)
    
    def get_path_cost(p):
        total = n
        for i in range(len(p) - 1):
            total += n - ov_matrix[p[i], p[i+1]]
        return total
        
    improved = True
    max_passes = 50
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, N - 1):
            # Swap path[i] and path[i+1]
            # Old edges: (i-1, i), (i, i+1), (i+1, i+2)
            # New edges: (i-1, i+1), (i+1, i), (i, i+2)
            
            a = path[i-1]
            b = path[i]
            c = path[i+1]
            d = path[i+2] if i+2 < N else None
            
            old_cost = 0
            new_cost = 0
            
            # Edge a->b
            old_cost += n - ov_matrix[a, b]
            # Edge b->c
            old_cost += n - ov_matrix[b, c]
            # Edge c->d
            if d is not None:
                old_cost += n - ov_matrix[c, d]
                
            # Edge a->c
            new_cost += n - ov_matrix[a, c]
            # Edge c->b
            new_cost += n - ov_matrix[c, b]
            # Edge b->d
            if d is not None:
                new_cost += n - ov_matrix[b, d]
                
            if new_cost < old_cost:
                path[i], path[i+1] = path[i+1], path[i]
                improved = True
                
    # 3. Local Search: Or-Opt (Move 1 element)
    # Try moving path[i] to position j
    
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(N):
            if i == 0: continue # Moving first element is like rotation, doesn't change cost much? 
                               # Actually, cost is cyclic? No, it's a path.
                               # Moving first element to end changes start/end costs.
                               # Let's allow it.
            
            item = path[i]
            # Remove item
            # The cost change depends on neighbors.
            
            # We will try inserting at all positions j != i
            # To avoid O(N^2) inside the while loop which is O(N^2) total, 
            # this might be slow if many improvements.
            # But N=5040. N^2 = 25M.
            # If we do this for 20 passes, 500M ops. Might be too slow in Python.
            
            # Let's restrict Or-Opt to a few passes or skip it if time is tight.
            # Given the 120s limit, and matrix build ~15s, NN ~2s, 1-Opt ~5s.
            # Or-Opt might take 30-60s.
            
            # Let's implement a limited Or-Opt: only try moving to nearby positions?
            # Or just skip Or-Opt and rely on 1-Opt.
            
            # Let's try a few moves.
            # We'll break out if time > 100s.
            
            if time.time() - start_time > 100:
                break
                
            # Try inserting at random positions? No, deterministic is better for reproducibility.
            # Let's try inserting at j = (i + 1) % N, (i + 2) % N, etc?
            # No, let's just do a full scan but break early if time is up.
            
            for j in range(N):
                if j == i or j == i + 1: # j==i+1 is same position effectively after removal? 
                    # If we remove i, the list shrinks.
                    # Inserting at j in the new list corresponds to original indices.
                    # This is messy.
                    continue
                    
                # Calculate cost change for moving item from i to j
                # This is complex to do O(1).
                # Let's skip full Or-Opt to be safe on time.
                pass
                
    # Build result string
    res = perm_strs[path[0]]
    for i in range(1, N):
        ov = ov_matrix[path[i-1], path[i]]
        res += perm_strs[path[i]][ov:]
        
    return res
