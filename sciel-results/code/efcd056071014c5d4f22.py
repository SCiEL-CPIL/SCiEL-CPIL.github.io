# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729

    # Precompute neighbors for each word index as a numpy array
    # neighbors[i] is a list of indices covered by word i (distance <= radius)
    neighbor_lists = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbor_lists.append(neigh)
    
    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbor_lists]
    neighbor_lens = [len(neigh) for neigh in neighbor_lists]
    
    # Initial code: Direct Sum (Tetracode x 3^2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts using numpy
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts[neighbor_arrays[idx]] += 1
        
    # Active set as a set for O(1) membership, and a list for iteration
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    def can_remove_vectorized(idx):
        """Check if word idx can be removed (all neighbors have count > 1)"""
        return np.all(counts[neighbor_arrays[idx]] > 1)
    
    def try_remove_all_redundant():
        """Iteratively remove all redundant words until none remain"""
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Shuffle to break symmetry
            random.shuffle(active_list)
            for idx in active_list[:]:  # iterate over a copy
                if idx not in active_set:
                    continue
                if can_remove_vectorized(idx):
                    # Remove idx
                    counts[neighbor_arrays[idx]] -= 1
                    active_set.remove(idx)
                    active_list.remove(idx)
                    removed_any = True
            # If we removed anything, we need to re-check because removing one
            # might make others redundant or critical
    
    # Initial removal pass
    try_remove_all_redundant()
    
    # Local search with perturbations
    best_size = len(active_list)
    best_state = (set(active_set), counts.copy())
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Try perturbation: add a random word, then try to remove
        # Pick a random word not in the code
        random_idx = random.randint(0, N - 1)
        
        if random_idx in active_set:
            continue
            
        # Add it
        active_set.add(random_idx)
        active_list.append(random_idx)
        counts[neighbor_arrays[random_idx]] += 1
        
        # Now try to remove redundant words
        # We need to remove at least 2 words to improve (since we added 1)
        # Or remove 1 to stay same (for exploration)
        
        # First, try to remove as many as possible
        removals = 0
        # Try multiple passes to find all removable words
        for _ in range(3):  # up to 3 passes
            pass_removed = False
            for idx in active_list[:]:
                if idx not in active_set:
                    continue
                if can_remove_vectorized(idx):
                    counts[neighbor_arrays[idx]] -= 1
                    active_set.remove(idx)
                    active_list.remove(idx)
                    removals += 1
                    pass_removed = True
            if not pass_removed:
                break
        
        # If we removed at least 2, we improved
        if removals >= 2:
            best_size = len(active_list)
            best_state = (set(active_set), counts.copy())
            no_improvement_count = 0
        else:
            # Revert if no improvement
            if random_idx in active_set:
                active_set.remove(random_idx)
                active_list.remove(random_idx)
                counts[neighbor_arrays[random_idx]] -= 1
            no_improvement_count += 1
    
    # Restore best state
    if best_state[0] != active_set:
        active_set = best_state[0]
        counts = best_state[1].copy()
        active_list = list(active_set)
    
    # Final cleanup: try to remove any remaining redundant words
    try_remove_all_redundant()
    
    # Convert back to list of lists
    return [list(idx_to_word[idx]) for idx in active_list]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
    if n > 8:
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        # Greedy choice: pick word covering most uncovered
        for w in all_words:
            if w in code: continue # Optimization: don't pick existing
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
