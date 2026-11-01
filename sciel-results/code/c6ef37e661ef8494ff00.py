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
    N = q ** n  # 729
    
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists and bitmasks
    neighbors_list = [[] for _ in range(N)]
    neighbors_mask = [0] * N
    
    for i, w in enumerate(all_words):
        # Self
        neighbors_list[i].append(i)
        neighbors_mask[i] |= (1 << i)
        
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                neighbors_list[i].append(j)
                neighbors_mask[i] |= (1 << j)
                
    full_mask = (1 << N) - 1
    
    # State variables
    # We will maintain:
    # code_set: set of indices in the current code
    # cover_count: list of length N, count of how many codewords cover each word
    # uncovered_mask: bitmask of words with cover_count == 0
    
    def greedy_start():
        code_set = set()
        cover_count = [0] * N
        uncovered_mask = full_mask
        
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            
            # Greedy choice: pick word covering most uncovered words
            # Iterate all words, skip if already in code (though adding same word is useless)
            # Optimization: We can just iterate all. N=729 is small.
            for i in range(N):
                if i in code_set:
                    continue
                # Count how many currently uncovered words this word covers
                # new_covered = neighbors_mask[i] & uncovered_mask
                # cnt = new_covered.bit_count()
                # Using Python's int.bit_count() is very fast
                cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
                    
            if best_idx == -1 or best_cnt == 0:
                break
                
            # Add best_idx
            code_set.add(best_idx)
            # Update cover_count
            for j in neighbors_list[best_idx]:
                cover_count[j] += 1
            # Update uncovered_mask
            uncovered_mask &= ~neighbors_mask[best_idx]
            
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count, uncovered_mask):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            # Iterate over a copy to avoid modification during iteration
            # We want to remove any codeword that is not critical
            # A codeword i is critical if there exists j in neighbors[i] such that cover_count[j] == 1
            # If we remove i, cover_count[j] decreases by 1.
            # So i is removable if min(cover_count[j] for j in neighbors[i]) > 1
            for i in list(code_set):
                # Check if i is removable
                removable = True
                for j in neighbors_list[i]:
                    if cover_count[j] == 1:
                        removable = False
                        break
                if removable:
                    # Remove i
                    code_set.remove(i)
                    for j in neighbors_list[i]:
                        cover_count[j] -= 1
                    # Update uncovered_mask if any word becomes uncovered
                    # Actually, if min count > 1, no word becomes 0. So uncovered_mask stays 0.
                    # Wait, if cover_count[j] was 2, it becomes 1. Still covered.
                    # So uncovered_mask remains 0 (or whatever it was, which is 0 if valid).
                    changed = True
                    break # Restart loop as state changed
        return code_set, cover_count, uncovered_mask

    def repair(code_set, cover_count, uncovered_mask):
        """Add codewords until valid cover."""
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                if i in code_set:
                    continue
                cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
                    
            if best_idx == -1 or best_cnt == 0:
                # Should not happen if graph is connected and we have full space
                break
                
            code_set.add(best_idx)
            for j in neighbors_list[best_idx]:
                cover_count[j] += 1
            uncovered_mask &= ~neighbors_mask[best_idx]
            
        return code_set, cover_count, uncovered_mask

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution
    code_set, cover_count, uncovered_mask = greedy_start()
    code_set, cover_count, uncovered_mask = reduce_code(code_set, cover_count, uncovered_mask)
    
    # Verify initial solution is valid (uncovered_mask should be 0)
    if uncovered_mask:
        code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
        code_set, cover_count, uncovered_mask = reduce_code(code_set, cover_count, uncovered_mask)

    best_size = len(code_set)
    best_code = set(code_set)
    best_cover_count = list(cover_count)
    best_uncovered_mask = uncovered_mask
    
    # ILS
    # We will perturb by removing k random codewords, then repair, then reduce.
    # We accept the new solution if it is strictly better, or equal with some probability.
    # Given the small search space, strictly better is usually sufficient to find 73.
    
    while time.time() < deadline - 2:
        # Perturb
        # Remove k random codewords
        k = rng.randint(2, 5)
        current_code = list(code_set)
        if len(current_code) < k:
            k = max(1, len(current_code) // 2)
        if k == 0:
            continue
            
        # Pick k random indices to remove
        to_remove = rng.sample(current_code, k)
        
        # Remove them
        for idx in to_remove:
            code_set.remove(idx)
            for j in neighbors_list[idx]:
                cover_count[j] -= 1
                if cover_count[j] == 0:
                    uncovered_mask |= (1 << j)
        
        # Repair
        code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
        
        # Reduce
        code_set, cover_count, uncovered_mask = reduce_code(code_set, cover_count, uncovered_mask)
        
        # Check if better
        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = set(code_set)
            best_cover_count = list(cover_count)
            best_uncovered_mask = uncovered_mask
            # Reset current to best to continue from there? 
            # Or continue from current? ILS usually continues from the accepted solution.
            code_set = set(best_code)
            cover_count = list(best_cover_count)
            uncovered_mask = best_uncovered_mask
        elif current_size == best_size:
            # Accept with probability to explore?
            # For minimizing, accepting equal solutions can help escape local minima.
            if rng.random() < 0.5:
                code_set = set(best_code)
                cover_count = list(best_cover_count)
                uncovered_mask = best_uncovered_mask
            else:
                # Keep current
                pass
        else:
            # Worse, revert to best
            code_set = set(best_code)
            cover_count = list(best_cover_count)
            uncovered_mask = best_uncovered_mask

    # Convert back to list of lists
    return [list(idx_to_word[idx]) for idx in best_code]

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
