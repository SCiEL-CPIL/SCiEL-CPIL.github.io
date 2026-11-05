import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

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
    neighbors_list = []
    neighbors_set = []
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
        neighbors_set.append(set(nb))

    # ---------- Greedy Set-Cover Construction ----------
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # Greedy initialization
    code.append(0)
    code_set.add(0)
    coverage[neighbors_list[0]] += 1
    uncovered_mask[neighbors_list[0]] = False
    
    while uncovered_mask.any() and time.time() < deadline:
        best_c = -1
        best_count = 0
        
        # Only consider candidates adjacent to uncovered words for speed
        candidate_pool = set()
        uncovered_indices = np.where(uncovered_mask)[0]
        for w in uncovered_indices:
            candidate_pool.update(neighbors_set[w])
        
        for c in candidate_pool:
            if c in code_set:
                continue
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
            
            # Heuristic: If we found a perfect cover (cnt == number of uncovered), break
            if best_count == len(uncovered_indices):
                break

        if best_c == -1 or best_count == 0:
            break
            
        code.append(best_c)
        code_set.add(best_c)
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False

    # ---------- Local Search ----------
    
    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        # i.e., coverage[neighbors] > 1 for all neighbors
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        """
        Given a set of uncovered word indices, find a minimal set of codewords
        to add to cover them.
        Returns list of added codewords.
        """
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])
            
        return added

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state.
        """
        nonlocal code, code_set, coverage
        for c in removed_list:
            if c in code_set:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        """
        Try to remove k random codewords and repair.
        Returns True if size decreased.
        """
        if len(code) < k:
            return False
            
        # Pick k distinct random indices
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            # Apply change
            apply_code_change(removed, added)
            return True
        else:
            # Even if size doesn't decrease, check if we can find a better configuration
            # by trying different repair strategies
            if len(uncovered_set) <= 20:  # Small uncovered set, try more carefully
                # Try to find a single codeword that covers all uncovered
                candidate_pool = set()
                for w in uncovered_set:
                    candidate_pool.update(neighbors_set[w])
                
                best_single = -1
                best_gain = 0
                for c in candidate_pool:
                    if c in code_set or c in removed:
                        continue
                    gain = len(neighbors_set[c] & uncovered_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_single = c
                
                if best_single != -1 and best_gain == len(uncovered_set):
                    # Single codeword covers all uncovered
                    apply_code_change(removed, [best_single])
                    return True
            
            return False

    def try_swap():
        """
        Try to swap a random codeword with a different one.
        Returns True if size decreased.
        """
        if len(code) < 2:
            return False
            
        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[removed]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal, no need to add
            apply_code_change([removed], [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Only consider candidates adjacent to uncovered words (much faster than scanning all N)
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])
        
        # Find a candidate to add
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(neighbors_set[c] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if best_gain >= len(uncovered_set):
                    break
        
        if best_c == -1:
            return False
            
        # Apply the swap if it covers all uncovered
        if best_gain >= len(uncovered_set):
            apply_code_change([removed], [best_c])
            return True
        else:
            return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 30
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(3):  # Try up to 3 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try 1-removal
            if try_kick(1):
                kick_success = True
                break
            # Try 2-removal
            elif try_kick(2):
                kick_success = True
                break
            # Try 3-removal
            elif try_kick(3):
                kick_success = True
                break
            # Try swap
            elif try_swap():
                kick_success = True
                break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                # Try a random kick with larger k
                if len(code) > 5:
                    k = random.randint(4, min(8, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                else:
                    # If we're stuck, try a more aggressive restart of local search
                    # by accepting a temporary size increase and then optimizing
                    if len(code) > 10:
                        # Remove 3 random codewords and repair
                        if try_kick(3):
                            no_improvement_count = 0

    result = [idx_to_tuple(c) for c in code]
    return result
