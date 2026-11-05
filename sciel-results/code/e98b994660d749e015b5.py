import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    degree = 1 + n * (q - 1) # 15

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

    def run_greedy():
        """Run greedy set cover with random tie-breaking."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 50: # Leave time for local search
                break
            
            # Compute gains
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                gains[neighbors_list[idx]] += 1
            
            # Random tie-breaking among max gain candidates
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors_list[best_idx]] = False
            
        return code

    # Run multiple greedy starts
    best_initial_code = []
    best_initial_size = float('inf')
    
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 80:
            break
        code = run_greedy()
        if len(code) < best_initial_size:
            best_initial_size = len(code)
            best_initial_code = code

    code = best_initial_code
    code_set = set(code)
    
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def greedy_repair(uncovered_set):
        """Greedy repair using vectorized gain on uncovered set."""
        if not uncovered_set:
            return []
        
        added = []
        temp_uncovered = set(uncovered_set)
        
        # Convert to numpy boolean array for fast indexing
        unc_mask = np.zeros(N, dtype=bool)
        for u in temp_uncovered:
            unc_mask[u] = True
        
        while np.any(unc_mask):
            # Compute gains for all candidates
            gains = np.zeros(N, dtype=np.int32)
            # Only need to check candidates that are neighbors of uncovered points
            # But N is small (2187), so just compute for all?
            # Or iterate uncovered points?
            # Iterating uncovered points is faster if uncovered is small.
            # But unc_mask is boolean.
            # Let's use the accumulation method again, it's robust.
            unc_indices = np.where(unc_mask)[0]
            for idx in unc_indices:
                gains[neighbors_list[idx]] += 1
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
                
            added.append(best_idx)
            unc_mask[neighbors_list[best_idx]] = False
            
        return added

    def try_remove():
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        found_swap = False
        shuffled_crit = list(crit_points)
        random.shuffle(shuffled_crit)

        for p in shuffled_crit:
            if found_swap:
                break

            cands = [x for x in neighbor_sets[p] if x not in code_set]
            
            best_w = -1
            best_score = -1
            # Score: minimize redundancy. 
            # Current redundancy in c's neighborhood: sum(counts[nb] - 1 for nb in nb_c if counts[nb]>1)
            # New redundancy: sum(counts[nb] - 1 for nb in nb_w if counts[nb]>1) 
            # Actually, we want to maximize the number of points that become "safe" (count > 1) 
            # or minimize the total sum of counts.
            
            # Simple score: maximize points covered by w that are currently count 1 (critical for c)
            # This is already handled by the validity check.
            # Secondary score: maximize points in w's neighborhood that are count 1 (making them count 2, increasing redundancy but safe)
            # Wait, we want to REMOVE c. So we want w to cover the critical points.
            # Any additional coverage by w is "free" if it doesn't increase size.
            # So we prefer w that covers *more* critical points? 
            # The validity check requires covering *all* critical points of c.
            # So all valid w cover all crit_points.
            # Then we prefer w that has high "potential" to cover other things?
            # Or low "cost"?
            # Let's stick to: prefer w that has the most neighbors with count == 1.
            # This creates redundancy, which helps future removals.
            
            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break

                if not valid:
                    continue

                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1

                found_swap = True
                return True

        # Strategy 2: Relocation
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)
        
        best_w = -1
        best_score = -1
        first_point = list(all_needed)[0]
        cands = [x for x in neighbor_sets[first_point] if x not in code_set]

        for w in cands:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue

            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False

    def kick(k):
        """Remove k random codewords and repair."""
        if len(code) < k:
            return False

        old_size = len(code)
        indices = random.sample(range(len(code)), k)
        to_remove = [code[i] for i in indices]

        temp_coverage = counts.copy()
        for c in to_remove:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        if len(uncovered) == 0:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - k + len(added)

        if new_size < old_size:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    # Main Loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        shuffled_code = list(code)
        random.shuffle(shuffled_code)
        for c in shuffled_code:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # Kicks
        for k in [1, 2, 3]:
            kicks_tried = 0
            max_kicks = 30
            while kicks_tried < max_kicks and time.time() < deadline - 10:
                if kick(k):
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Restore best
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    result = [idx_to_tuple(cw) for cw in code]
    return result
