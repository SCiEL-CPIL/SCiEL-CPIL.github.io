"""Vectorized greedy + fixed-size ILS local search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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

    # Precompute neighbor lists (each word covers itself + 14 neighbors = 15 words total for radius 1 in ternary length 7)
    # Actually for ternary: 1 + 7*2 = 15
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

    # Precompute: for each position and value, which words have that value at that position
    # This helps with fast gain computation
    # Actually, let's just use vectorized numpy

    def fast_greedy():
        """Vectorized greedy set cover."""
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        while np.any(uncovered) and time.time() < deadline - 30:
            # Compute gain for all codewords: number of uncovered points each covers
            gains = np.zeros(N, dtype=np.int32)
            for c in range(N):
                nb = neighbors_list[c]
                gains[c] = np.sum(uncovered[nb])
            
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            
            best_indices = np.where(gains == max_gain)[0]
            if len(best_indices) > 1:
                best_idx = int(random.choice(best_indices))
            else:
                best_idx = int(best_indices[0])
            
            code.append(best_idx)
            uncovered[neighbors_list[best_idx]] = False
        
        return code

    def build_coverage(code):
        """Build coverage counts array from a list of codeword indices."""
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors_list[c]] += 1
        return counts

    def is_covered(counts):
        """Check if all points are covered (count >= 1 for all)."""
        return np.all(counts >= 1)

    def try_remove_pass(code, counts):
        """Try to remove redundant codewords."""
        removed = False
        c_list = list(code)
        random.shuffle(c_list)
        for c in c_list:
            if np.min(counts[neighbors_list[c]]) > 1:
                counts[neighbors_list[c]] -= 1
                code.remove(c)
                removed = True
        return removed

    def try_swap(code, counts):
        """Try to swap a codeword for a better one."""
        c_list = list(code)
        random.shuffle(c_list)
        for c in c_list:
            # Find critical points (only covered by c)
            nb = neighbors_list[c]
            crit_points = [i for i in nb if counts[i] == 1]
            if not crit_points:
                continue
            
            # Try to find a replacement that covers all critical points
            # Candidates must cover at least one critical point
            candidate_pool = set()
            for p in crit_points:
                candidate_pool.update(neighbor_sets[p])
            candidate_pool.discard(c)
            
            for w in candidate_pool:
                if w in code:
                    continue
                # Check if w covers all critical points
                valid = True
                for kp in crit_points:
                    if kp not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: how many additional redundancies does w create?
                # Remove c, add w, see if we still cover everything
                temp_counts = counts.copy()
                temp_counts[nb] -= 1
                temp_counts[neighbors_list[w]] += 1
                
                if np.all(temp_counts >= 1):
                    # Check if this swap allows further removals
                    # For now, accept the swap
                    counts[nb] -= 1
                    counts[neighbors_list[w]] += 1
                    code.remove(c)
                    code.append(w)
                    return True
        
        return False

    def fixed_size_ils(code, target_size, time_limit):
        """Fixed-size Iterated Local Search."""
        best_code = list(code)
        best_size = len(code)
        
        while time.time() < time_limit:
            # Phase 1: Removal pass
            while try_remove_pass(code, counts_global):
                pass
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            
            if len(code) <= target_size:
                break
            
            # Phase 2: Swap pass
            improved = False
            for _ in range(20):  # Max 20 swap attempts per iteration
                if try_swap(code, counts_global):
                    improved = True
                    break
            
            if not improved and len(code) > target_size:
                # Kick: remove a random codeword and repair
                if len(code) > 1:
                    idx = random.randint(0, len(code) - 1)
                    c_to_remove = code[idx]
                    counts_global[neighbors_list[c_to_remove]] -= 1
                    code.remove(c_to_remove)
                    
                    # Greedy repair
                    while not is_covered(counts_global):
                        uncovered_idx = np.where(counts_global == 0)[0]
                        if len(uncovered_idx) == 0:
                            break
                        # Find best covering codeword
                        best_c = -1
                        best_gain = -1
                        # Sample some candidates
                        sample_size = min(100, N)
                        candidates = random.sample(range(N), sample_size)
                        for c in candidates:
                            gain = 0
                            for p in neighbors_list[c]:
                                if counts_global[p] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_c = c
                        if best_c == -1 or best_gain <= 0:
                            # Fallback: try all
                            for c in range(N):
                                gain = 0
                                for p in neighbors_list[c]:
                                    if counts_global[p] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_c = c
                        if best_c != -1:
                            counts_global[neighbors_list[best_c]] += 1
                            code.append(best_c)
                        else:
                            # Restore
                            counts_global[neighbors_list[c_to_remove]] += 1
                            code.append(c_to_remove)
                            break
        
        # Restore best
        if len(best_code) < len(code):
            code.clear()
            counts_global[:] = 0
            for c in best_code:
                code.append(c)
                counts_global[neighbors_list[c]] += 1
        
        return code, best_size

    # ---------- Main ----------
    # Initial greedy
    code = fast_greedy()
    counts_global = build_coverage(code)
    
    if not code:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
    
    # Local search with multiple restarts
    global_best_code = list(code)
    global_best_size = len(code)
    
    # Target: aim for <= 186
    target = 186
    
    while time.time() < deadline - 20:
        # Run ILS
        remaining = deadline - time.time() - 10
        ls_time = min(remaining * 0.8, 30.0)
        code, sz = fixed_size_ils(code, target, time.time() + ls_time)
        
        if sz < global_best_size:
            global_best_size = sz
            global_best_code = list(code)
        
        if global_best_size <= target:
            break
    
    # Final removal pass
    counts_final = build_coverage(global_best_code)
    removed = True
    while removed and time.time() < deadline - 5:
        removed = False
        c_list = list(global_best_code)
        random.shuffle(c_list)
        for c in c_list:
            if np.min(counts_final[neighbors_list[c]]) > 1:
                counts_final[neighbors_list[c]] -= 1
                global_best_code.remove(c)
                removed = True
    
    if not global_best_code:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
    
    result = [idx_to_tuple(cw) for cw in global_best_code]
    return result
# EVOLVE-BLOCK-END
