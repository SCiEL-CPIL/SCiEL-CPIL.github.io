import time
import random
import itertools
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 120s limit

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
    ball_size = 1 + n * (q - 1)  # 15

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

    # Precompute neighbor lists (balls)
    balls = []
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
        balls.append(nb)

    # Precompute ball sets for fast membership and intersection
    ball_sets = [set(b) for b in balls]

    # Precompute ball indices as numpy arrays for faster iteration
    ball_indices = [np.array(b, dtype=np.int32) for b in balls]

    # Precompute cover_map: for each word w, which codewords cover it
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # ---------- Greedy Set-Cover Construction with Multiple Restarts ----------
    
    def greedy_construct(seed_val, start_code=None):
        """Run greedy set-cover construction with heap-based lazy deletion.
        If start_code is given, start from that partial cover."""
        rng = random.Random(seed_val)
        covered = np.zeros(N, dtype=bool)
        cnt = np.full(N, ball_size, dtype=np.int32)
        code = []
        uncovered_count = N

        # Build heap: (-cnt[i], i) for max-heap behavior
        heap = [(-ball_size, i) for i in range(N)]
        heapq.heapify(heap)

        if start_code:
            for c in start_code:
                code.append(c)
                for w in ball_indices[c]:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

        while uncovered_count > 0 and time.time() < deadline:
            # Pop valid candidates from heap
            c = -1
            while heap:
                neg_val, i = heapq.heappop(heap)
                if cnt[i] == -neg_val and not covered[i]:
                    c = i
                    break
            if c == -1:
                break

            code.append(c)

            for w in ball_indices[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1
                        if cnt[c2] > 0:
                            heapq.heappush(heap, (-cnt[c2], c2))

        return code

    # Try multiple greedy restarts with different seeds
    best_code = None
    best_size = float('inf')
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 60:  # Need more time for local search
            break
        cod = greedy_construct(42 + restart)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod
        # Early stop if we've found a very good solution
        if best_size <= 170:
            break
    
    code = list(best_code)
    code_set = set(code)
    code_idx_map = {c: i for i, c in enumerate(code)}
    coverage = np.zeros(N, dtype=np.int32)
    for c in code:
        coverage[ball_indices[c]] += 1

    # ---------- Local Search ----------
    
    def is_redundant(c):
        # Check if removing c leaves all its neighbors covered by others
        return np.min(coverage[ball_indices[c]]) > 1

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state. Uses code_idx_map for O(1) removal.
        """
        nonlocal code, code_set, coverage, code_idx_map
        for c in removed_list:
            if c in code_set:
                idx = code_idx_map[c]
                # Swap with last element for O(1) removal
                last = code[-1]
                code[idx] = last
                code_idx_map[last] = idx
                code.pop()
                code_idx_map.pop(c)
                code_set.remove(c)
                coverage[ball_indices[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code_idx_map[c] = len(code)
                code.append(c)
                code_set.add(c)
                coverage[ball_indices[c]] += 1

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
            temp_coverage[ball_indices[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Find minimal repair
        added = []
        temp_uncovered = set(uncovered_set)
        max_iters = len(temp_uncovered) + 5
        
        for _ in range(max_iters):
            if not temp_uncovered:
                break
            best_c = -1
            best_count = 0
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(ball_sets[w])
            
            for c in candidate_pool:
                if c in code_set or c in removed:
                    continue
                cnt = len(ball_sets[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(ball_sets[best_c])
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            apply_code_change(removed, added)
            return True
        else:
            return False

    def try_swap():
        """
        Try to swap a random codeword with a different one.
        Returns True if size decreased (i.e., perfect removal found) or swap improves.
        """
        if len(code) < 2:
            return False
            
        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        temp_coverage[ball_indices[removed]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal, no need to add
            apply_code_change([removed], [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Only consider candidates adjacent to uncovered words
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(ball_sets[w])
        
        # Find a candidate to add
        best_c = -1
        best_gain = 0
        for c in candidate_pool:
            if c == removed or c in code_set:
                continue
            gain = len(ball_sets[c] & uncovered_set)
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

    def try_2for1():
        """
        Try to replace 2 codewords with 1 that covers their union better.
        Returns True if size decreased.
        """
        if len(code) < 2:
            return False
            
        # Pick 2 random distinct codewords
        idx1 = random.randint(0, len(code) - 1)
        idx2 = random.randint(0, len(code) - 1)
        if idx1 == idx2:
            return False
        
        removed = [code[idx1], code[idx2]]
        
        # Temporarily remove to check coverage
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[ball_indices[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Perfect removal of both
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        
        # Find a single codeword that covers all uncovered
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(ball_sets[w])
        
        for c in candidate_pool:
            if c in code_set or c in removed:
                continue
            if ball_sets[c] >= uncovered_set:
                apply_code_change(removed, [c])
                return True
        
        return False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 30
    random.seed(42)
    
    while time.time() < deadline - 3:  # Safety margin
        # 1. Exhaustive Removal - repeated until no more redundant codewords
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
        for _ in range(8):  # Try up to 8 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try various kick strategies in random order
            strategy = random.random()
            if strategy < 0.25:
                # Try 1-removal
                if try_kick(1):
                    kick_success = True
                    break
            elif strategy < 0.45:
                # Try 2-removal
                if try_kick(2):
                    kick_success = True
                    break
            elif strategy < 0.60:
                # Try 3-removal
                if try_kick(3):
                    kick_success = True
                    break
            elif strategy < 0.75:
                # Try swap
                if try_swap():
                    kick_success = True
                    break
            else:
                # Try 2-for-1
                if try_2for1():
                    kick_success = True
                    break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                if len(code) > 5:
                    k = random.randint(4, min(10, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                    elif try_2for1():
                        no_improvement_count = 0
                else:
                    if len(code) > 10:
                        if try_kick(3):
                            no_improvement_count = 0
                        elif try_2for1():
                            no_improvement_count = 0

    result = [idx_to_tuple(c) for c in code]
    return result
