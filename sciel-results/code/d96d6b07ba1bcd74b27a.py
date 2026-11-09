"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _local_search(U, max_iterations=50000, timeout=8.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        
        # Try adding a new element - more diverse candidates
        candidates = set()
        # Try small additions near existing elements with various offsets
        for x in list(U_set):
            for offset in [-2, -1, 1, 2, 3]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        # Try sums of pairs (structure-preserving additions)
        elems = list(U_set)
        for idx_a in range(min(len(elems), 8)):
            for idx_b in range(min(len(elems), 8)):
                s = elems[idx_a] + elems[idx_b]
                if s <= 2 * max_u + 50:
                    candidates.add(s)
                d = abs(elems[idx_a] - elems[idx_b])
                if d > 0 and d <= 2 * max_u + 50:
                    candidates.add(d)
        # Try powers of 2 or 3
        p2 = 1
        while p2 <= 2 * max_u + 50:
            candidates.add(p2)
            p2 *= 2
        p3 = 1
        while p3 <= 2 * max_u + 50:
            candidates.add(p3)
            p3 *= 3
        # Try values that are multiples or fractions of existing elements
        for x in elems[:10]:
            if x > 0:
                candidates.add(2 * x)
                candidates.add(3 * x)
                if x >= 2:
                    candidates.add(x // 2)
        # Random candidates in a wider range
        for _ in range(15):
            candidates.add(random.randint(0, 2 * max_u + 50))
        
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:25]:
                if time.time() - start_time > timeout:
                    break
                new_set = set(current) - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 50:
            # Try a bigger jump or swap
            if random.random() < 0.5:
                # Bigger jump
                jump_size = random.randint(10, max(50, max_u // 5))
                new_elem = random.randint(max(0, max_u - jump_size), max_u + jump_size)
                new_set = U_set | {new_elem}
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
            else:
                # Try swapping: remove one, add another
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    new_elem = random.randint(0, 2 * max_u + 50)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with known good small sets
    best_result = [0, 1, 2, 4]
    best_score = _compute_score(best_result)
    
    # Try several initial structures
    initial_sets = []
    
    # Known good small sets from exhaustive search
    initial_sets.append([0, 1, 2, 4])
    initial_sets.append([0, 1, 2, 4, 8])
    initial_sets.append([0, 1, 2, 4, 8, 16])
    initial_sets.append([0, 1, 2, 4, 8, 16, 32])
    initial_sets.append([0, 1, 2, 3, 4, 8])
    initial_sets.append([0, 1, 2, 4, 7])
    initial_sets.append([0, 1, 2, 4, 8, 13])
    initial_sets.append([0, 1, 2, 5, 10])
    initial_sets.append([0, 1, 3, 4, 8])
    
    # Powers of 2 based
    p2 = [0]
    v = 1
    while v <= 2000:
        p2.append(v)
        v *= 2
    initial_sets.append(p2)
    
    # Powers of 3 based
    p3 = [0]
    v = 1
    while v <= 2000:
        p3.append(v)
        v *= 3
    initial_sets.append(p3)
    
    # Mixed: 0, 1, then exponential
    mixed = [0, 1]
    v = 3
    while v <= 2000:
        mixed.append(v)
        v *= 2
    initial_sets.append(mixed)
    
    # Try a set with dense low part and sparse high part
    dense_sparse = list(range(15))
    v = 30
    while v <= 2000:
        dense_sparse.append(v)
        v *= 2
    initial_sets.append(dense_sparse)
    
    # Fibonacci-like growth
    fib = [0, 1]
    while fib[-1] <= 2000:
        fib.append(fib[-1] + fib[-2])
    initial_sets.append(fib)
    
    # Try sets with specific gaps
    gap_set = [0]
    v = 1
    for g in [2, 3, 5, 8, 13, 21]:
        v += g
        if v <= 2000:
            gap_set.append(v)
    initial_sets.append(gap_set)
    
    # Binary-like with some density
    binary_dense = [0, 1, 2]
    v = 4
    while v <= 2000:
        binary_dense.append(v)
        binary_dense.append(v + v // 2) if v + v // 2 <= 2000 else None
        v *= 2
    initial_sets.append(sorted(set(binary_dense)))
    
    # Sort initial sets by their immediate score (best first)
    initial_sets.sort(key=lambda s: _compute_score(s), reverse=True)
    
    for init_set in initial_sets:
        if time.time() > deadline - 10:
            break
        
        # Do local search on this initial set
        remaining = deadline - time.time() - 3
        if remaining < 1:
            break
        
        # Use more iterations for better initial sets
        if _compute_score(init_set) > best_score - 0.01:
            timeout = min(10.0, remaining)
            max_iter = 40000
        else:
            timeout = min(6.0, remaining)
            max_iter = 25000
            
        improved, score = _local_search(init_set, max_iterations=max_iter, timeout=timeout)
        
        if score > best_score:
            best_score = score
            best_result = improved
    
    # Final polish: try a few more targeted improvements
    if time.time() < deadline - 5:
        polish_timeout = min(5.0, deadline - time.time() - 1)
        if polish_timeout > 1:
            polished, polished_score = _local_search(best_result, max_iterations=20000, timeout=polish_timeout)
            if polished_score > best_score:
                best_score = polished_score
                best_result = polished
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
    
    return best_result
# EVOLVE-BLOCK-END
