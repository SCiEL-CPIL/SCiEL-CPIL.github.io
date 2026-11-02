"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _score_set_bitset(U):
    """Use numpy bitset for faster computation when max_u is moderate."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    
    # For small max_u, use bitset approach
    if max_u <= 20000:
        size = 2 * max_u + 1
        # Sumset: U + U is in range [0, 2*max_u]
        # Difference set: U - U is in range [-max_u, max_u], shift by max_u
        sum_bits = np.zeros(size, dtype=bool)
        diff_bits = np.zeros(size, dtype=bool)
        
        for a in U:
            for b in U:
                sum_bits[a + b] = True
                diff_bits[a - b + max_u] = True
        
        n_sums = sum_bits.sum()
        n_diffs = diff_bits.sum()
        if n_sums == 0 or n_diffs == 0:
            return -1
        ratio = n_diffs / n_sums
        if ratio <= 0:
            return -1
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return -1
        return 1 + math.log(ratio) / denom
    else:
        return _score_set(U)

def _local_search(U, deadline, max_iter=5000):
    """Local search with incremental scoring: add, remove, or swap elements."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    # Build initial caches
    sums_cache = set()
    diffs_cache = set()
    for a in U:
        for b in U:
            sums_cache.add(a + b)
            diffs_cache.add(a - b)
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        max_u = max(U)
        U_set = set(U)
        
        # Try adding an element - incremental scoring
        for new_elem in range(1, max_u + 15):
            if new_elem in U_set:
                continue
            # Compute new sums and diffs
            new_sums = sums_cache.copy()
            new_diffs = diffs_cache.copy()
            for u in U:
                new_sums.add(new_elem + u)
                new_diffs.add(new_elem - u)
                new_diffs.add(u - new_elem)
            new_max = max(max_u, new_elem)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = sorted(U + [new_elem])
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try removing an element
        for i in range(1, len(U)):
            elem_to_remove = U[i]
            if len(U) < 3:
                continue
            # Rebuild without this element
            new_U = U[:i] + U[i+1:]
            new_sums = set()
            new_diffs = set()
            for a in new_U:
                for b in new_U:
                    new_sums.add(a + b)
                    new_diffs.add(a - b)
            new_max = max(new_U)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-12:
                            U = new_U
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try swapping an element
        for i in range(1, len(U)):
            old_elem = U[i]
            for new_elem in range(1, max_u + 20):
                if new_elem in U_set or new_elem == old_elem:
                    continue
                new_U = U[:i] + [new_elem] + U[i+1:]
                new_U_sorted = sorted(new_U)
                new_sums = set()
                new_diffs = set()
                for a in new_U_sorted:
                    for b in new_U_sorted:
                        new_sums.add(a + b)
                        new_diffs.add(a - b)
                new_max = max(new_U_sorted)
                n_sums = len(new_sums)
                n_diffs = len(new_diffs)
                if n_sums > 0 and n_diffs > 0:
                    ratio = n_diffs / n_sums
                    if ratio > 0:
                        denom = math.log(2 * new_max + 1)
                        if denom > 0:
                            s = 1 + math.log(ratio) / denom
                            if s > current_score + 1e-12:
                                U = new_U_sorted
                                U_set = set(U)
                                current_score = s
                                sums_cache = new_sums
                                diffs_cache = new_diffs
                                improved = True
                                break
            if improved:
                break
        
        if not improved:
            break
    
    return U, current_score

def _simulated_annealing(U, deadline, max_iter=3000):
    """Simulated annealing to escape local minima."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    
    T = 0.05  # Initial temperature
    cooling = 0.995
    
    for _ in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        max_u = max(U)
        U_set = set(U)
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add an element
            new_elem = random.randint(1, max_u + 25)
            if new_elem in U_set:
                continue
            new_U = sorted(U + [new_elem])
        elif move_type < 0.6 and len(U) > 2:
            # Remove an element
            i = random.randint(1, len(U) - 1)
            new_U = U[:i] + U[i+1:]
        else:
            # Swap an element
            i = random.randint(1, len(U) - 1)
            new_elem = random.randint(1, max_u + 25)
            if new_elem in U_set:
                continue
            new_U = U[:i] + [new_elem] + U[i+1:]
            new_U = sorted(new_U)
        
        new_score = _score_set(new_U)
        if new_score < 0:
            continue
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        
        T *= cooling
    
    return best_U, best_score

def _polish(U, deadline):
    """Final high-precision polish with tighter tolerance."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    # Build caches
    sums_cache = set()
    diffs_cache = set()
    for a in U:
        for b in U:
            sums_cache.add(a + b)
            diffs_cache.add(a - b)
    
    max_u = max(U)
    U_set = set(U)
    
    # Try fine-grained additions and removals
    for _ in range(200):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # Try adding nearby elements
        for new_elem in range(1, max_u + 10):
            if new_elem in U_set:
                continue
            new_sums = sums_cache.copy()
            new_diffs = diffs_cache.copy()
            for u in U:
                new_sums.add(new_elem + u)
                new_diffs.add(new_elem - u)
                new_diffs.add(u - new_elem)
            new_max = max(max_u, new_elem)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-14:
                            U = sorted(U + [new_elem])
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            max_u = new_max
                            improved = True
                            break
        
        if improved:
            continue
        
        # Try removing elements
        for i in range(1, len(U)):
            if len(U) < 3:
                break
            new_U = U[:i] + U[i+1:]
            new_sums = set()
            new_diffs = set()
            for a in new_U:
                for b in new_U:
                    new_sums.add(a + b)
                    new_diffs.add(a - b)
            new_max = max(new_U)
            n_sums = len(new_sums)
            n_diffs = len(new_diffs)
            if n_sums > 0 and n_diffs > 0:
                ratio = n_diffs / n_sums
                if ratio > 0:
                    denom = math.log(2 * new_max + 1)
                    if denom > 0:
                        s = 1 + math.log(ratio) / denom
                        if s > current_score + 1e-14:
                            U = new_U
                            U_set = set(U)
                            current_score = s
                            sums_cache = new_sums
                            diffs_cache = new_diffs
                            max_u = new_max
                            improved = True
                            break
        
        if not improved:
            break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Systematic search for size 3
    for a in range(1, 30):
        for b in range(a + 1, 55):
            if time.time() > deadline - 5:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Systematic search for size 4
    for a in range(1, 20):
        for b in range(a + 1, 35):
            for c in range(b + 1, 60):
                if time.time() > deadline - 5:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Local search from best found
    if time.time() < deadline - 20:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=5000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 4, 9],
        [0, 2, 5, 11],
        [0, 1, 6, 10],
        [0, 3, 5, 8],
        [0, 1, 2, 4],
        [0, 1, 3, 6],
        [0, 2, 3, 7],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 10:
            break
        s = _score_set(start)
        if s > best_score - 0.15:
            improved_set, improved_score = _local_search(start, deadline, max_iter=3000)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
    
    # Simulated annealing from best set
    if time.time() < deadline - 15:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iter=3000)
        if sa_score > best_score:
            best_set = sa_set
            best_score = sa_score
    
    # Simulated annealing from other starting points
    for start in starting_points[:8]:
        if time.time() > deadline - 10:
            break
        s = _score_set(start)
        if s > best_score - 0.20:
            sa_set, sa_score = _simulated_annealing(start, deadline, max_iter=2000)
            if sa_score > best_score:
                best_set = sa_set
                best_score = sa_score
    
    # Final polish
    if time.time() < deadline - 5:
        polished_set, polished_score = _polish(best_set, deadline)
        if polished_score > best_score:
            best_set = polished_set
            best_score = polished_score
    
    return best_set

# EVOLVE-BLOCK-END
