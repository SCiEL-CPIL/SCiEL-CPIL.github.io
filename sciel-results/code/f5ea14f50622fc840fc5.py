"""Combined program: Exhaustive small-set search + continuous geometric optimization + local search."""
import time
import random
from math import log
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def exhaustive_small_search(deadline):
    """Exhaustively search small sets to find good structured candidates."""
    best_score = -1
    best_set = [0, 1]
    
    # Search over sets of size 3 to 8 with values up to 20
    for size in range(3, 9):
        if time.time() > deadline - 30:
            break
        for combo in combinations(range(1, 21), size - 1):
            if time.time() > deadline - 30:
                break
            U = [0] + list(combo)
            U.sort()
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1
    return score_set(U)

def score_arithmetic(a, d, m):
    """Score an arithmetic progression U = {a + k*d for k=0..m} with 0 included."""
    if d <= 0 or m < 1:
        return -1
    U = set()
    U.add(0)
    for k in range(m + 1):
        U.add(a + k * d)
    U = sorted(U)
    if len(U) < 2:
        return -1
    return score_set(U)

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions using ternary search."""
    best_score = -1
    best_set = None
    
    # Coarse grid to identify promising regions
    b_grid = np.linspace(1.5, 5.0, 25)
    m_range = range(3, 16)
    
    coarse_best_b = 2.0
    coarse_best_m = 5
    coarse_best_score = -1
    
    for m in m_range:
        if time.time() > deadline - 15:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > coarse_best_score:
                coarse_best_score = s
                coarse_best_b = b
                coarse_best_m = m
    
    # Fine ternary search around coarse best for each nearby m
    for m in range(max(3, coarse_best_m - 2), min(16, coarse_best_m + 3)):
        if time.time() > deadline - 10:
            break
        b_lo = max(1.5, coarse_best_b - 0.8)
        b_hi = min(5.0, coarse_best_b + 0.8)
        
        lo, hi = b_lo, b_hi
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            s1 = score_geometric(m1, m)
            s2 = score_geometric(m2, m)
            if s1 < s2:
                lo = m1
            else:
                hi = m2
        
        b_opt = (lo + hi) / 2.0
        s_opt = score_geometric(b_opt, m)
        
        if s_opt > best_score:
            best_score = s_opt
            U = [0]
            val = 1.0
            for k in range(1, m + 1):
                U.append(int(val))
                val *= b_opt
            best_set = sorted(set(U))
    
    return best_set, best_score

def arithmetic_search(deadline):
    """Search over arithmetic progressions."""
    best_score = -1
    best_set = None
    
    for d in range(1, 15):
        if time.time() > deadline - 10:
            break
        for a in range(0, 10):
            for m in range(3, 12):
                s = score_arithmetic(a, d, m)
                if s > best_score:
                    best_score = s
                    U = set([0])
                    for k in range(m + 1):
                        U.add(a + k * d)
                    best_set = sorted(U)
    
    return best_set, best_score

def generate_discrete_candidates(deadline):
    """Generate diverse discrete candidate sets."""
    candidates = []
    
    # Geometric progressions with integer bases
    for B in range(2, 12):
        for K in range(3, 14):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            if time.time() > deadline - 10:
                return candidates
            candidates.append(U)
    
    # Fibonacci-like
    for K in range(3, 12):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
    
    # Randomized geometric
    random.seed(42)
    for _ in range(40):
        B = random.uniform(1.5, 5.0)
        K = random.randint(4, 14)
        U = [0]
        val = 1.0
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        candidates.append(U)
    
    # Power-of-2 variants
    for K in range(4, 12):
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(val)
            val *= 2
        candidates.append(U)
    
    return candidates

def local_search(start_set, deadline, max_iter=3000):
    """Enhanced local search with diverse move operators."""
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False
    
    def try_swap(i, j, new_val_i, new_val_j):
        """Swap two elements with new values."""
        nonlocal current, current_score, best_set, best_score
        new_set = current[:]
        new_set[i] = new_val_i
        new_set[j] = new_val_j
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
            return True
        return False
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 100
    
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        moved = False
        prev_score = current_score
        
        # For small sets, try all single-element changes
        if len(current) <= 10:
            # Try moving each element to all reasonable positions
            for i in range(1, len(current)):
                if time.time() > deadline - 5:
                    break
                old_val = current[i]
                # Try a range of new values
                for new_val in range(1, min(old_val + 10, 30)):
                    if new_val == old_val:
                        continue
                    new_set = current[:]
                    new_set[i] = new_val
                    new_set = sorted(set(new_set))
                    if 0 in new_set and len(new_set) >= 2:
                        new_score = score_set(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current[:]
                            moved = True
                            break
                if moved:
                    break
            
            # Try adding a new element
            if not moved and len(current) < 12:
                for new_val in range(1, 30):
                    if new_val in current:
                        continue
                    new_set = current + [new_val]
                    new_set = sorted(new_set)
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        moved = True
                        break
            
            # Try removing an element
            if not moved and len(current) > 3:
                for i in range(1, len(current)):
                    new_set = [x for j, x in enumerate(current) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        moved = True
                        break
        else:
            # For larger sets, use random perturbations
            # 1. Move one element (small shifts)
            if len(current) > 1 and random.random() < 0.4:
                i = random.randint(1, len(current) - 1)
                old_val = current[i]
                deltas = [-5, -3, -2, -1, 1, 2, 3, 5]
                random.shuffle(deltas)
                for delta in deltas[:3]:
                    new_val = old_val + delta
                    if new_val > 0:
                        new_set = current[:]
                        new_set[i] = new_val
                        if try_move(new_set):
                            moved = True
                            break
            
            # 2. Add an element
            if len(current) < 25 and random.random() < 0.25:
                max_val = current[-1]
                candidates = []
                for x in current:
                    candidates.extend([x + 1, x - 1, x * 2, x + random.randint(1, 8)])
                candidates.append(max_val + random.randint(1, 15))
                candidates.append(random.randint(1, max_val + 10))
                
                random.shuffle(candidates)
                for new_val in candidates[:5]:
                    if new_val > 0 and new_val not in current:
                        new_set = current + [new_val]
                        if try_move(new_set):
                            moved = True
                            break
            
            # 3. Remove an element
            if len(current) > 5 and random.random() < 0.15:
                i = random.randint(1, len(current) - 1)
                new_set = [x for j, x in enumerate(current) if j != i]
                if try_move(new_set):
                    moved = True
            
            # 4. Swap two elements
            if len(current) > 3 and random.random() < 0.15:
                i, j = random.sample(range(1, len(current)), 2)
                vi, vj = current[i], current[j]
                # Try small perturbations
                di = random.choice([-3, -2, -1, 1, 2, 3])
                dj = random.choice([-3, -2, -1, 1, 2, 3])
                nvi, nvj = vi + di, vj + dj
                if nvi > 0 and nvj > 0:
                    if try_swap(i, j, nvi, nvj):
                        moved = True
            
            # 5. Structural move: replace element with geometric extension
            if len(current) > 2 and random.random() < 0.1:
                i = random.randint(1, len(current) - 1)
                base = current[0] if len(current) > 1 else 1
                if base > 0:
                    new_val = int(current[i] * random.uniform(1.5, 3.0))
                    if new_val > 0 and new_val not in current:
                        new_set = current[:]
                        new_set[i] = new_val
                        if try_move(new_set):
                            moved = True
        
        # Track no-improve streak for adaptive behavior
        if current_score > prev_score + 1e-9:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, try a bigger jump
        if no_improve_count > max_no_improve and random.random() < 0.3:
            i = random.randint(1, len(current) - 1)
            new_val = current[i] + random.randint(-10, 10)
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) > 1:
                    new_score = score_set(new_set)
                    if new_score > current_score - 0.02:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        moved = True
                        no_improve_count = 0
    
    return best_set, best_score

def construct():
    deadline = time.time() + 110  # Safety margin for 150s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Phase 1: Exhaustive small-set search
    small_set, small_score = exhaustive_small_search(deadline)
    if small_score > best_score:
        best_score = small_score
        best_set = small_set[:]
    
    # Phase 2: Continuous geometric search
    geo_set, geo_score = continuous_geometric_search(deadline)
    if geo_set is not None and geo_score > best_score:
        best_set = geo_set[:]
        best_score = geo_score
    
    # Phase 3: Arithmetic progression search
    arith_set, arith_score = arithmetic_search(deadline)
    if arith_set is not None and arith_score > best_score:
        best_set = arith_set[:]
        best_score = arith_score
    
    # Phase 4: Generate and evaluate discrete candidates
    candidates = generate_discrete_candidates(deadline)
    
    scored_candidates = []
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 5: Multi-start local search
    seen_structures = set()
    top_starts = []
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 5:
            break
    
    # Include all found results
    if small_set is not None and tuple(small_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(small_set)
    if geo_set is not None and tuple(geo_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(geo_set)
    if arith_set is not None and tuple(arith_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(arith_set)
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
    
    # Allocate remaining time across local searches
    remaining_time = deadline - time.time() - 5
    if remaining_time > 20:
        time_per_search = max(15, remaining_time / max(1, len(top_starts)))
        
        for idx, start in enumerate(top_starts):
            if time.time() > deadline - 8:
                break
            search_deadline = min(deadline - 5, time.time() + time_per_search)
            local_best, local_score = local_search(start, search_deadline, max_iter=2000)
            if local_score > best_score:
                best_score = local_score
                best_set = local_best[:]
    
    # Final polish: one more local search on the best found
    if time.time() < deadline - 10:
        final_best, final_score = local_search(best_set, deadline - 3, max_iter=1500)
        if final_score > best_score:
            best_score = final_score
            best_set = final_best[:]
    
    return best_set

# EVOLVE-BLOCK-END
