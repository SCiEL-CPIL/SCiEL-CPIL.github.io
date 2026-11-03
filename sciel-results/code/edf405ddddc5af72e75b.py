import time
import random
import math
import numpy as np

def _score(U):
    """Compute the score for a given set U."""
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    u_arr = np.array(U, dtype=np.int64)
    n = len(u_arr)
    
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _score_fast(U_list):
    """Fast score using numpy - U_list is a list of ints."""
    if not U_list:
        return 0.0
    max_u = max(U_list)
    if max_u == 0:
        return 1.0
    u_arr = np.array(U_list, dtype=np.int64)
    n = len(u_arr)
    
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _generate_structural_starts():
    """Generate diverse structural starting sets based on algebraic families."""
    starts = []
    
    # 1. Geometric Progressions (the primary structural family from inspiration)
    for B in range(2, 15):
        for K in range(3, 12):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            starts.append(sorted(U))
            
    # 2. "Powers of B" with offset or scaling
    for B in range(2, 10):
        for K in range(3, 10):
            for offset in range(0, 3):
                U = [offset]
                val = offset + 1
                for i in range(1, K):
                    U.append(val)
                    val = val * B + offset
                if 0 not in U:
                    U = sorted(set(U + [0]))
                starts.append(U)
                
    # 3. Fibonacci-like
    for K in range(3, 10):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        starts.append(sorted(U))
        
    # 4. Randomized geometric (continuous base)
    random.seed(42)
    for _ in range(50):
        B = random.uniform(1.5, 5.0)
        K = random.randint(4, 12)
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        starts.append(U)
        
    # 5. Quadratic sequences (from current program)
    for a in [1, 2, 3]:
        for n in range(5, 12):
            s = [0]
            for i in range(1, n):
                s.append(a * i * i)
            starts.append(sorted(s))

    # 6. Arithmetic with gaps (from current program)
    for step in [2, 3, 5]:
        for n in range(5, 15):
            s = [0]
            for i in range(1, n):
                s.append(step * i)
            starts.append(sorted(s))
            
    return starts

def _continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    best_score = -1.0
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.5, 5.0, 20)
    m_grid = range(3, 15)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            # Construct U
            U = [0]
            val = 1.0
            for k in range(1, m + 1):
                U.append(int(val))
                val *= b
            U = sorted(set(U))
            if len(U) < 2:
                continue
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return [0, 1, 3], 1.0
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.5, b0 - 0.5)
    b_hi = min(5.0, b0 + 0.5)
    
    lo = b_lo
    hi = b_hi
    for _ in range(40):  # 40 iterations for fine-grained search
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        
        # Score at m1
        U1 = [0]
        val = 1.0
        for k in range(1, m0 + 1):
            U1.append(int(val))
            val *= m1
        U1 = sorted(set(U1))
        s1 = _score_fast(U1) if len(U1) >= 2 else -1.0
        
        # Score at m2
        U2 = [0]
        val = 1.0
        for k in range(1, m0 + 1):
            U2.append(int(val))
            val *= m2
        U2 = sorted(set(U2))
        s2 = _score_fast(U2) if len(U2) >= 2 else -1.0
        
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    
    # Construct optimal U for this m0
    U_opt = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U_opt.append(int(val))
        val *= b_opt
    U_opt = sorted(set(U_opt))
    s_opt = _score_fast(U_opt)
    
    # Also try adjacent m values
    for m_adj in [m0 - 1, m0 + 1]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        # Quick ternary search for this m
        lo2 = max(1.5, b_opt - 0.3)
        hi2 = min(5.0, b_opt + 0.3)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            
            U1 = [0]
            val = 1.0
            for k in range(1, m_adj + 1):
                U1.append(int(val))
                val *= m1
            U1 = sorted(set(U1))
            s1 = _score_fast(U1) if len(U1) >= 2 else -1.0
            
            U2 = [0]
            val = 1.0
            for k in range(1, m_adj + 1):
                U2.append(int(val))
                val *= m2
            U2 = sorted(set(U2))
            s2 = _score_fast(U2) if len(U2) >= 2 else -1.0
            
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        
        b_adj = (lo2 + hi2) / 2.0
        
        U_adj = [0]
        val = 1.0
        for k in range(1, m_adj + 1):
            U_adj.append(int(val))
            val *= b_adj
        U_adj = sorted(set(U_adj))
        s_adj = _score_fast(U_adj)
        
        if s_adj > s_opt:
            s_opt = s_adj
            U_opt = U_adj
    
    return U_opt, s_opt

def _local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation based on arithmetic neighborhoods.
    """
    U = sorted(set(U))
    current_score = _score_fast(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
        else:
            # Try to MOVE an element
            if len(U) > 3 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U or c == old_val:
                        continue
                    new_U = list(U)
                    new_U[idx] = c
                    new_U.sort()
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
    
    return best_local_set, best_local_score

def _generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This is the structural improvement: exploring the arithmetic neighborhood.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def _simulated_annealing(start_set, deadline, initial_temp=100.0, cooling=0.95):
    """Simulated annealing to optimize the set."""
    current = list(start_set)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    
    temp = initial_temp
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Choose a random move
        move_type = rng.choice(['add', 'remove', 'move'])
        
        if move_type == 'add':
            if len(current) > 25:
                continue
            max_c = max(current) if current else 0
            # Try adding near existing elements or at gaps
            if current and rng.random() < 0.7:
                base = rng.choice(current)
                x = base + rng.randint(1, max(5, max_c // 10))
            else:
                x = rng.randint(1, 2 * max_c + 20)
            
            if x in current or x < 0:
                continue
            new_set = current + [x]
            new_score = _score_fast(new_set)
            
        elif move_type == 'remove':
            if len(current) <= 3:
                continue
            x = rng.choice([y for y in current if y != 0])
            if x == 0:
                continue
            new_set = [y for y in current if y != x]
            new_score = _score_fast(new_set)
            
        else:  # move
            if len(current) <= 3:
                continue
            idx = rng.randint(1, len(current) - 1)  # don't move 0
            x = current[idx]
            max_c = max(current) if current else 0
            # Move to nearby or random position
            if rng.random() < 0.5:
                y = x + rng.randint(-max(3, x // 10), max(3, x // 10))
            else:
                y = rng.randint(1, 2 * max_c + 10)
            
            if y < 0 or y in current or y == x:
                continue
            new_set = list(current)
            new_set[idx] = y
            new_score = _score_fast(new_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (temp > 0.01 and rng.random() < math.exp(delta / temp)):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = [0, 1, 3]
    best_score = _score_fast(best_set)
    
    # Phase 1: Continuous geometric search (structural exploration)
    if time.time() < deadline - 60:
        geo_deadline = time.time() + 30
        geo_set, geo_score = _continuous_geometric_search(geo_deadline)
        if geo_score > best_score:
            best_score = geo_score
            best_set = geo_set
    
    # Phase 2: Generate diverse structural starting points
    starts = _generate_structural_starts()
    
    # Add the result from continuous search as a start
    starts.append(best_set)
    
    # Add some random starts
    for _ in range(10):
        size = random.randint(4, 12)
        s = {0}
        max_val = 100
        while len(s) < size:
            s.add(random.randint(1, max_val))
        starts.append(sorted(s))
    
    # Shuffle starts for diversity
    random.shuffle(starts)
    
    # Phase 3: Run local search with dynamic candidates on each start
    num_starts = len(starts)
    time_per_start = (deadline - time.time() - 10) / max(1, num_starts)
    
    for i, start in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on remaining budget
        alloc_time = min(time_per_start, remaining - 5)
        if alloc_time < 1:
            break
        
        ls_deadline = time.time() + alloc_time
        ls_set, ls_score = _local_search_dynamic(start, ls_deadline)
        
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 4: Final polish with simulated annealing on the best set
    if time.time() < deadline - 10:
        polish_deadline = deadline - 5
        polished, polish_score = _simulated_annealing(best_set, polish_deadline, initial_temp=10.0, cooling=0.99)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polished
    
    # Final sanity check
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return sorted(set(best_set))
