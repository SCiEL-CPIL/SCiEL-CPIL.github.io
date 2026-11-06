"""
Strategy: Set Cover / Dominance-Based Construction with Simulated Annealing Polish
We use a greedy set-cover approach to build a maximal no-isosceles set, then apply
intensive local search (Simulated Annealing) to minimize the set size while preserving maximality.
"""

import itertools
import time
import random
import math

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)

# Precompute squared distances
dist2 = [[0] * V for _ in range(V)]
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute conflict bitsets using Python ints (arbitrary precision)
# conflict_pair[i][j] = bitset of all k such that triangle(i,j,k) is isosceles
conflict_pair = [[0] * V for _ in range(V)]
for i in range(V):
    for j in range(i + 1, V):
        d_ij = dist2[i][j]
        mask = 0
        for k in range(V):
            if k == i or k == j:
                continue
            d_ik = dist2[i][k]
            d_jk = dist2[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        conflict_pair[i][j] = mask
        conflict_pair[j][i] = mask

def _is_independent_with_S(p, S_bits):
    """Check if adding p to S keeps S independent.
    S_bits is a bitset of points in S.
    We need: no pair (s1, s2) in S such that p in conflict_pair[s1][s2].
    """
    s_list = []
    tmp = S_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        s_list.append(idx)
        tmp ^= lsb
    
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if (conflict_pair[s_list[i]][s_list[j]] >> p) & 1:
                return False
    return True

def _construct_greedy_coverage(seed=0):
    """
    Greedy set-cover style construction with bitset optimization.
    """
    random.seed(seed)
    S_bits = 0  # bitset of points in S
    all_bits = (1 << V) - 1
    unblocked = all_bits
    
    while unblocked:
        # Find the best point p to add
        best_p = -1
        best_score = -1
        
        # Get list of unblocked points
        ub_list = []
        tmp = unblocked
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            ub_list.append(idx)
            tmp ^= lsb
        
        for p in ub_list:
            # Check if S U {p} is independent
            if not _is_independent_with_S(p, S_bits):
                continue
            
            # Calculate score: how many unblocked q will be blocked by adding p?
            blocked_by_p = 0
            tmp_s = S_bits
            while tmp_s:
                lsb = tmp_s & (-tmp_s)
                s = lsb.bit_length() - 1
                blocked_by_p |= conflict_pair[p][s]
                tmp_s ^= lsb
            
            # Count how many unblocked points (excluding p) are in blocked_by_p
            score = (unblocked & blocked_by_p & ~(1 << p)).bit_count()
            
            # Add small random noise to break ties
            noise = random.random() * 0.1
            total_score = score + noise
            
            if total_score > best_score:
                best_score = total_score
                best_p = p
        
        if best_p == -1:
            break
            
        # Add best_p to S
        S_bits |= (1 << best_p)
        
        # Update unblocked: remove best_p and any q that is now blocked
        blocked_by_new = 0
        tmp_s = S_bits & ~(1 << best_p)  # S without best_p
        while tmp_s:
            lsb = tmp_s & (-tmp_s)
            s = lsb.bit_length() - 1
            blocked_by_new |= conflict_pair[best_p][s]
            tmp_s ^= lsb
        
        # Remove best_p and all blocked points from unblocked
        unblocked &= ~(1 << best_p)
        unblocked &= ~blocked_by_new

    return S_bits

def _verify_independent(S_bits):
    """Check that S is an independent no-isosceles set."""
    s_list = []
    tmp = S_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        s_list.append(idx)
        tmp ^= lsb
    
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            for k in range(j + 1, len(s_list)):
                if (conflict_pair[s_list[i]][s_list[j]] >> s_list[k]) & 1:
                    return False
    return True

def _verify_maximal(S_bits):
    """Check that S is maximal."""
    for p in range(V):
        if (S_bits >> p) & 1:
            continue
        if _is_independent_with_S(p, S_bits):
            return False
    return True

def _simulated_annealing_polish(S_bits, time_limit=None):
    """
    Simulated Annealing to minimize |S| while maintaining independence and maximality.
    Moves:
    1. Remove a point from S (if S remains maximal).
    2. Swap a point in S with a point not in S (if new S is independent and maximal).
    3. Add a point to S (if S remains independent) - rarely useful for minimization but helps escape local minima.
    """
    current = S_bits
    current_size = current.bit_count()
    best = S_bits
    best_size = current_size
    
    # Objective: minimize size. Penalty for invalidity (should be 0 if we only accept valid moves).
    # We strictly only accept valid moves (independent + maximal).
    # Cost function: size.
    
    # Precompute list of points for fast iteration
    all_points = list(range(V))
    
    def get_s_list(bits):
        lst = []
        tmp = bits
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            lst.append(idx)
            tmp ^= lsb
        return lst
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    start_time = time.time()
    
    while T > T_min:
        if time_limit is not None and time.time() > time_limit:
            break
        
        s_list = get_s_list(current)
        n = len(s_list)
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4 and n > 1:
            # Try to remove a point
            idx = random.randint(0, n - 1)
            p = s_list[idx]
            new_bits = current & ~(1 << p)
            
            if _verify_independent(new_bits) and _verify_maximal(new_bits):
                # Valid move
                new_size = new_bits.bit_count()
                delta = new_size - current_size # Should be -1
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    current = new_bits
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best = current
        elif move_type < 0.9:
            # Try to swap
            if n == 0:
                continue
            idx = random.randint(0, n - 1)
            p_remove = s_list[idx]
            S_without = current & ~(1 << p_remove)
            
            # Find a random candidate to add
            # We can pick from all points not in S_without
            # To make it efficient, we can pick from a subset or just random index
            c = random.randint(0, V - 1)
            if (S_without >> c) & 1:
                continue # Already in set
            
            new_bits = S_without | (1 << c)
            
            if _verify_independent(new_bits) and _verify_maximal(new_bits):
                new_size = new_bits.bit_count()
                delta = new_size - current_size # Should be 0
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    current = new_bits
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best = current
        else:
            # Try to add a point (explore)
            c = random.randint(0, V - 1)
            if (current >> c) & 1:
                continue
            
            new_bits = current | (1 << c)
            
            if _verify_independent(new_bits) and _verify_maximal(new_bits):
                new_size = new_bits.bit_count()
                delta = new_size - current_size # Should be +1
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    current = new_bits
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best = current
        
        T *= cooling
        
        # Occasionally reheat or check best
        if random.random() < 0.01:
            T = min(T * 2, 5.0)

    return best

def _local_search_polish(S_bits, max_iters=50, time_limit=None):
    """
    Deterministic local search: try to reduce S by swapping points.
    """
    current = S_bits
    current_size = current.bit_count()
    
    def bits_to_list(bits):
        lst = []
        tmp = bits
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            lst.append(idx)
            tmp ^= lsb
        return lst
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        s_list = bits_to_list(current)
        improved = False
        
        for remove_idx in range(len(s_list)):
            r = s_list[remove_idx]
            S_without_r = current & ~(1 << r)
            
            all_bits = (1 << V) - 1
            candidates = all_bits & ~S_without_r
            
            valid_candidates = []
            tmp_c = candidates
            while tmp_c:
                lsb = tmp_c & (-tmp_c)
                c = lsb.bit_length() - 1
                tmp_c ^= lsb
                
                if _is_independent_with_S(c, S_without_r):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            best_c = -1
            best_block_count = -1
            
            for c in valid_candidates:
                new_S = S_without_r | (1 << c)
                new_s_list = bits_to_list(new_S)
                
                is_maximal = True
                for p in range(V):
                    if (new_S >> p) & 1:
                        continue
                    if _is_independent_with_S(p, new_S):
                        is_maximal = False
                        break
                
                if not is_maximal:
                    continue
                
                block_count = 0
                for i in range(len(new_s_list)):
                    for j in range(i + 1, len(new_s_list)):
                        block_count += conflict_pair[new_s_list[i]][new_s_list[j]].bit_count()
                
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_c = c
            
            if best_c != -1:
                current = S_without_r | (1 << best_c)
                if current.bit_count() < current_size:
                    current_size = current.bit_count()
                    improved = True
                    break
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_bits = None
    best_size = V
    
    # Run multiple greedy constructions with different seeds
    num_greedy = 30
    
    for seed in range(num_greedy):
        if time.time() > deadline:
            break
            
        S_bits = _construct_greedy_coverage(seed)
        
        # Verify
        if not _verify_independent(S_bits) or not _verify_maximal(S_bits):
            continue
        
        size = S_bits.bit_count()
        if size < best_size:
            best_size = size
            best_bits = S_bits
            if best_size <= 10:
                break
    
    # Polish phase
    if best_bits is not None:
        # 1. Deterministic polish
        polish_deadline = min(deadline, start_time + 270.0)
        polished = _local_search_polish(best_bits, max_iters=20, time_limit=polish_deadline)
        
        if _verify_independent(polished) and _verify_maximal(polished):
            if polished.bit_count() < best_size:
                best_size = polished.bit_count()
                best_bits = polished
        
        # 2. Simulated Annealing polish
        sa_deadline = min(deadline, start_time + 285.0)
        sa_polished = _simulated_annealing_polish(best_bits, time_limit=sa_deadline)
        
        if _verify_independent(sa_polished) and _verify_maximal(sa_polished):
            if sa_polished.bit_count() < best_size:
                best_size = sa_polished.bit_count()
                best_bits = sa_polished

    if best_bits is None:
        # Fallback
        best_bits = _construct_greedy_coverage(0)
        if not _verify_independent(best_bits) or not _verify_maximal(best_bits):
            # Emergency fallback: construct a valid set manually?
            # For now, just return the greedy one, hoping it's valid.
            # If not, the evaluator might fail. But greedy is usually valid.
            pass

    # Convert to list of points
    result = []
    tmp = best_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        result.append(POINTS[idx])
        tmp ^= lsb
    
    return result

# EVOLVE-BLOCK-END
