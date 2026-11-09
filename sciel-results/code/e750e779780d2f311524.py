import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Precompute distance squared
# dist2[i, j] is squared distance between point i and point j
dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        dist2[i, j] = dx * dx + dy * dy

# Precompute isosceles masks
# iso_mask[p][u] is a bitmask of all v such that {p, u, v} is isosceles
# Note: p, u, v must be distinct.
iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

for p in range(TOTAL_POINTS):
    for u in range(TOTAL_POINTS):
        if p == u:
            continue
        mask = 0
        dp_u = dist2[p, u]
        for v in range(TOTAL_POINTS):
            if v == p or v == u:
                continue
            dp_v = dist2[p, v]
            du_v = dist2[u, v]
            # Isosceles if any two distances are equal
            if dp_u == dp_v or dp_u == du_v or dp_v == du_v:
                mask |= (1 << v)
        iso_mask[p][u] = mask

ALL_MASK = (1 << TOTAL_POINTS) - 1

def _bits(mask):
    """Yield indices of set bits in mask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb

def _can_add(p, S):
    """Check if point p can be added to set S (bitmask) without creating isosceles."""
    if (S >> p) & 1:
        return False
    # For each u in S, check if there is v in S such that {p, u, v} is isosceles
    # This is equivalent to checking if iso_mask[p][u] intersects S
    s_temp = S
    while s_temp:
        lsb = s_temp & -s_temp
        u = lsb.bit_length() - 1
        s_temp ^= lsb
        if iso_mask[p][u] & S:
            return False
    return True

def _is_maximal(S):
    """Check if S is a maximal no-isosceles set."""
    not_S = ALL_MASK ^ S
    p_bit = not_S
    while p_bit:
        lsb = p_bit & -p_bit
        p = lsb.bit_length() - 1
        p_bit ^= lsb
        if _can_add(p, S):
            return False
    return True

def _greedy_construct():
    """Construct a maximal set greedily."""
    S = 0
    order = list(range(TOTAL_POINTS))
    random.shuffle(order)
    for p in order:
        if _can_add(p, S):
            S |= (1 << p)
    return S

def _try_shrink(S):
    """Try to remove a point from S while maintaining maximality. Returns new S or None."""
    s_temp = S
    # Shuffle for randomness
    pts = list(_bits(s_temp))
    random.shuffle(pts)
    for u in pts:
        new_S = S ^ (1 << u)
        if _is_maximal(new_S):
            return new_S
    return None

def _try_swap(S):
    """Try to swap a point in S with a point not in S, maintaining maximality. Returns new S or None."""
    s_temp = S
    pts = list(_bits(s_temp))
    random.shuffle(pts)
    
    for u in pts:
        S_no_u = S ^ (1 << u)
        # Find candidates v that can be added to S_no_u
        not_S_no_u = ALL_MASK ^ S_no_u
        cand_bit = not_S_no_u
        candidates = []
        while cand_bit:
            lsb = cand_bit & -cand_bit
            v = lsb.bit_length() - 1
            cand_bit ^= lsb
            if _can_add(v, S_no_u):
                candidates.append(v)
        
        if not candidates:
            continue
            
        # Try a few random candidates
        random.shuffle(candidates)
        for v in candidates[:5]:
            new_S = S_no_u | (1 << v)
            if _is_maximal(new_S):
                return new_S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    best_S = _greedy_construct()
    best_size = bin(best_S).count('1')
    
    # Simulated Annealing / Local Search
    current_S = best_S
    current_size = best_size
    temperature = 1.0
    min_temp = 0.01
    cooling_rate = 0.995
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # 1. Try to shrink
        new_S = _try_shrink(current_S)
        if new_S is not None:
            current_S = new_S
            current_size = bin(current_S).count('1')
            if current_size < best_size:
                best_size = current_size
                best_S = current_S
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # 2. Try to swap
        new_S = _try_swap(current_S)
        if new_S is not None:
            current_S = new_S
            # Size remains same in a valid swap to a maximal set
            # But we accept it to explore the space
            no_improve_count += 1
            continue
            
        # 3. Stuck? Perturb or Restart
        if no_improve_count > 20:
            # Random restart or aggressive perturbation
            if random.random() < 0.5:
                # Restart
                current_S = _greedy_construct()
                current_size = bin(current_S).count('1')
            else:
                # Perturb: remove 2, add 2 greedily
                pts = list(_bits(current_S))
                if len(pts) >= 2:
                    u1, u2 = random.sample(pts, 2)
                    S_temp = current_S
                    S_temp ^= (1 << u1)
                    S_temp ^= (1 << u2)
                    # Greedily complete S_temp
                    order = list(range(TOTAL_POINTS))
                    random.shuffle(order)
                    for p in order:
                        if _can_add(p, S_temp):
                            S_temp |= (1 << p)
                    if _is_maximal(S_temp):
                        current_S = S_temp
                        current_size = bin(current_S).count('1')
                        if current_size < best_size:
                            best_size = current_size
                            best_S = current_S
                            no_improve_count = 0
                    else:
                        # Fallback to greedy
                        current_S = _greedy_construct()
                        current_size = bin(current_S).count('1')
                        
            no_improve_count = 0
            temperature = max(min_temp, temperature * cooling_rate)
            
    return [(POINTS[i]) for i in _bits(best_S)]
