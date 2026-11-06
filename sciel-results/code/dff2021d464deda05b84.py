"""Minimize monochromatic 4-term arithmetic progressions in binary coloring of [1..256]."""
import numpy as np
import time
import random

_N = 256

# Precompute all 4-term APs as index arrays
_AP_A = []
_AP_B = []
_AP_C = []
_AP_D = []
for d in range(1, _N // 3 + 1):
    for a in range(_N - 3 * d):
        _AP_A.append(a)
        _AP_B.append(a + d)
        _AP_C.append(a + 2 * d)
        _AP_D.append(a + 3 * d)

_AP_A = np.array(_AP_A, dtype=np.int32)
_AP_B = np.array(_AP_B, dtype=np.int32)
_AP_C = np.array(_AP_C, dtype=np.int32)
_AP_D = np.array(_AP_D, dtype=np.int32)
_NUM_APS = len(_AP_A)

# Precompute for each position, the list of AP indices that involve it
# and the role (0,1,2,3) in that AP
_pos_to_aps = [[] for _ in range(_N)]
for ap_idx in range(_NUM_APS):
    for role in range(4):
        pos = [_AP_A[ap_idx], _AP_B[ap_idx], _AP_C[ap_idx], _AP_D[ap_idx]][role]
        _pos_to_aps[pos].append((ap_idx, role))

# Precompute AP index arrays for each position for fast delta computation
_pos_ap_indices = [np.array([x[0] for x in _pos_to_aps[i]], dtype=np.int32) if _pos_to_aps[i] else np.array([], dtype=np.int32) for i in range(_N)]
_pos_ap_roles = [np.array([x[1] for x in _pos_to_aps[i]], dtype=np.int32) if _pos_to_aps[i] else np.array([], dtype=np.int32) for i in range(_N)]

def _score(arr):
    """Count monochromatic 4-term APs. arr: numpy int8 array of 0/1, length 256."""
    c0 = arr[_AP_A]
    c1 = arr[_AP_B]
    c2 = arr[_AP_C]
    c3 = arr[_AP_D]
    mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
    return int(np.sum(mask))

def _delta_flip(arr, i):
    """
    Compute the change in monochromatic 4-AP count if we flip position i.
    Returns delta (vectorized).
    """
    ap_idxs = _pos_ap_indices[i]
    if len(ap_idxs) == 0:
        return 0
    
    a = _AP_A[ap_idxs]
    b = _AP_B[ap_idxs]
    c = _AP_C[ap_idxs]
    d = _AP_D[ap_idxs]
    roles = _pos_ap_roles[i]
    
    c0 = arr[a]
    c1 = arr[b]
    c2 = arr[c]
    c3 = arr[d]
    
    before_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Compute after_mono for each role
    flipped = 1 - arr[i]
    after_mono = np.zeros(len(ap_idxs), dtype=bool)
    
    m0 = roles == 0
    m1 = roles == 1
    m2 = roles == 2
    m3 = roles == 3
    
    after_mono[m0] = (flipped == c1[m0]) & (c1[m0] == c2[m0]) & (c2[m0] == c3[m0])
    after_mono[m1] = (c0[m1] == flipped) & (flipped == c2[m1]) & (c2[m1] == c3[m1])
    after_mono[m2] = (c0[m2] == c1[m2]) & (c1[m2] == flipped) & (flipped == c3[m2])
    after_mono[m3] = (c0[m3] == c1[m3]) & (c1[m3] == c2[m3]) & (c2[m3] == flipped)
    
    delta = int(np.sum(~before_mono & after_mono) - np.sum(before_mono & ~after_mono))
    return delta

def _delta_flip_pair(arr, i, j):
    """
    Compute the change in monochromatic 4-AP count if we flip positions i and j simultaneously.
    Returns delta.
    """
    if i == j:
        return 0
    
    # Get all APs involving either i or j
    ap_set_i = set(_pos_ap_indices[i].tolist())
    ap_set_j = set(_pos_ap_indices[j].tolist())
    all_aps = list(ap_set_i | ap_set_j)
    
    if not all_aps:
        return 0
    
    all_aps = np.array(all_aps, dtype=np.int32)
    
    a = _AP_A[all_aps]
    b = _AP_B[all_aps]
    c = _AP_C[all_aps]
    d = _AP_D[all_aps]
    
    c0 = arr[a]
    c1 = arr[b]
    c2 = arr[c]
    c3 = arr[d]
    
    before_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Determine which positions are affected
    # For each AP, check if it contains i or j
    affected_pos = np.zeros((len(all_aps), 4), dtype=np.int8)
    
    # Check role 0
    m_i0 = (a == i)
    m_j0 = (a == j)
    affected_pos[m_i0, 0] = 1
    affected_pos[m_j0, 0] = 1
    
    # Check role 1
    m_i1 = (b == i)
    m_j1 = (b == j)
    affected_pos[m_i1, 1] = 1
    affected_pos[m_j1, 1] = 1
    
    # Check role 2
    m_i2 = (c == i)
    m_j2 = (c == j)
    affected_pos[m_i2, 2] = 1
    affected_pos[m_j2, 2] = 1
    
    # Check role 3
    m_i3 = (d == i)
    m_j3 = (d == j)
    affected_pos[m_i3, 3] = 1
    affected_pos[m_j3, 3] = 1
    
    # Compute after colors
    after_c0 = np.where(affected_pos[:, 0] == 1, 1 - c0, c0)
    after_c1 = np.where(affected_pos[:, 1] == 1, 1 - c1, c1)
    after_c2 = np.where(affected_pos[:, 2] == 1, 1 - c2, c2)
    after_c3 = np.where(affected_pos[:, 3] == 1, 1 - c3, c3)
    
    after_mono = (after_c0 == after_c1) & (after_c1 == after_c2) & (after_c2 == after_c3)
    
    delta = int(np.sum(~before_mono & after_mono) - np.sum(before_mono & ~after_mono))
    return delta

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _score(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    random.seed(42)
    
    # Generate several initial colorings
    candidates = []
    
    # 1. Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    tm = [thue_morse(i) for i in range(_N)]
    candidates.append(tm)
    
    # 2. Alternating
    alt = [i % 2 for i in range(_N)]
    candidates.append(alt)
    
    # 3. Block patterns
    for block_size in [2, 3, 4, 5, 6, 7, 8, 12, 16, 21, 32, 64]:
        block = [(i // block_size) % 2 for i in range(_N)]
        candidates.append(block)
    
    # 4. Random initializations
    for seed in range(40):
        random.seed(seed)
        rc = [random.randint(0, 1) for _ in range(_N)]
        candidates.append(rc)
    
    # 5. Greedy construction: at each step, choose color that minimizes new mono-4-APs
    def greedy_construct(seed=0):
        random.seed(seed)
        arr = [-1] * _N
        for i in range(_N):
            best_color = 0
            best_count = float('inf')
            for color in [0, 1]:
                arr[i] = color
                count = 0
                for ap_idx, role in _pos_to_aps[i]:
                    a = _AP_A[ap_idx]
                    b = _AP_B[ap_idx]
                    c = _AP_C[ap_idx]
                    d = _AP_D[ap_idx]
                    if arr[a] != -1 and arr[b] != -1 and arr[c] != -1 and arr[d] != -1:
                        if arr[a] == arr[b] == arr[c] == arr[d]:
                            count += 1
                if count < best_count:
                    best_count = count
                    best_color = color
            arr[i] = best_color
        return arr
    
    for seed in range(8):
        candidates.append(greedy_construct(seed))
    
    # 6. Noise-injected Thue-Morse variants
    for noise_level in [0.05, 0.1, 0.15, 0.2]:
        for seed in range(3):
            random.seed(1000 + seed)
            tm_noisy = tm[:]
            for i in range(_N):
                if random.random() < noise_level:
                    tm_noisy[i] = 1 - tm_noisy[i]
            candidates.append(tm_noisy)
    
    # Evaluate all candidates
    best_colours = None
    best_score = float('inf')
    
    for cand in candidates:
        arr = np.array(cand, dtype=np.int8)
        s = _score(arr)
        if s < best_score:
            best_score = s
            best_colours = cand[:]
    
    # Local search with single-flip and pair-flip moves
    def local_search(colours, max_time):
        """Hill-climbing with single-flip and pair-flip delta evaluation."""
        arr = np.array(colours, dtype=np.int8)
        current_score = _score(arr)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Single flips
            for i in range(_N):
                if time.time() > max_time:
                    break
                delta = _delta_flip(arr, i)
                if delta < 0:
                    arr[i] = 1 - arr[i]
                    current_score += delta
                    improved = True
            # Pair flips (sampled to keep time bounded)
            if time.time() < max_time:
                for _ in range(200):
                    if time.time() > max_time:
                        break
                    i, j = random.sample(range(_N), 2)
                    delta = _delta_flip_pair(arr, i, j)
                    if delta < 0:
                        arr[i] = 1 - arr[i]
                        arr[j] = 1 - arr[j]
                        current_score += delta
                        improved = True
        return arr.tolist(), current_score
    
    # Run local search from best candidate
    if time.time() < deadline - 30.0:
        best_colours, best_score = local_search(best_colours, deadline - 30.0)
    
    # Simulated annealing with mixed single and pair moves
    if time.time() < deadline - 10.0:
        current = np.array(best_colours, dtype=np.int8)
        current_score = best_score
        temp = 8.0
        cooling = 0.997
        pair_prob = 0.3
        
        while time.time() < deadline - 5.0 and temp > 0.1:
            if random.random() < pair_prob:
                i, j = random.sample(range(_N), 2)
                delta = _delta_flip_pair(current, i, j)
                current[i] = 1 - current[i]
                current[j] = 1 - current[j]
                new_score = current_score + delta
                
                if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_colours = current.tolist()
                else:
                    current[i] = 1 - current[i]
                    current[j] = 1 - current[j]
            else:
                i = random.randint(0, _N - 1)
                delta = _delta_flip(current, i)
                current[i] = 1 - current[i]
                new_score = current_score + delta
                
                if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_colours = current.tolist()
                else:
                    current[i] = 1 - current[i]
            
            temp *= cooling
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        best_colours, best_score = local_search(best_colours, deadline - 2.0)
    
    # Verify
    final_arr = np.array(best_colours, dtype=np.int8)
    final_score = _score(final_arr)
    if final_score != best_score:
        best_score = final_score
    
    return best_colours
# EVOLVE-BLOCK-END
