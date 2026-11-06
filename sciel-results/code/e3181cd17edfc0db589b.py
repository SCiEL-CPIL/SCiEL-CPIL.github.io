"""4-AP-free set in [1,256] maximizing 3-AP count, via vectorized SA."""
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin under 90s wall-clock

def _build_tables():
    """Precompute 3-AP pairs and 4-AP tuples as numpy arrays for speed."""
    # 3-AP: (a, c) with b = (a+c)//2, a < c, a+c even, a>=1, c<=256
    pairs = []
    for a in range(1, _N + 1):
        for c in range(a + 2, _N + 1, 2):  # c - a must be even => a+c even
            b = (a + c) // 2
            if 1 <= b <= _N:
                pairs.append((a - 1, c - 1))  # 0-indexed
    pairs = np.array(pairs, dtype=np.int32)

    # 4-AP: (a, a+d, a+2d, a+3d) 0-indexed
    quads = []
    for a in range(_N):
        for d in range(1, (_N - a) // 3 + 1):
            a3 = a + 3 * d
            if a3 < _N:
                quads.append((a, a + d, a + 2 * d, a3))
    quads = np.array(quads, dtype=np.int32)

    return pairs, quads

_PAIRS, _QUADS = _build_tables()
_NPAIRS = len(_PAIRS)
_NQUADS = len(_QUADS)

def _count_3aps(mask):
    """Count 3-APs using vectorized dot product."""
    if _NPAIRS == 0:
        return 0
    a = mask[_PAIRS[:, 0]]
    c = mask[_PAIRS[:, 1]]
    return int(np.dot(a, c))

def _count_4ap_violations(mask):
    """Count nonconstant 4-APs present in the set."""
    if _NQUADS == 0:
        return 0
    q = mask[_QUADS]
    return int(np.sum(np.prod(q, axis=1)))

def _greedy_seed():
    """Greedy construction: add elements that don't create 4-APs, prefer high 3-AP gain."""
    mask = np.zeros(_N, dtype=np.int32)
    for i in range(_N):
        # Try adding i; check if it creates any 4-AP
        # A 4-AP containing i: i could be at position 0,1,2,3
        # Check all quads containing i
        ok = True
        for q in _QUADS:
            if i in q:
                others = [x for x in q if x != i]
                if all(mask[x] for x in others):
                    ok = False
                    break
        if ok:
            mask[i] = 1
    return mask

def _local_search(mask, max_iter=5000, time_limit=2.0):
    """Hill-climbing: try single insertions and removals to improve 3-AP count."""
    cur_score = _count_3aps(mask)
    t0 = time.time()
    for _ in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        improved = False
        # Try inserting each absent element
        absent = np.where(mask == 0)[0]
        if len(absent) > 0:
            # Vectorized: for each absent element, check if insertion is valid
            # and compute 3-AP gain
            best_gain = -1
            best_idx = -1
            for idx in absent:
                # Check 4-AP violations if we add idx
                # Find quads containing idx
                # This is expensive per element; use a precomputed lookup
                valid = True
                # Check all quads containing idx
                for q in _QUADS:
                    if idx in q:
                        others = [x for x in q if x != idx]
                        if all(mask[x] for x in others):
                            valid = False
                            break
                if not valid:
                    continue
                # Compute 3-AP gain: pairs where one endpoint is idx and other is in mask
                # idx as 'a': need c such that (idx+c)//2 in mask, c > idx
                # idx as 'c': need a such that (a+idx)//2 in mask, a < idx
                gain = 0
                # idx as a (first term)
                for c in range(idx + 2, _N, 2):
                    b = (idx + c) // 2
                    if mask[b]:
                        gain += 1
                # idx as c (last term)
                for a in range(idx - 2, -1, -2):
                    b = (a + idx) // 2
                    if mask[b]:
                        gain += 1
                # idx as b (middle term)
                for d in range(1, _N):
                    a = idx - d
                    c = idx + d
                    if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx >= 0 and best_gain > 0:
                mask[best_idx] = 1
                cur_score += best_gain
                improved = True
        # Try removing elements that have low contribution
        present = np.where(mask == 1)[0]
        if len(present) > 0:
            best_loss = 999999
            best_remove = -1
            for idx in present:
                # Compute 3-AP loss if we remove idx
                loss = 0
                for c in range(idx + 2, _N, 2):
                    b = (idx + c) // 2
                    if mask[b]:
                        loss += 1
                for a in range(idx - 2, -1, -2):
                    b = (a + idx) // 2
                    if mask[b]:
                        loss += 1
                for d in range(1, _N):
                    a = idx - d
                    c = idx + d
                    if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                        loss += 1
                if loss < best_loss:
                    best_loss = loss
                    best_remove = idx
            # Only remove if it allows future improvements (heuristic: skip for now)
        if not improved:
            break
    return mask, cur_score

def _sa_search(mask, time_budget):
    """Simulated annealing with multiple move types."""
    cur_score = _count_3aps(mask)
    best_mask = mask.copy()
    best_score = cur_score
    t0 = time.time()
    
    # Precompute for each element, which quads it's in (for fast validity check)
    # element -> list of quad indices
    elem_quads = [[] for _ in range(_N)]
    for qi, q in enumerate(_QUADS):
        for elem in q:
            elem_quads[elem].append(qi)
    
    def _valid_insert(mask, idx):
        """Check if inserting idx creates no 4-AP."""
        for qi in elem_quads[idx]:
            q = _QUADS[qi]
            for x in q:
                if x != idx and not mask[x]:
                    break
            else:
                return False
        return True
    
    def _valid_remove(mask, idx):
        """Removing always valid (fewer 4-APs)."""
        return True
    
    def _swap_valid(mask, out_idx, in_idx):
        """Check if swapping out_idx->in_idx creates no 4-AP."""
        # Temporarily set out=0, in=1
        mask[out_idx] = 0
        mask[in_idx] = 1
        valid = _valid_insert(mask, in_idx)  # but out is already 0
        mask[in_idx] = 0
        mask[out_idx] = 1
        return valid
    
    T = 50.0
    T_min = 0.1
    alpha = 0.9997  # cooling rate
    
    while time.time() - t0 < time_budget:
        if T <= T_min:
            T = T_min
        
        # Choose move type
        r = np.random.random()
        if r < 0.3:
            # Single insert
            absent = np.where(mask == 0)[0]
            if len(absent) == 0:
                continue
            idx = absent[np.random.randint(len(absent))]
            if not _valid_insert(mask, idx):
                continue
            # Compute 3-AP gain
            gain = 0
            for c in range(idx + 2, _N, 2):
                b = (idx + c) // 2
                if mask[b]:
                    gain += 1
            for a in range(idx - 2, -1, -2):
                b = (a + idx) // 2
                if mask[b]:
                    gain += 1
            for d in range(1, _N):
                a = idx - d
                c = idx + d
                if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                    gain += 1
            delta = gain
            if delta >= 0 or np.random.random() < np.exp(delta / T):
                mask[idx] = 1
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_mask = mask.copy()
        elif r < 0.5:
            # Single remove
            present = np.where(mask == 1)[0]
            if len(present) == 0:
                continue
            idx = present[np.random.randint(len(present))]
            # Compute 3-AP loss
            loss = 0
            for c in range(idx + 2, _N, 2):
                b = (idx + c) // 2
                if mask[b]:
                    loss += 1
            for a in range(idx - 2, -1, -2):
                b = (a + idx) // 2
                if mask[b]:
                    loss += 1
            for d in range(1, _N):
                a = idx - d
                c = idx + d
                if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                    loss += 1
            delta = -loss
            if delta >= 0 or np.random.random() < np.exp(delta / T):
                mask[idx] = 0
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_mask = mask.copy()
        elif r < 0.8:
            # Swap
            present = np.where(mask == 1)[0]
            absent = np.where(mask == 0)[0]
            if len(present) == 0 or len(absent) == 0:
                continue
            out_idx = present[np.random.randint(len(present))]
            in_idx = absent[np.random.randint(len(absent))]
            if out_idx == in_idx:
                continue
            if not _swap_valid(mask, out_idx, in_idx):
                continue
            # Compute net 3-AP change
            # Remove out_idx contributions, add in_idx contributions (with out_idx gone)
            mask[out_idx] = 0
            # Gain from adding in_idx
            gain = 0
            for c in range(in_idx + 2, _N, 2):
                b = (in_idx + c) // 2
                if mask[b]:
                    gain += 1
            for a in range(in_idx - 2, -1, -2):
                b = (a + in_idx) // 2
                if mask[b]:
                    gain += 1
            for d in range(1, _N):
                a = in_idx - d
                c = in_idx + d
                if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                    gain += 1
            # Loss from removing out_idx (already removed, so compute what it contributed)
            # But some of its 3-APs may have involved in_idx... in_idx was 0 before, so no overlap
            loss = 0
            for c in range(out_idx + 2, _N, 2):
                b = (out_idx + c) // 2
                if mask[b]:
                    loss += 1
            for a in range(out_idx - 2, -1, -2):
                b = (a + out_idx) // 2
                if mask[b]:
                    loss += 1
            for d in range(1, _N):
                a = out_idx - d
                c = out_idx + d
                if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                    loss += 1
            mask[out_idx] = 1  # restore temporarily for delta calc
            delta = gain - loss
            if delta >= 0 or np.random.random() < np.exp(delta / T):
                mask[out_idx] = 0
                mask[in_idx] = 1
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_mask = mask.copy()
            else:
                pass  # mask already restored
        else:
            # Block move: remove 2, insert 2
            present = np.where(mask == 1)[0]
            absent = np.where(mask == 0)[0]
            if len(present) < 2 or len(absent) < 2:
                continue
            out1 = present[np.random.randint(len(present))]
            out2 = present[np.random.randint(len(present))]
            if out1 == out2:
                continue
            in1 = absent[np.random.randint(len(absent))]
            in2 = absent[np.random.randint(len(absent))]
            if in1 == in2:
                continue
            # Check validity: after removing out1,out2 and adding in1,in2
            mask[out1] = 0
            mask[out2] = 0
            valid = _valid_insert(mask, in1) and _valid_insert(mask, in2)
            if not valid:
                mask[out1] = 1
                mask[out2] = 1
                continue
            # Compute delta
            # Loss from out1, out2
            loss = 0
            for idx in (out1, out2):
                for c in range(idx + 2, _N, 2):
                    b = (idx + c) // 2
                    if mask[b]:
                        loss += 1
                for a in range(idx - 2, -1, -2):
                    b = (a + idx) // 2
                    if mask[b]:
                        loss += 1
                for d in range(1, _N):
                    a = idx - d
                    c = idx + d
                    if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                        loss += 1
            # Subtract any 3-APs between out1 and out2 (double counted)
            # Check if out1, mid, out2 form a 3-AP or vice versa
            diff = abs(out1 - out2)
            if diff % 2 == 0:
                mid = (out1 + out2) // 2
                if 0 <= mid < _N and mask[mid]:
                    loss -= 1  # this 3-AP was counted in both
            # Gain from in1, in2
            gain = 0
            for idx in (in1, in2):
                for c in range(idx + 2, _N, 2):
                    b = (idx + c) // 2
                    if mask[b]:
                        gain += 1
                for a in range(idx - 2, -1, -2):
                    b = (a + idx) // 2
                    if mask[b]:
                        gain += 1
                for d in range(1, _N):
                    a = idx - d
                    c = idx + d
                    if 0 <= a < _N and 0 <= c < _N and mask[a] and mask[c]:
                        gain += 1
            # Add back 3-APs between in1 and in2
            diff2 = abs(in1 - in2)
            if diff2 % 2 == 0:
                mid2 = (in1 + in2) // 2
                if 0 <= mid2 < _N and mask[mid2]:
                    gain += 1
            delta = gain - loss
            if delta >= 0 or np.random.random() < np.exp(delta / T):
                mask[in1] = 1
                mask[in2] = 1
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_mask = mask.copy()
            else:
                mask[out1] = 1
                mask[out2] = 1
        T *= alpha
    return best_mask, best_score

def construct():
    """Construct a 4-AP-free subset of [1,256] maximizing 3-AP count."""
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    # Start with greedy seed
    mask = _greedy_seed()
    
    # Local search polish
    mask, score = _local_search(mask, max_iter=2000, time_limit=3.0)
    
    if time.time() > deadline - 10:
        result = [i + 1 for i in range(_N) if mask[i]]
        return result
    
    # SA with multiple restarts
    best_mask = mask.copy()
    best_score = score
    
    while time.time() < deadline - 5:
        # Occasionally restart from greedy or perturbed best
        if np.random.random() < 0.1:
            # Perturb best: random remove/insert
            m = best_mask.copy()
            n_present = np.sum(m)
            if n_present > 5:
                for _ in range(min(5, n_present // 4)):
                    present = np.where(m == 1)[0]
                    idx = present[np.random.randint(len(present))]
                    m[idx] = 0
                    absent = np.where(m == 0)[0]
                    if len(absent) > 0:
                        new_idx = absent[np.random.randint(len(absent))]
                        # Check validity
                        valid = True
                        for qi in elem_quads[new_idx]:
                            q = _QUADS[qi]
                            for x in q:
                                if x != new_idx and not m[x]:
                                    break
                            else:
                                valid = False
                                break
                        if valid:
                            m[new_idx] = 1
            mask = m
        else:
            mask = best_mask.copy()
        
        # Run SA for a time slice
        time_left = deadline - 5 - time.time()
        if time_left <= 0:
            break
        sa_budget = min(8.0, time_left * 0.5)
        if sa_budget < 0.5:
            break
        sa_mask, sa_score = _sa_search(mask, sa_budget)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask.copy()
        
        # Quick local search polish on best
        if time.time() < deadline - 3:
            best_mask, best_score = _local_search(best_mask, max_iter=1000, time_limit=2.0)
    
    result = [i + 1 for i in range(_N) if best_mask[i]]
    if not result:
        result = [1]
    return result

# Precompute elem_quads at module level for use in construct
elem_quads = [[] for _ in range(_N)]
for qi, q in enumerate(_QUADS):
    for elem in q:
        elem_quads[elem].append(qi)
