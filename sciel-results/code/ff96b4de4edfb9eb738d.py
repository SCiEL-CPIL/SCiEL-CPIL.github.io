"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 12
M = 40
FULL = (1 << N) - 1

def _cmp(mi, mj):
    """Return 1 if mi strict subset of mj, -1 if mj strict subset of mi, 0 otherwise."""
    inter = mi & mj
    if inter == mi and mi != mj:
        return 1
    if inter == mj and mi != mj:
        return -1
    return 0

def _count_comparable(masks):
    cnt = 0
    m = len(masks)
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            if _cmp(mi, masks[j]) != 0:
                cnt += 1
    return cnt

def _delta(masks, idx, new_mask):
    """Change in comparable count when replacing masks[idx] with new_mask."""
    old = masks[idx]
    if old == new_mask:
        return 0
    m = len(masks)
    delta = 0
    for j in range(m):
        if j == idx:
            continue
        mj = masks[j]
        # Remove old contribution
        c_old = _cmp(old, mj)
        if c_old != 0:
            delta -= 1
        # Add new contribution
        c_new = _cmp(new_mask, mj)
        if c_new != 0:
            delta += 1
    return delta

def _all_masks_sorted_by_popcount():
    """Return list of all 2^12 masks, sorted by popcount then value."""
    masks = list(range(1 << N))
    masks.sort(key=lambda m: (bin(m).count('1'), m))
    return masks

def _init_greedy_by_layer():
    """
    Initialize by picking masks to maximize chain structure.
    Strategy: build chains of nested sets.
    With 12 elements, the longest chain has 13 sets (sizes 0 through 12).
    We want 40 sets. Let's think about how to structure this.
    
    A key insight: if we pick sets that form chains, we get many comparable pairs.
    For a chain of length L, we get L*(L-1)/2 comparable pairs.
    
    To maximize total comparable pairs with 40 distinct sets, we should
    create as many long chains as possible, but chains can overlap (share sets).
    
    Actually, the best structure is likely to pick sets that are mostly
    nested. Let's try: pick all sets of sizes in a small range, or
    pick sets that form a few long chains.
    
    Alternative: pick 40 sets that are all subsets of some large set,
    arranged in a chain-like structure.
    
    Let's try a different approach: pick sets that are all comparable
    to each other (a single chain of 40). But max chain length is 13.
    
    So we need multiple chains. Let's say we pick k chains of length L each.
    Total sets = k * L (if disjoint). Comparable pairs = k * L*(L-1)/2.
    
    With 40 sets and max chain 13: 3 chains of 13 = 39 sets, plus 1 more.
    Comparable pairs = 3 * 13*12/2 + 0 = 234. But the 40th set adds some.
    
    But chains can share sets! If two chains share a set, we get more comparability.
    
    Actually, the theoretical maximum for 40 subsets of a 12-set:
    The total number of comparable pairs in the entire Boolean lattice B_12
    is sum over all pairs. But we only pick 40.
    
    Let me just use a greedy approach: start with a few chains, then add sets
    that maximize comparability with existing sets.
    """
    # First, create a backbone chain: empty set, {0}, {0,1}, ..., {0,...,11}
    # That's 13 sets forming a chain.
    backbone = []
    cur = 0
    for i in range(N + 1):
        backbone.append(cur)
        if i < N:
            cur |= (1 << i)
    
    # Now add more sets greedily to maximize comparable pairs
    used = set(backbone)
    chosen = list(backbone)
    
    # Precompute popcount
    all_masks = list(range(1 << N))
    rng = random.Random(42)
    rng.shuffle(all_masks)
    
    while len(chosen) < M:
        best_mask = None
        best_score = -1
        for mask in all_masks:
            if mask in used:
                continue
            # Score: number of comparable pairs with chosen
            s = 0
            for cm in chosen:
                if _cmp(mask, cm) != 0:
                    s += 1
            if s > best_score:
                best_score = s
                best_mask = mask
        if best_mask is None:
            break
        chosen.append(best_mask)
        used.add(best_mask)
    
    return chosen[:M]

def _local_search(masks, deadline):
    """Local search with multiple move types."""
    rng = random.Random(123)
    m = len(masks)
    used_set = set(masks)
    current_score = _count_comparable(masks)
    best_score = current_score
    best_masks = list(masks)
    
    # Precompute all masks sorted by popcount for structured moves
    all_by_pop = [[] for _ in range(N + 1)]
    for mask in range(1 << N):
        pc = bin(mask).count('1')
        all_by_pop[pc].append(mask)
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        no_improve += 1
        improved = False
        
        # Try each position
        for idx in range(m):
            if time.time() >= deadline:
                break
            old = masks[idx]
            
            # Move type 1: try all single-bit additions/removals
            candidates = set()
            # Add one bit
            for b in range(N):
                nm = old | (1 << b)
                if nm not in used_set:
                    candidates.add(nm)
            # Remove one bit
            for b in range(N):
                if (old >> b) & 1:
                    nm = old & ~(1 << b)
                    if nm not in used_set:
                        candidates.add(nm)
            
            # Move type 2: try masks of nearby popcount
            pc_old = bin(old).count('1')
            for dp in [-2, -1, 1, 2]:
                pc_new = pc_old + dp
                if 0 <= pc_new <= N:
                    for nm in all_by_pop[pc_new]:
                        if nm not in used_set and nm != old:
                            candidates.add(nm)
                            if len(candidates) > 200:
                                break
                if len(candidates) > 200:
                    break
            
            # Move type 3: random candidates
            for _ in range(30):
                nm = rng.randint(0, (1 << N) - 1)
                if nm not in used_set and nm != old:
                    candidates.add(nm)
            
            # Find best candidate
            best_d = 0
            best_nm = None
            for nm in candidates:
                d = _delta(masks, idx, nm)
                if d > best_d:
                    best_d = d
                    best_nm = nm
            
            if best_nm is not None:
                used_set.discard(old)
                masks[idx] = best_nm
                used_set.add(best_nm)
                current_score += best_d
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
                improved = True
                no_improve = 0
        
        if not improved:
            # Try random restarts within local search: perturb
            if no_improve < max_no_improve:
                for _ in range(5):
                    idx = rng.randrange(m)
                    old = masks[idx]
                    nm = rng.randint(0, (1 << N) - 1)
                    if nm not in used_set and nm != old:
                        d = _delta(masks, idx, nm)
                        if d >= 0:  # Allow neutral or improving moves
                            used_set.discard(old)
                            masks[idx] = nm
                            used_set.add(nm)
                            current_score += d
                            improved = True
                            break
    
    return best_masks, best_score

def _refine_with_chains(masks, deadline):
    """
    Try to reorganize masks into chains to maximize comparability.
    """
    rng = random.Random(777)
    m = len(masks)
    used_set = set(masks)
    current_score = _count_comparable(masks)
    best_score = current_score
    best_masks = list(masks)
    
    # Try to identify and strengthen chains
    # For each set, find its "best neighbor" (a set it's comparable with)
    # and try to build longer chains
    
    while time.time() < deadline:
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old = masks[idx]
            pc = bin(old).count('1')
            
            # Try to replace with a set that has more comparables
            best_d = 0
            best_nm = None
            
            # Try sets that are subsets or supersets of many existing sets
            # Focus on sets that can be in the middle of chains
            for target_pc in range(max(0, pc - 3), min(N + 1, pc + 4)):
                for nm in range(1 << N):
                    if bin(nm).count('1') != target_pc:
                        continue
                    if nm in used_set or nm == old:
                        continue
                    d = _delta(masks, idx, nm)
                    if d > best_d:
                        best_d = d
                        best_nm = nm
                    if best_d >= 3:  # Good enough, don't over-search
                        break
                if best_d >= 3:
                    break
            
            if best_nm is not None:
                used_set.discard(old)
                masks[idx] = best_nm
                used_set.add(best_nm)
                current_score += best_d
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
                improved = True
        
        if not improved:
            break
    
    return best_masks, best_score

def construct():
    start = time.time()
    deadline = start + 80  # Leave 10s safety margin for 90s limit
    
    # Phase 1: Greedy initialization
    masks = _init_greedy_by_layer()
    score = _count_comparable(masks)
    
    # Phase 2: Local search
    masks, score = _local_search(masks, min(deadline, start + 60))
    
    # Phase 3: Further refinement
    if time.time() < deadline - 5:
        masks, score2 = _refine_with_chains(masks, deadline - 5)
        if score2 > score:
            score = score2
    
    # Phase 4: Final quick local search
    if time.time() < deadline - 2:
        masks, score3 = _local_search(masks, deadline - 2)
        if score3 > score:
            score = score3
    
    # Convert to list of lists of Python ints
    result = []
    for mask in masks:
        s = []
        for b in range(N):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    # Verify: exactly 40 distinct lists of distinct ints
    assert len(result) == 40
    seen = set()
    for s in result:
        fs = frozenset(s)
        assert fs not in seen, "Duplicate set found"
        seen.add(fs)
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool)
            assert 0 <= x <= 11
    
    return result
# EVOLVE-BLOCK-END
