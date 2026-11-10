# EVOLVE-BLOCK-START
import numpy as np
import time
import random

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        val = pow(x, 2, M)
        if val != 0:
            forbidden.add(val)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Precompute conflict masks
# conflict_mask[i] is a bitset of all j such that i and j cannot coexist
CONFLICT_MASKS = [0] * M
for i in range(M):
    mask = 0
    for j in range(M):
        if i != j and (i - j) % M in FORBIDDEN:
            mask |= (1 << j)
    CONFLICT_MASKS[i] = mask

def _popcount(x):
    return bin(x).count('1')

def _can_add(cand, sel_bits):
    return (sel_bits & CONFLICT_MASKS[cand]) == 0

def _greedy_init():
    sel = 0
    remaining = list(range(M))
    random.shuffle(remaining)
    for v in remaining:
        if _can_add(v, sel):
            sel |= (1 << v)
    return sel

def _crt_init():
    """
    Use CRT structure: 205 = 5 * 41.
    A number n mod 205 corresponds to (n mod 5, n mod 41).
    The forbidden differences are exactly those (d5, d41) where d5 is a QR mod 5 
    AND d41 is a QR mod 41 (both non-zero), OR d5=0 and d41 is a non-zero QR mod 41, 
    OR d41=0 and d5 is a non-zero QR mod 5.
    
    Actually, let's be more careful. x^2 mod 205 = (x^2 mod 5, x^2 mod 41) by CRT.
    So the forbidden set F mod 205 is the set of all (a, b) mod (5, 41) where a = x^2 mod 5
    for some x, and b = x^2 mod 41 for the same x.
    
    QR mod 5: {0, 1, 4} (from x=0,1,2,3,4)
    QR mod 41: {0, 1, 4, 5, 9, 16, 20, 25, 36, 39} (10 values)
    
    But not all combinations (a, b) with a in QR_5 and b in QR_41 are achievable as x^2 mod 205.
    We need x such that x^2 ≡ a (mod 5) AND x^2 ≡ b (mod 41).
    
    For the construction: if we pick a set S_5 of residues mod 5 and S_41 of residues mod 41
    such that no two elements in S_5 differ by a QR mod 5, and no two in S_41 differ by a QR mod 41,
    then the Cartesian product S_5 × S_41 lifted via CRT gives a valid set mod 205.
    
    Wait, that's not quite right either. Let me think again.
    
    If a - b mod 205 is in F, then there exists x such that x^2 ≡ a - b (mod 205).
    This means x^2 ≡ a - b (mod 5) AND x^2 ≡ a - b (mod 41).
    So (a-b) mod 5 must be a QR mod 5, AND (a-b) mod 41 must be a QR mod 41.
    
    So the condition is: for any two elements a, b in our set, (a-b) mod 5 must NOT be a non-zero QR mod 5,
    OR (a-b) mod 41 must NOT be a non-zero QR mod 41.
    
    Actually no. The forbidden condition is: (a-b) mod 205 is in F.
    (a-b) mod 205 is in F iff there exists x with 1<=x<=204 such that x^2 ≡ a-b (mod 205).
    By CRT, this iff x^2 ≡ a-b (mod 5) AND x^2 ≡ a-b (mod 41) for some x.
    This is equivalent to: (a-b) mod 5 is a QR mod 5 (possibly 0) AND (a-b) mod 41 is a QR mod 41 (possibly 0).
    
    But we removed 0 from F. So (a-b) mod 205 = 0 is allowed (but a≠b so this can't happen for distinct elements).
    For a ≠ b, (a-b) mod 205 ≠ 0.
    
    So the forbidden condition for a≠b is: (a-b) mod 5 is in QR_5\{0} = {1,4} AND (a-b) mod 41 is in QR_41\{0}.
    OR: (a-b) mod 5 = 0 AND (a-b) mod 41 is in QR_41\{0}.
    OR: (a-b) mod 5 is in QR_5\{0} AND (a-b) mod 41 = 0.
    
    Wait, I need to be more careful. x^2 mod 205 = 0 iff x ≡ 0 (mod 5) AND x ≡ 0 (mod 41), i.e., x ≡ 0 (mod 205).
    Since x is in 1..204, x^2 mod 205 ≠ 0. So 0 is not in F (which is consistent with removing it).
    
    For x in 1..204, x^2 mod 205 can have:
    - x^2 mod 5 = 0 (if 5|x) and x^2 mod 41 ≠ 0 (if 41∤x): then the pair is (0, b) where b is a non-zero QR mod 41
    - x^2 mod 5 ≠ 0 and x^2 mod 41 = 0 (if 41|x and 5∤x): then the pair is (a, 0) where a is a non-zero QR mod 5
    - x^2 mod 5 ≠ 0 and x^2 mod 41 ≠ 0: then the pair is (a, b) where a is a non-zero QR mod 5 and b is a non-zero QR mod 41
    
    So F mod 205 corresponds to the set of pairs:
    {(0, b) : b in QR_41\{0}} ∪ {(a, 0) : a in QR_5\{0}} ∪ {(a, b) : a in QR_5\{0}, b in QR_41\{0}}
    
    This is exactly: {d : d mod 5 in QR_5 AND d mod 41 in QR_41} \ {(0,0)}
    = {d : d mod 5 in QR_5 AND d mod 41 in QR_41} (since d≠0 mod 205 for our purposes)
    
    Wait, QR_5 = {0, 1, 4} and QR_41 = {0, 1, 4, 5, 9, 16, 20, 25, 36, 39}.
    So the condition for d being in F is: d mod 5 in {0,1,4} AND d mod 41 in QR_41, AND d ≠ 0 mod 205.
    
    For our set S, we need: for all a≠b in S, NOT (a-b mod 5 in {0,1,4} AND a-b mod 41 in QR_41).
    Equivalently: for all a≠b in S, (a-b mod 5 not in {0,1,4}) OR (a-b mod 41 not in QR_41).
    
    Note: a-b mod 5 not in {0,1,4} means a-b mod 5 in {2,3}.
    
    So a sufficient condition is: for all a≠b in S, a-b mod 5 in {2,3}.
    This means S mod 5 has the property that all differences are in {2,3}.
    What's the max size of such a set mod 5? 
    If we have two elements x,y with x-y ≡ 2 or 3 (mod 5), then y-x ≡ 3 or 2 (mod 5). Good.
    Can we have 3 elements? x,y,z. x-y in {2,3}, y-z in {2,3}, x-z in {2,3}.
    x-y=2, y-z=2 => x-z=4. Not in {2,3}. 
    x-y=2, y-z=3 => x-z=0. Not in {2,3}.
    x-y=3, y-z=2 => x-z=0. Not in {2,3}.
    x-y=3, y-z=3 => x-z=1. Not in {2,3}.
    So max size mod 5 with all pairwise differences in {2,3} is 2.
    
    Similarly for mod 41: max size with all pairwise differences in non-QR mod 41.
    QR_41 = {0,1,4,5,9,16,20,25,36,39}. Non-QR (non-zero) = {2,3,6,7,8,10,11,12,13,14,15,17,18,19,21,22,23,24,26,27,28,29,30,31,32,33,34,35,37,38}.
    That's 30 non-QR values out of 40 non-zero values. So the QR graph mod 41 is sparse.
    The independence number of the QR graph mod 41 is likely around 20-25.
    
    But the sufficient condition "all differences in {2,3} mod 5" is too restrictive (gives only 2 per mod 5 class).
    
    A better approach: use the actual forbidden set and do greedy/local search on the full 205-vertex graph,
    but use CRT structure for better initialization.
    
    Let me just use the CRT to generate a good initial set:
    - Pick a large independent set in the QR graph mod 41 (call it T_41)
    - For each element t in T_41, we can potentially include multiple residues mod 5
    - But the interactions are complex.
    
    Let me just try a different approach: use the CRT to enumerate all 205 elements as pairs (a mod 5, b mod 41),
    and do a greedy construction that considers both components.
    """
    # QR sets
    qr5 = {0, 1, 4}
    qr41 = set()
    for x in range(1, 41):
        qr41.add(pow(x, 2, 41))
    qr41.add(0)
    
    # Build mapping from value mod 205 to (mod 5, mod 41)
    # Using CRT: n = a*41*(41^{-1} mod 5) + b*5*(5^{-1} mod 41) mod 205
    # 41 mod 5 = 1, so 41^{-1} mod 5 = 1
    # 5^{-1} mod 41: 5*33 = 165 = 4*41 + 1, so 5^{-1} = 33 mod 41
    # n = a*41 + b*5*33 mod 205 = a*41 + b*165 mod 205
    
    def crt(a, b):
        return (a * 41 + b * 165) % 205
    
    # For each pair (a, b) with a in 0..4, b in 0..40, compute n = crt(a,b)
    # Then check if n is valid to add to current set
    
    # Greedy: iterate through all 205 elements in a smart order
    # Sort by: first by b (mod 41 component), then by a (mod 5 component)
    # This way we group elements with similar mod 41 values
    
    elements = []
    for a in range(5):
        for b in range(41):
            n = crt(a, b)
            elements.append((n, a, b))
    
    # Try multiple random orderings
    best_sel = 0
    best_cnt = 0
    
    for trial in range(50):
        random.shuffle(elements)
        sel = 0
        cnt = 0
        for n, a, b in elements:
            if _can_add(n, sel):
                sel |= (1 << n)
                cnt += 1
        if cnt > best_cnt:
            best_cnt = cnt
            best_sel = sel
    
    return best_sel

def _local_search(bits, deadline):
    """
    Perform local search to maximize the size of the independent set.
    Uses 1-swaps and 2-swaps.
    """
    current = bits
    current_size = _popcount(current)
    best = current
    best_size = current_size
    
    # List of vertices for random selection
    vertices = list(range(M))
    
    # We will try to improve by:
    # 1. Adding a vertex if possible (Size +1)
    # 2. Swapping 1 vertex (Size 0) - helps rearrange for future additions
    # 3. Swapping 2 vertices (Size 0) - helps escape local minima
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add a vertex
        # We can iterate through all vertices not in current
        # To make this faster, we can just pick random ones, but a full scan is cheap (205 items)
        for v in vertices:
            if not (current >> v) & 1:
                if _can_add(v, current):
                    current |= (1 << v)
                    current_size += 1
                    best = current
                    best_size = current_size
                    improved = True
                    break # Restart scan after addition
        
        if improved:
            continue
            
        # Phase 2: Try 1-swaps
        # Pick a random u in S and v not in S
        # If we find a valid swap, we accept it (neutral move)
        # To avoid cycling, we can use a Tabu list or just random restarts
        # For now, we accept any valid swap.
        
        # Efficient swap finding:
        # We can iterate through u in S and check all v not in S.
        # But that's O(N^2). 205^2 = 42k. This is very fast.
        # Let's do a full scan for 1-swaps to find ANY valid swap.
        found_swap = False
        # Get list of vertices in S
        s_vertices = [i for i in range(M) if (current >> i) & 1]
        # Get list of vertices not in S
        not_s_vertices = [i for i in range(M) if not (current >> i) & 1]
        
        # Randomize order to avoid bias
        random.shuffle(s_vertices)
        random.shuffle(not_s_vertices)
        
        for u in s_vertices:
            temp_state = current & ~(1 << u)
            # Check if any v in not_s_vertices can be added to temp_state
            # We can check all v
            for v in not_s_vertices:
                if (temp_state & CONFLICT_MASKS[v]) == 0:
                    # Valid swap
                    current = temp_state | (1 << v)
                    # Size remains same
                    found_swap = True
                    break
            if found_swap:
                break
        
        if found_swap:
            continue
            
        # Phase 3: Try 2-swaps
        # This is more expensive. We only do it if stuck.
        # Pick two random u1, u2 in S and two random v1, v2 not in S
        # Check if {v1, v2} is independent and compatible with S \ {u1, u2}
        
        # To avoid O(N^4), we sample.
        # Sample 50 random 2-swap attempts
        for _ in range(50):
            if len(s_vertices) < 2 or len(not_s_vertices) < 2:
                break
            u1, u2 = random.sample(s_vertices, 2)
            v1, v2 = random.sample(not_s_vertices, 2)
            
            # Check if v1 and v2 are compatible with each other
            if (1 << v1) & CONFLICT_MASKS[v2]:
                continue
                
            temp_state = current & ~(1 << u1) & ~(1 << u2)
            
            # Check if v1 and v2 are compatible with temp_state
            if (temp_state & CONFLICT_MASKS[v1]) == 0 and (temp_state & CONFLICT_MASKS[v2]) == 0:
                current = temp_state | (1 << v1) | (1 << v2)
                # Size remains same
                found_swap = True
                break
        
        if found_swap:
            continue
            
        # If no improvement found (no add, no 1-swap, no 2-swap)
        # We are in a local optimum.
        # Perturbation: Remove 3 random elements and add 3 random compatible ones?
        # Or just break and let ILS handle restart.
        # For this function, we just break to signal ILS to perturb.
        break
        
    return current, best, best_size

def furstenberg_sarkozy_205() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0  # Safety margin
    
    random.seed(42)
    
    best_global = 0
    best_global_size = 0
    
    # Phase 1: Multiple initialization strategies
    # Strategy A: Random greedy
    for i in range(50):
        if time.time() > deadline - 100:
            break
        sel = _greedy_init()
        cnt = _popcount(sel)
        if cnt > best_global_size:
            best_global = sel
            best_global_size = cnt
    
    # Strategy B: CRT-based greedy
    if time.time() < deadline - 100:
        sel_crt = _crt_init()
        cnt_crt = _popcount(sel_crt)
        if cnt_crt > best_global_size:
            best_global = sel_crt
            best_global_size = cnt_crt
    
    # Phase 2: Iterated Local Search
    # We will run local search, then perturb, then local search again.
    current_best = best_global
    current_size = best_global_size
    
    while time.time() < deadline - 10:
        # Local search on current_best
        improved_bits, best_bits, best_size = _local_search(current_best, deadline - 10)
        
        if best_size > best_global_size:
            best_global = best_bits
            best_global_size = best_size
            current_best = best_bits
            current_size = best_size
            
        # If we improved, we might stay here a bit more, but ILS usually perturbs immediately
        # Perturb current_best
        if time.time() > deadline - 10:
            break
            
        # Perturb current_best
        # Remove 5 random elements
        vertices_in = [i for i in range(M) if (current_best >> i) & 1]
        if len(vertices_in) < 5:
            # If set is small, just random restart
            current_best = _greedy_init()
        else:
            to_remove = random.sample(vertices_in, 5)
            for v in to_remove:
                current_best &= ~(1 << v)
            current_size -= 5
            
            # Add 5 random compatible elements
            # We try to add them one by one
            vertices_out = [i for i in range(M) if not (current_best >> i) & 1]
            random.shuffle(vertices_out)
            added = 0
            for v in vertices_out:
                if added >= 5:
                    break
                if _can_add(v, current_best):
                    current_best |= (1 << v)
                    current_size += 1
                    added += 1
                    
    result = [i for i in range(M) if (best_global >> i) & 1]
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
