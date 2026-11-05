# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N_MAX = 512
DEADLINE = 85.0  # 90s kill, 5s margin

def _can_add_bool(s_bool, x, max_d1, max_d2, max_d3, max_d4):
    """Check if element x (1-indexed) can be added to a 4-AP-free set.
    s_bool is a boolean array where s_bool[i] means element i+1 is in the set.
    Precomputed max_d values bound the search for each of the 4 roles x can play.
    """
    # Role 4: x is the last term: x-3d, x-2d, x-d all in set
    for d in range(1, max_d1 + 1):
        if s_bool[x-3*d-1] and s_bool[x-2*d-1] and s_bool[x-d-1]:
            return False
    # Role 3: x is 3rd term: x-2d, x-d, x+d all in set
    for d in range(1, max_d2 + 1):
        if s_bool[x-2*d-1] and s_bool[x-d-1] and s_bool[x+d-1]:
            return False
    # Role 2: x is 2nd term: x-d, x+d, x+2d all in set
    for d in range(1, max_d3 + 1):
        if s_bool[x-d-1] and s_bool[x+d-1] and s_bool[x+2*d-1]:
            return False
    # Role 1: x is first term: x+d, x+2d, x+3d all in set
    for d in range(1, max_d4 + 1):
        if s_bool[x+d-1] and s_bool[x+2*d-1] and s_bool[x+3*d-1]:
            return False
    return True

def _max_ds(x):
    """Precompute the max d for each of the 4 roles for element x."""
    max_d1 = (x - 1) // 3
    max_d2 = (x - 1) // 2
    max_d3 = min(x - 1, (N_MAX - x) // 2)
    max_d4 = (N_MAX - x) // 3
    return max_d1, max_d2, max_d3, max_d4

def _has_4ap_bool(s_bool):
    """Full O(n^2) verification. Returns True if a non-constant 4-AP exists."""
    n = N_MAX
    for a in range(1, n + 1):
        if not s_bool[a-1]:
            continue
        for b in range(a+1, n+1):
            if not s_bool[b-1]:
                continue
            d = b - a
            c = b + d
            e = c + d
            if c <= n and e <= n and s_bool[c-1] and s_bool[e-1]:
                return True
    return False

def _base3_construction():
    """Return a set (as bool array) of all numbers in [1,512] whose base-3 digits are 0 or 1 only.
    This is provably 4-AP-free: if a,b,c,d form a 4-AP, then in base-3 addition with no carries
    each digit position must satisfy the digit-AP property, and the only 4-digit APs over {0,1}
    are trivial (constant). Since all digits are in {0,1}, no carries can occur in a,b,c,d,
    so the digit-AP must be constant, contradicting non-constant 4-AP."""
    s_bool = np.zeros(N_MAX, dtype=bool)
    for x in range(1, N_MAX + 1):
        n = x
        ok = True
        while n > 0:
            if n % 3 == 2:
                ok = False
                break
            n //= 3
        if ok:
            s_bool[x-1] = True
    return s_bool

def _greedy_fill(s_bool, deadline):
    """Greedily add elements in order 1..512 to fill the set. Returns modified s_bool."""
    for x in range(1, N_MAX + 1):
        if s_bool[x-1]:
            continue
        if time.time() > deadline:
            break
        md1, md2, md3, md4 = _max_ds(x)
        if _can_add_bool(s_bool, x, md1, md2, md3, md4):
            s_bool[x-1] = True
    return s_bool

def _greedy_random(s_bool, deadline, rng):
    """Greedily add elements in random order. Returns modified s_bool."""
    candidates = [x for x in range(1, N_MAX+1) if not s_bool[x-1]]
    rng.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        md1, md2, md3, md4 = _max_ds(x)
        if _can_add_bool(s_bool, x, md1, md2, md3, md4):
            s_bool[x-1] = True
    return s_bool

def _local_search(s_bool, deadline, rng):
    """Local search: try to add missing elements; if stuck, try swaps (remove one, add one);
    then try 2-for-1 (remove two, add one) and 1-for-2 (remove one, add two)."""
    count = int(s_bool.sum())
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Try adding a random missing element
        missing = [x for x in range(1, N_MAX+1) if not s_bool[x-1]]
        if missing:
            x = rng.choice(missing)
            md1, md2, md3, md4 = _max_ds(x)
            if _can_add_bool(s_bool, x, md1, md2, md3, md4):
                s_bool[x-1] = True
                count += 1
                no_improve = 0
                continue
        
        no_improve += 1
        
        # Try a swap: remove a random present element, try to add a random missing element
        present = [x for x in range(1, N_MAX+1) if s_bool[x-1]]
        if present and missing:
            x_rm = rng.choice(present)
            x_add = rng.choice(missing)
            # Temporarily remove x_rm
            s_bool[x_rm-1] = False
            count -= 1
            md1, md2, md3, md4 = _max_ds(x_add)
            if _can_add_bool(s_bool, x_add, md1, md2, md3, md4):
                s_bool[x_add-1] = True
                count += 1
                no_improve = 0
            else:
                # Restore
                s_bool[x_rm-1] = True
                count += 1
        
        # Occasionally try 2-for-1: remove two, add one
        if no_improve > 100 and len(present) >= 2 and missing:
            if rng.random() < 0.3:
                x_rm1, x_rm2 = rng.sample(present, 2)
                x_add = rng.choice(missing)
                s_bool[x_rm1-1] = False
                s_bool[x_rm2-1] = False
                count -= 2
                md1, md2, md3, md4 = _max_ds(x_add)
                if _can_add_bool(s_bool, x_add, md1, md2, md3, md4):
                    s_bool[x_add-1] = True
                    count += 1
                    no_improve = 0
                else:
                    s_bool[x_rm1-1] = True
                    s_bool[x_rm2-1] = True
                    count += 2
        
        # Occasionally try 1-for-2: remove one, add two
        if no_improve > 200 and present and len(missing) >= 2:
            if rng.random() < 0.3:
                x_rm = rng.choice(present)
                x_add1, x_add2 = rng.sample(missing, 2)
                s_bool[x_rm-1] = False
                count -= 1
                ok1 = False
                ok2 = False
                md1a, md2a, md3a, md4a = _max_ds(x_add1)
                if _can_add_bool(s_bool, x_add1, md1a, md2a, md3a, md4a):
                    s_bool[x_add1-1] = True
                    ok1 = True
                md1b, md2b, md3b, md4b = _max_ds(x_add2)
                if _can_add_bool(s_bool, x_add2, md1b, md2b, md3b, md4b):
                    s_bool[x_add2-1] = True
                    ok2 = True
                if ok1 and ok2:
                    count += 2
                    no_improve = 0
                else:
                    # Restore: undo additions and re-add removed
                    if ok1:
                        s_bool[x_add1-1] = False
                    if ok2:
                        s_bool[x_add2-1] = False
                    s_bool[x_rm-1] = True
                    count += 1
    
    return s_bool

def construct():
    """
    Construct a large subset of [1, 512] with no non-constant 4-term AP.
    Strategy:
    1. Start from the base-3 {0,1}-digit set (provably 4-AP-free, ~64 elements).
    2. Greedy fill in order and in random orders to get a good base.
    3. Local search with add, swap, 2-for-1, and 1-for-2 moves.
    4. Multiple restarts to find the best set.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_s_bool = None
    best_count = 0
    
    # Phase 1: Build a strong base via greedy on base-3 set
    base3 = _base3_construction()
    
    # Try greedy fill in natural order
    s1 = base3.copy()
    s1 = _greedy_fill(s1, deadline)
    if int(s1.sum()) > best_count:
        best_count = int(s1.sum())
        best_s_bool = s1.copy()
    
    # Try several random greedy fills
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        s_trial = base3.copy()
        s_trial = _greedy_random(s_trial, deadline - 10, rng)
        cnt = int(s_trial.sum())
        if cnt > best_count:
            best_count = cnt
            best_s_bool = s_trial.copy()
    
    # Also try pure greedy from scratch (no base-3 seed)
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        s_pure = np.zeros(N_MAX, dtype=bool)
        s_pure = _greedy_random(s_pure, deadline - 10, rng)
        cnt = int(s_pure.sum())
        if cnt > best_count:
            best_count = cnt
            best_s_bool = s_pure.copy()
    
    # Phase 2: Local search on the best base found
    if best_s_bool is None:
        best_s_bool = base3
        best_count = int(best_s_bool.sum())
    
    s_current = best_s_bool.copy()
    s_current = _local_search(s_current, deadline - 5, rng)
    cnt = int(s_current.sum())
    if cnt > best_count:
        best_count = cnt
        best_s_bool = s_current.copy()
    
    # Phase 3: More restarts with local search
    for restart in range(8):
        if time.time() > deadline - 5:
            break
        # Start from a random greedy base
        s_r = np.zeros(N_MAX, dtype=bool)
        # Seed with base-3 or not
        if rng.random() < 0.5:
            s_r = base3.copy()
        s_r = _greedy_random(s_r, deadline - 5, rng)
        s_r = _local_search(s_r, deadline - 5, rng)
        cnt = int(s_r.sum())
        if cnt > best_count:
            best_count = cnt
            best_s_bool = s_r.copy()
    
    # Phase 4: Intensive local search on the best set found
    if time.time() < deadline - 2:
        s_best = best_s_bool.copy()
        s_best = _local_search(s_best, deadline - 2, rng)
        cnt = int(s_best.sum())
        if cnt > best_count:
            best_count = cnt
            best_s_bool = s_best.copy()
    
    # Final verification
    if _has_4ap_bool(best_s_bool):
        # Fallback: base-3 set is guaranteed valid
        best_s_bool = base3
        best_count = int(best_s_bool.sum())
    
    result = [i + 1 for i in range(N_MAX) if best_s_bool[i]]
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
