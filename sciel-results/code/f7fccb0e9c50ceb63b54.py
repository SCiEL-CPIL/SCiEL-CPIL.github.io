import time
import random

N = 128
DEADLINE = 85.0  # safety margin under 90s

def _can_add(Sset, x):
    """Check if adding x to Sset creates any violation."""
    if x in Sset:
        return False
    for a in Sset:
        if (x + a) in Sset:
            for b in Sset:
                if b != a and (x + b) in Sset and (a + b) in Sset:
                    return False
    return True

def _check_valid(Sset):
    """Full O(n^3) validity check. Returns True if valid."""
    lst = sorted(Sset)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def _greedy_fill(Sset, order=None, rng=None):
    """Greedy fill: try adding elements in given order (or 1..N) if valid."""
    if order is None:
        order = list(range(1, N+1))
    for x in order:
        if x not in Sset and _can_add(Sset, x):
            Sset.add(x)
    return Sset

def _local_search(Sset, max_iters=2000, rng=None):
    """Local search: try removing one element and adding another to improve size."""
    if rng is None:
        rng = random.Random(42)
    Sset = set(Sset)
    best_size = len(Sset)
    all_vals = list(range(1, N+1))
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Try a random swap: remove y, add x
        notin = [x for x in all_vals if x not in Sset]
        if not notin:
            break
        
        x = rng.choice(notin)
        y = rng.choice(list(Sset))
        
        # Remove y
        S2 = Sset - {y}
        
        # Try to add x
        if _can_add(S2, x):
            S2.add(x)
            # Try to add back more elements
            improved = True
            while improved:
                improved = False
                for z in all_vals:
                    if z not in S2 and _can_add(S2, z):
                        S2.add(z)
                        improved = True
            
            if len(S2) > best_size:
                best_size = len(S2)
                Sset = S2
    
    return Sset

def _two_part_construct():
    """Two-part construction with optimized base and aggressive local search."""
    # Base: U = {65..128} (sum-free since min pair sum = 65+66=131 > 128)
    # L = odd numbers in 1..32 (sum-free, and 2*max(L)=64 < 65=min(U))
    # This gives 16 + 64 = 80 elements
    
    L = set(x for x in range(1, 33) if x % 2 == 1)
    U = set(range(65, 129))
    Sset = L | U
    
    if not _check_valid(Sset):
        Sset = set(range(65, 129))
    
    # Aggressive local search with more iterations
    Sset = _local_search(Sset, max_iters=5000)
    
    return Sset

def _random_restart_construct():
    """Random restart with greedy + local search, plus perturbation from base."""
    best_set = set()
    best_size = 0
    rng = random.Random(7)
    
    # Start from the two-part base and perturb
    base_L = set(x for x in range(1, 33) if x % 2 == 1)
    base_U = set(range(65, 129))
    base = base_L | base_U
    
    for trial in range(30):
        if time.time() > DEADLINE - 10:
            break
        
        if trial < 10:
            # Perturb the base: randomly remove some elements and try to fill gaps
            Sset = set(base)
            # Remove 3-8 random elements
            to_remove = rng.sample(sorted(Sset), rng.randint(3, 8))
            Sset -= set(to_remove)
            # Greedy fill
            for x in range(1, N+1):
                if x not in Sset and _can_add(Sset, x):
                    Sset.add(x)
        else:
            # Pure greedy with random order
            candidates = list(range(1, N+1))
            rng.shuffle(candidates)
            Sset = set()
            for x in candidates:
                if _can_add(Sset, x):
                    Sset.add(x)
        
        Sset = _local_search(Sset, max_iters=800, rng=rng)
        
        if len(Sset) > best_size:
            best_size = len(Sset)
            best_set = set(Sset)
    
    return best_set

def _simulated_annealing(Sset, max_iters=3000, rng=None):
    """Simulated annealing: accept worse moves with probability to escape local optima."""
    if rng is None:
        rng = random.Random(123)
    Sset = set(Sset)
    best_set = set(Sset)
    best_size = len(Sset)
    all_vals = list(range(1, N+1))
    current_temp = 1.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Try a random swap: remove y, add x
        notin = [x for x in all_vals if x not in Sset]
        if not notin:
            break
        
        x = rng.choice(notin)
        y = rng.choice(list(Sset))
        
        # Remove y
        S2 = Sset - {y}
        
        # Try to add x
        if _can_add(S2, x):
            S2.add(x)
            # Try to add back more elements
            improved = True
            while improved:
                improved = False
                for z in all_vals:
                    if z not in S2 and _can_add(S2, z):
                        S2.add(z)
                        improved = True
            
            new_size = len(S2)
            if new_size > len(Sset):
                Sset = S2
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(Sset)
            else:
                # Accept with probability exp(-delta/temp)
                delta = len(Sset) - new_size
                if delta == 0:
                    Sset = S2
                else:
                    if rng.random() < (1.0 / (1.0 + delta * current_temp)):
                        Sset = S2
        
        current_temp *= cooling
    
    return best_set

def construct():
    start = time.time()
    
    # Start with the two-part construction (known to give ~80)
    best_set = _two_part_construct()
    best_size = len(best_set)
    
    # Try random restarts to see if we can do better
    if time.time() < DEADLINE - 15:
        rs_set = _random_restart_construct()
        if len(rs_set) > best_size:
            best_set = rs_set
            best_size = len(rs_set)
    
    # Simulated annealing on the best set
    if time.time() < DEADLINE - 10:
        sa_set = _simulated_annealing(best_set, max_iters=2000)
        if len(sa_set) > best_size:
            best_set = sa_set
            best_size = len(sa_set)
    
    # Final local search on the best set with more iterations
    if time.time() < DEADLINE - 5:
        best_set = _local_search(best_set, max_iters=5000)
    
    # Verify validity
    if not _check_valid(best_set):
        # Fallback to safe set
        best_set = set(range(65, 129))
    
    res = sorted(best_set)
    return [int(x) for x in res]
