import time
import random
import math
import numpy as np

BASE = 51975

# Precompute divisors
def get_divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i > 1:
                divs.append(i)
            other = n // i
            if other > 1 and other != i:
                divs.append(other)
    return divs

DIVISORS = get_divisors(BASE)
# Filter to odd divisors (all are odd since BASE is odd)
# Ensure distinct and > 1
DIVISORS = list(set(d for d in DIVISORS if d > 1))
DIVISORS.sort()

class CoveringSystem:
    def __init__(self):
        self.coverage_count = np.zeros(BASE, dtype=np.int32)
        self.uncovered_count = BASE
        self.active = {}  # modulus -> residue
        self.inactive = set(DIVISORS)
        
    def add_modulus(self, m, r):
        """Add modulus m with residue r. Returns change in uncovered_count."""
        if m in self.active:
            return 0
        self.active[m] = r
        self.inactive.discard(m)
        
        delta = 0
        for x in range(r, BASE, m):
            if self.coverage_count[x] == 0:
                delta -= 1
            self.coverage_count[x] += 1
        self.uncovered_count += delta
        return delta

    def remove_modulus(self, m):
        """Remove modulus m. Returns change in uncovered_count."""
        if m not in self.active:
            return 0
        r = self.active[m]
        del self.active[m]
        self.inactive.add(m)
        
        delta = 0
        for x in range(r, BASE, m):
            self.coverage_count[x] -= 1
            if self.coverage_count[x] == 0:
                delta += 1
        self.uncovered_count += delta
        return delta

    def change_residue(self, m, new_r):
        """Change residue of active modulus m to new_r. Returns change in uncovered_count."""
        if m not in self.active:
            return 0
        old_r = self.active[m]
        if old_r == new_r:
            return 0
            
        # Uncover old
        delta = 0
        for x in range(old_r, BASE, m):
            self.coverage_count[x] -= 1
            if self.coverage_count[x] == 0:
                delta += 1
                
        # Cover new
        for x in range(new_r, BASE, m):
            if self.coverage_count[x] == 0:
                delta -= 1
            self.coverage_count[x] += 1
            
        self.active[m] = new_r
        self.uncovered_count += delta
        return delta

    def get_classes(self):
        return [[m, r] for m, r in self.active.items()]

def find_best_residue(m, coverage_count):
    """
    Finds the residue r in [0, m) that covers the most currently uncovered residues
    given the current coverage_count (assuming m is NOT active in coverage_count).
    Returns (best_r, max_gain).
    """
    gains = np.zeros(m, dtype=np.int32)
    # We want to count how many x are uncovered (coverage_count[x] == 0)
    # and belong to the class (m, r).
    # x % m == r.
    # Iterate through all x. If uncovered, increment gains[x % m].
    # This is O(N).
    # Optimization: Only iterate if m is not too large? 
    # N=51975. Looping 51975 times in Python is fast (~5ms).
    # We can vectorize this using numpy.
    # indices = np.arange(BASE)
    # uncovered_mask = coverage_count == 0
    # residues = indices[uncovered_mask] % m
    # gains = np.bincount(residues, minlength=m)
    # best_r = np.argmax(gains)
    
    # Let's try the vectorized approach.
    indices = np.arange(BASE)
    uncovered_mask = coverage_count == 0
    if not np.any(uncovered_mask):
        return 0, 0
    residues = indices[uncovered_mask] % m
    gains = np.bincount(residues, minlength=m)
    best_r = int(np.argmax(gains))
    max_gain = int(gains[best_r])
    return best_r, max_gain

def greedy_initialization():
    sys_obj = CoveringSystem()
    # Sort divisors. 
    # Heuristic: Adding smaller moduli first might be better because they have more residues to choose from 
    # and constrain the problem more? Or larger first to cover broad areas?
    # Let's try ascending.
    divs_asc = sorted(DIVISORS)
    
    for m in divs_asc:
        # Find best residue for m given current state
        best_r, gain = find_best_residue(m, sys_obj.coverage_count)
        if gain > 0:
            sys_obj.add_modulus(m, best_r)
            
    # Second pass: try adding any remaining
    divs_asc = sorted(DIVISORS)
    for m in divs_asc:
        if m in sys_obj.active:
            continue
        best_r, gain = find_best_residue(m, sys_obj.coverage_count)
        if gain > 0:
            sys_obj.add_modulus(m, best_r)
            
    return sys_obj

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try changing residues using exact best improvement
        for m in list(sys_obj.active.keys()):
            old_r = sys_obj.active[m]
            
            # Temporarily remove m to see the landscape
            # We need to be careful with state modification.
            # Let's compute the best new residue assuming m is removed.
            
            # 1. Remove m
            delta_rem = sys_obj.remove_modulus(m)
            # 2. Find best residue for m in the new landscape
            best_new_r, gain = find_best_residue(m, sys_obj.coverage_count)
            # 3. Add m with best_new_r
            delta_add = sys_obj.add_modulus(m, best_new_r)
            
            new_score = sys_obj.uncovered_count
            
            if new_score < best_score:
                best_score = new_score
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
            else:
                # Revert: Remove m, Add m with old_r
                sys_obj.remove_modulus(m)
                sys_obj.add_modulus(m, old_r)
                
        # 2. Try removing moduli
        for m in list(sys_obj.active.keys()):
            r = sys_obj.active[m]
            delta = sys_obj.remove_modulus(m)
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
            else:
                # Revert
                sys_obj.add_modulus(m, r)
                
        # 3. Try adding moduli
        if sys_obj.inactive:
            # Try all inactive moduli? 
            # There are at most ~30 divisors. 
            # Cost per modulus: O(N) for find_best_residue + O(N/m) for add.
            # Total cost: O(N * |D|) ~ 50k * 30 = 1.5M ops.
            # This is feasible.
            for m in list(sys_obj.inactive):
                best_r, gain = find_best_residue(m, sys_obj.coverage_count)
                if gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        # Revert
                        sys_obj.remove_modulus(m)
                        
        if not improved:
            break
            
    return sys_obj, best_score

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization
    sys_obj = greedy_initialization()
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    # We can run local search for a while.
    # Each pass is O(|D| * N). 
    # Let's run it until no improvement or time limit.
    sys_obj, best_score = local_search(sys_obj, time_limit=min(30.0, DEADLINE - (time.time() - start_time)))
    
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 3: Simulated Annealing (if time remains)
    # The local search is quite strong. SA might help escape local minima.
    # But given the strict deadline and the strength of the exact local search, 
    # we might just use the remaining time for more local search passes 
    # or a randomized restart.
    
    # Let's try a few randomized restarts if time permits.
    best_global_score = best_score
    best_global_active = sys_obj.active.copy()
    
    while time.time() - start_time < DEADLINE - 5.0:
        # Restart with random initialization? 
        # Or perturb current best?
        # Perturbation is usually better.
        
        # Copy current best
        sys_obj2 = CoveringSystem()
        for m, r in best_global_active.items():
            sys_obj2.add_modulus(m, r)
            
        # Perturb: change residues of a few random moduli
        if sys_obj2.active:
            keys = list(sys_obj2.active.keys())
            for _ in range(min(3, len(keys))):
                m = random.choice(keys)
                old_r = sys_obj2.active[m]
                new_r = random.randint(0, m - 1)
                sys_obj2.change_residue(m, new_r)
                
        # Run local search on perturbed state
        remaining_time = DEADLINE - 5.0 - (time.time() - start_time)
        if remaining_time > 1.0:
            sys_obj2, score2 = local_search(sys_obj2, time_limit=remaining_time)
            if score2 < best_global_score:
                best_global_score = score2
                best_global_active = sys_obj2.active.copy()
                
    return [[m, r] for m, r in best_global_active.items()]
