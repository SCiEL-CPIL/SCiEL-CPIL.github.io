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

# Precompute indices for vectorized operations
INDICES = np.arange(BASE, dtype=np.int32)

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
        
        # Vectorized update
        # Indices: r, r+m, r+2m, ... < BASE
        idx = np.arange(r, BASE, m)
        # Count how many were uncovered before
        was_uncovered = self.coverage_count[idx] == 0
        delta = -np.sum(was_uncovered)
        
        self.coverage_count[idx] += 1
        self.uncovered_count += delta
        return delta

    def remove_modulus(self, m):
        """Remove modulus m. Returns change in uncovered_count."""
        if m not in self.active:
            return 0
        r = self.active[m]
        del self.active[m]
        self.inactive.add(m)
        
        # Vectorized update
        idx = np.arange(r, BASE, m)
        # Count how many become uncovered after removal
        # They become uncovered if coverage_count[idx] - 1 == 0 => coverage_count[idx] == 1
        becomes_uncovered = self.coverage_count[idx] == 1
        delta = np.sum(becomes_uncovered)
        
        self.coverage_count[idx] -= 1
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
        idx_old = np.arange(old_r, BASE, m)
        becomes_uncovered = self.coverage_count[idx_old] == 1
        delta = np.sum(becomes_uncovered)
        self.coverage_count[idx_old] -= 1
        
        # Cover new
        idx_new = np.arange(new_r, BASE, m)
        was_uncovered = self.coverage_count[idx_new] == 0
        delta -= np.sum(was_uncovered)
        self.coverage_count[idx_new] += 1
        
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
    uncovered_mask = coverage_count == 0
    if not np.any(uncovered_mask):
        return 0, 0
    
    # Get residues of all uncovered indices modulo m
    residues = INDICES[uncovered_mask] % m
    gains = np.bincount(residues, minlength=m)
    best_r = int(np.argmax(gains))
    max_gain = int(gains[best_r])
    return best_r, max_gain

def greedy_initialization(order='asc'):
    sys_obj = CoveringSystem()
    if order == 'asc':
        divs = sorted(DIVISORS)
    else:
        divs = sorted(DIVISORS, reverse=True)
    
    for m in divs:
        # Find best residue for m given current state
        best_r, gain = find_best_residue(m, sys_obj.coverage_count)
        if gain > 0:
            sys_obj.add_modulus(m, best_r)
            
    # Second pass: try adding any remaining
    for m in divs:
        if m in sys_obj.active:
            continue
        best_r, gain = find_best_residue(m, sys_obj.coverage_count)
        if gain > 0:
            sys_obj.add_modulus(m, best_r)
            
    return sys_obj

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    # We don't need to copy coverage_count for best_state if we just track score and active
    # But to restore, we might need it. However, local search modifies in place.
    # We can just keep the best active dict.
    best_active = sys_obj.active.copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try changing residues using exact best improvement
        for m in list(sys_obj.active.keys()):
            old_r = sys_obj.active[m]
            
            # 1. Remove m
            sys_obj.remove_modulus(m)
            # 2. Find best residue for m in the new landscape
            best_new_r, gain = find_best_residue(m, sys_obj.coverage_count)
            # 3. Add m with best_new_r
            sys_obj.add_modulus(m, best_new_r)
            
            new_score = sys_obj.uncovered_count
            
            if new_score < best_score:
                best_score = new_score
                improved = True
                best_active = sys_obj.active.copy()
            else:
                # Revert: Remove m, Add m with old_r
                sys_obj.remove_modulus(m)
                sys_obj.add_modulus(m, old_r)
                
        # 2. Try removing moduli
        for m in list(sys_obj.active.keys()):
            r = sys_obj.active[m]
            sys_obj.remove_modulus(m)
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                improved = True
                best_active = sys_obj.active.copy()
            else:
                # Revert
                sys_obj.add_modulus(m, r)
                
        # 3. Try adding moduli
        if sys_obj.inactive:
            for m in list(sys_obj.inactive):
                best_r, gain = find_best_residue(m, sys_obj.coverage_count)
                if gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_active = sys_obj.active.copy()
                    else:
                        # Revert
                        sys_obj.remove_modulus(m)
                        
        if not improved:
            # If no improvement, restore the best state found in this pass (which might be the start state if no improvement)
            # Actually, if no improvement, sys_obj is already in the state of the start of the pass (since all reverts happened)
            # So we can break.
            break
            
    return sys_obj, best_score, best_active

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Phase 1: Greedy Initialization (try both orders)
    sys_asc = greedy_initialization('asc')
    sys_desc = greedy_initialization('desc')
    
    if sys_asc.uncovered_count <= sys_desc.uncovered_count:
        sys_obj = sys_asc
    else:
        sys_obj = sys_desc
        
    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()
    
    # Phase 2: Local Search
    # Run local search until no improvement or time limit
    remaining = DEADLINE - (time.time() - start_time)
    if remaining < 1.0:
        return sys_obj.get_classes()
        
    sys_obj, best_score, best_active = local_search(sys_obj, time_limit=remaining * 0.8)
    
    if time.time() - start_time > DEADLINE:
        return [[m, r] for m, r in best_active.items()]
    
    # Phase 3: Randomized Restarts / Perturbation
    # If time remains, try to escape local minima
    best_global_score = best_score
    best_global_active = best_active
    
    # Reconstruct sys_obj from best_active if it diverged
    # local_search returns the modified sys_obj. If it improved, sys_obj is in the best state.
    # If it didn't improve, sys_obj is in the start state.
    # We should ensure sys_obj is in the best state before restarting.
    # The local_search function restores reverts, so if it broke due to no improvement, 
    # sys_obj is in the state it started with (which was the best from previous phases).
    # So sys_obj is valid.
    
    while time.time() - start_time < DEADLINE - 2.0:
        # Perturb current best
        sys_obj2 = CoveringSystem()
        for m, r in best_global_active.items():
            sys_obj2.add_modulus(m, r)
            
        if sys_obj2.active:
            keys = list(sys_obj2.active.keys())
            # Perturb 1-3 moduli
            num_perturb = min(random.randint(1, 3), len(keys))
            for _ in range(num_perturb):
                m = random.choice(keys)
                old_r = sys_obj2.active[m]
                new_r = random.randint(0, m - 1)
                sys_obj2.change_residue(m, new_r)
                
        remaining_time = DEADLINE - 2.0 - (time.time() - start_time)
        if remaining_time > 0.5:
            sys_obj2, score2, active2 = local_search(sys_obj2, time_limit=remaining_time)
            if score2 < best_global_score:
                best_global_score = score2
                best_global_active = active2
                
    return [[m, r] for m, r in best_global_active.items()]
