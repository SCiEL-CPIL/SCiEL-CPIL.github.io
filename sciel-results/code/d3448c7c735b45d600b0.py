import time
import random
import math
import numpy as np

BASE = 51975

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

DIVISORS = list(set(get_divisors(BASE)))
DIVISORS.sort()

# Precompute residue ranges for fast access
SMALL_MOD_THRESHOLD = 100  # Full search below this, sample above

class CoveringSystem:
    def __init__(self):
        self.coverage_count = np.zeros(BASE, dtype=np.int32)
        self.uncovered_count = BASE
        self.active = {}
        self.inactive = set(DIVISORS)
        
    def add_modulus(self, m, r):
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
        if m not in self.active:
            return 0
        old_r = self.active[m]
        if old_r == new_r:
            return 0
        delta = 0
        for x in range(old_r, BASE, m):
            self.coverage_count[x] -= 1
            if self.coverage_count[x] == 0:
                delta += 1
        for x in range(new_r, BASE, m):
            if self.coverage_count[x] == 0:
                delta -= 1
            self.coverage_count[x] += 1
        self.active[m] = new_r
        self.uncovered_count += delta
        return delta

    def get_classes(self):
        return [[m, r] for m, r in self.active.items()]

def find_best_residue(m, coverage_count, full_search=False):
    """
    Find the residue r in [0, m) that covers the most currently uncovered residues.
    For small moduli or when full_search=True, enumerate all residues.
    Otherwise, sample intelligently.
    """
    if m <= SMALL_MOD_THRESHOLD or full_search:
        # Full enumeration
        gains = np.zeros(m, dtype=np.int32)
        uncovered_mask = coverage_count == 0
        if not np.any(uncovered_mask):
            return 0, 0
        residues = np.where(uncovered_mask)[0] % m
        gains = np.bincount(residues, minlength=m)
        best_r = int(np.argmax(gains))
        max_gain = int(gains[best_r])
        return best_r, max_gain
    else:
        # Smart sampling for large moduli
        uncovered_mask = coverage_count == 0
        if not np.any(uncovered_mask):
            return 0, 0
        uncovered_indices = np.where(uncovered_mask)[0]
        # Sample up to 20000 uncovered positions
        sample_size = min(len(uncovered_indices), 20000)
        if sample_size < len(uncovered_indices):
            sampled = uncovered_indices[random.sample(range(len(uncovered_indices)), sample_size)]
        else:
            sampled = uncovered_indices
        residues = sampled % m
        gains = np.bincount(residues, minlength=m)
        best_r = int(np.argmax(gains))
        max_gain = int(gains[best_r])
        return best_r, max_gain

def greedy_initialization():
    def _greedy(order):
        sys_obj = CoveringSystem()
        for m in order:
            best_r, gain = find_best_residue(m, sys_obj.coverage_count)
            if gain > 0:
                sys_obj.add_modulus(m, best_r)
        # Second pass
        for m in order:
            if m in sys_obj.active:
                continue
            best_r, gain = find_best_residue(m, sys_obj.coverage_count)
            if gain > 0:
                sys_obj.add_modulus(m, best_r)
        return sys_obj

    divs_asc = sorted(DIVISORS)
    divs_desc = sorted(DIVISORS, reverse=True)

    sys_asc = _greedy(divs_asc)
    sys_desc = _greedy(divs_desc)

    if sys_desc.uncovered_count < sys_asc.uncovered_count:
        return sys_desc
    return sys_asc

def local_search(sys_obj, time_limit=5.0):
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        active_keys = list(sys_obj.active.keys())
        
        # 1. Try changing residues - full search for small, sampled for large
        for m in active_keys:
            old_r = sys_obj.active[m]
            full = m <= SMALL_MOD_THRESHOLD
            
            delta_rem = sys_obj.remove_modulus(m)
            best_new_r, gain = find_best_residue(m, sys_obj.coverage_count, full_search=full)
            delta_add = sys_obj.add_modulus(m, best_new_r)
            
            new_score = sys_obj.uncovered_count
            
            if new_score < best_score:
                best_score = new_score
                improved = True
                best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
            else:
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
                sys_obj.add_modulus(m, r)
                
        # 3. Try adding moduli
        if sys_obj.inactive:
            inactive_list = list(sys_obj.inactive)
            random.shuffle(inactive_list)
            for m in inactive_list[:min(10, len(inactive_list))]:
                best_r, gain = find_best_residue(m, sys_obj.coverage_count)
                if gain > 0:
                    sys_obj.add_modulus(m, best_r)
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        sys_obj.remove_modulus(m)
                        
        # 4. Try 2-opt: change two residues simultaneously
        active_keys = list(sys_obj.active.keys())
        if len(active_keys) >= 2 and time.time() - start_time < time_limit - 0.5:
            max_pairs = min(100, len(active_keys) * (len(active_keys) - 1) // 2)
            random.shuffle(active_keys)
            pairs_tried = 0
            for i in range(len(active_keys)):
                for j in range(i+1, len(active_keys)):
                    if pairs_tried >= max_pairs:
                        break
                    m1 = active_keys[i]
                    m2 = active_keys[j]
                    r1_old = sys_obj.active[m1]
                    r2_old = sys_obj.active[m2]
                    
                    sys_obj.remove_modulus(m1)
                    sys_obj.remove_modulus(m2)
                    
                    full1 = m1 <= SMALL_MOD_THRESHOLD
                    full2 = m2 <= SMALL_MOD_THRESHOLD
                    best_r1, gain1 = find_best_residue(m1, sys_obj.coverage_count, full_search=full1)
                    sys_obj.add_modulus(m1, best_r1)
                    best_r2, gain2 = find_best_residue(m2, sys_obj.coverage_count, full_search=full2)
                    sys_obj.add_modulus(m2, best_r2)
                    
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                        best_state = (sys_obj.active.copy(), sys_obj.coverage_count.copy())
                    else:
                        sys_obj.remove_modulus(m1)
                        sys_obj.remove_modulus(m2)
                        sys_obj.add_modulus(m1, r1_old)
                        sys_obj.add_modulus(m2, r2_old)
                    
                    pairs_tried += 1
                    if time.time() - start_time >= time_limit:
                        break
                
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                break
        else:
            no_improve_count = 0
            
    return sys_obj, best_score

def final_polish(sys_obj, time_limit=5.0):
    """High-precision polish: exhaustive residue search on all small moduli, repeated passes."""
    start_time = time.time()
    best_score = sys_obj.uncovered_count
    
    while time.time() - start_time < time_limit:
        improved = False
        active_keys = list(sys_obj.active.keys())
        
        # Pass 1: Exhaustive search on small moduli
        for m in sorted(active_keys):
            if m > SMALL_MOD_THRESHOLD:
                continue
            old_r = sys_obj.active[m]
            sys_obj.remove_modulus(m)
            best_new_r, gain = find_best_residue(m, sys_obj.coverage_count, full_search=True)
            sys_obj.add_modulus(m, best_new_r)
            
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                improved = True
        
        # Pass 2: Try removing and re-adding with exhaustive search
        for m in list(sys_obj.active.keys()):
            if m > SMALL_MOD_THRESHOLD * 2:
                continue
            r = sys_obj.active[m]
            sys_obj.remove_modulus(m)
            best_r, gain = find_best_residue(m, sys_obj.coverage_count, full_search=True)
            sys_obj.add_modulus(m, best_r)
            
            if sys_obj.uncovered_count < best_score:
                best_score = sys_obj.uncovered_count
                improved = True
        
        # Pass 3: Try 2-opt on small moduli only
        small_keys = [m for m in sys_obj.active.keys() if m <= SMALL_MOD_THRESHOLD]
        if len(small_keys) >= 2:
            random.shuffle(small_keys)
            max_pairs = min(50, len(small_keys) * (len(small_keys) - 1) // 2)
            pairs_tried = 0
            for i in range(len(small_keys)):
                for j in range(i+1, len(small_keys)):
                    if pairs_tried >= max_pairs:
                        break
                    m1 = small_keys[i]
                    m2 = small_keys[j]
                    r1_old = sys_obj.active[m1]
                    r2_old = sys_obj.active[m2]
                    
                    sys_obj.remove_modulus(m1)
                    sys_obj.remove_modulus(m2)
                    
                    best_r1, _ = find_best_residue(m1, sys_obj.coverage_count, full_search=True)
                    sys_obj.add_modulus(m1, best_r1)
                    best_r2, _ = find_best_residue(m2, sys_obj.coverage_count, full_search=True)
                    sys_obj.add_modulus(m2, best_r2)
                    
                    if sys_obj.uncovered_count < best_score:
                        best_score = sys_obj.uncovered_count
                        improved = True
                    else:
                        sys_obj.remove_modulus(m1)
                        sys_obj.remove_modulus(m2)
                        sys_obj.add_modulus(m1, r1_old)
                        sys_obj.add_modulus(m2, r2_old)
                    
                    pairs_tried += 1
                    if time.time() - start_time >= time_limit:
                        break
        
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
    remaining = DEADLINE - (time.time() - start_time)
    sys_obj, best_score = local_search(sys_obj, time_limit=min(30.0, remaining * 0.5))

    if time.time() - start_time > DEADLINE:
        return sys_obj.get_classes()

    # Phase 3: Multi-start with perturbations
    best_global_score = best_score
    best_global_active = sys_obj.active.copy()
    
    num_starts = 0
    max_starts = 10
    
    while time.time() - start_time < DEADLINE - 10.0 and num_starts < max_starts:
        sys_obj2 = CoveringSystem()
        for m, r in best_global_active.items():
            sys_obj2.add_modulus(m, r)

        # Perturbation strategies
        if sys_obj2.active:
            keys = list(sys_obj2.active.keys())
            strategy = random.random()
            
            if strategy < 0.3:
                # Random perturbation of a few residues
                n_perturb = random.randint(2, min(4, len(keys)))
                for _ in range(n_perturb):
                    m = random.choice(keys)
                    new_r = random.randint(0, m - 1)
                    sys_obj2.change_residue(m, new_r)
            elif strategy < 0.6:
                # Remove one modulus and re-add with best residue
                m_remove = random.choice(keys)
                sys_obj2.remove_modulus(m_remove)
                best_r, gain = find_best_residue(m_remove, sys_obj2.coverage_count)
                sys_obj2.add_modulus(m_remove, best_r)
            else:
                # Swap residues between two compatible moduli
                if len(keys) >= 2:
                    m1, m2 = random.sample(keys, 2)
                    r1 = sys_obj2.active[m1]
                    r2 = sys_obj2.active[m2]
                    if r2 < m1 and r1 < m2:
                        sys_obj2.change_residue(m1, r2)
                        sys_obj2.change_residue(m2, r1)
        
        # Run local search on perturbed state
        remaining_time = DEADLINE - 10.0 - (time.time() - start_time)
        if remaining_time > 2.0:
            sys_obj2, score2 = local_search(sys_obj2, time_limit=min(15.0, remaining_time * 0.6))
            if score2 < best_global_score:
                best_global_score = score2
                best_global_active = sys_obj2.active.copy()
        
        num_starts += 1

    # Phase 4: Final high-precision polish
    if time.time() - start_time < DEADLINE - 2.0:
        sys_obj3 = CoveringSystem()
        for m, r in best_global_active.items():
            sys_obj3.add_modulus(m, r)
        
        polish_time = min(10.0, DEADLINE - 2.0 - (time.time() - start_time))
        if polish_time > 1.0:
            sys_obj3, polish_score = final_polish(sys_obj3, time_limit=polish_time)
            if polish_score < best_global_score:
                best_global_score = polish_score
                best_global_active = sys_obj3.active.copy()

    return [[m, r] for m, r in best_global_active.items()]
