"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random
import heapq
import numpy as np

_DEADLINE = 310.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


class _AdmissibleTracker:
    """Incremental tracker for admissibility using residue class counts per prime."""
    
    def __init__(self, points, primes):
        self.point_set = set(points)
        self.primes = primes
        self.residue_counts = {}
        self.nonzero_classes = {}
        self.empty_class = {} # For active primes (nonzero == p-1)
        self.active_primes = set()
        
        for p in primes:
            counts = [0] * p
            for v in self.point_set:
                counts[v % p] += 1
            self.residue_counts[p] = counts
            nz = sum(1 for c in counts if c > 0)
            self.nonzero_classes[p] = nz
            if nz == p - 1:
                self.active_primes.add(p)
                for r in range(p):
                    if counts[r] == 0:
                        self.empty_class[p] = r
                        break
    
    def is_admissible(self):
        """Check if the current set is admissible."""
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                return False
        return True

    def can_add(self, v):
        """Check if v can be added without violating admissibility."""
        for p in self.active_primes:
            if v % p == self.empty_class[p]:
                return False
        return True

    def add(self, v):
        """Add a point, updating all prime residue counts."""
        if v in self.point_set:
            return
        self.point_set.add(v)
        
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            
            if counts[r] == 0:
                self.nonzero_classes[p] += 1
                nz = self.nonzero_classes[p]
                
                if nz == p - 1:
                    self.active_primes.add(p)
                    for rr in range(p):
                        if counts[rr] == 0:
                            self.empty_class[p] = rr
                            break
            counts[r] += 1

    def remove(self, v):
        """Remove a point, updating all prime residue counts."""
        if v not in self.point_set:
            return
        self.point_set.remove(v)
        
        for p in self.primes:
            r = v % p
            counts = self.residue_counts[p]
            
            if counts[r] == 1:
                self.nonzero_classes[p] -= 1
                nz = self.nonzero_classes[p]
                
                if p in self.active_primes:
                    self.active_primes.remove(p)
            
            counts[r] -= 1

    def covered_primes(self):
        """Return list of primes where all residue classes are covered."""
        result = []
        for p in self.primes:
            if p > len(self.point_set):
                continue
            if self.nonzero_classes[p] == p:
                result.append(p)
        return result


def _greedy_removal(points, primes_to_check, deadline, strategy="smallest"):
    """Greedy removal using incremental tracker."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    
    while time.time() < deadline:
        covered = tracker.covered_primes()
        if not covered:
            break
        
        if strategy == "smallest":
            p = min(covered)
        elif strategy == "largest":
            p = max(covered)
        elif strategy == "min_class":
            p = min(covered, key=lambda q: min(tracker.residue_counts[q]))
        elif strategy == "random":
            p = random.choice(covered)
        else:
            p = min(covered)
        
        counts = tracker.residue_counts[p]
        min_class = min(range(p), key=lambda r: counts[r])
        
        to_remove = [v for v in tracker.point_set if v % p == min_class]
        for v in to_remove:
            tracker.remove(v)
    
    return list(tracker.point_set)


def _fast_add_back(points, primes_to_check, deadline):
    """Optimized add-back using active_primes tracking."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    all_candidates = set(range(N + 1)) - tracker.point_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        valid_candidates = []
        for v in all_candidates:
            is_valid = True
            for p in tracker.active_primes:
                if v % p == tracker.empty_class[p]:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(v)
        
        if not valid_candidates:
            break
        
        v_to_add = min(valid_candidates)
        
        tracker.add(v_to_add)
        all_candidates.discard(v_to_add)
        improved = True
    
    return list(tracker.point_set)


def _branch_and_sieve():
    """
    Branch over residue classes of first 5 primes (2, 3, 5, 7, 11).
    For each combination, sieve [0, N], repair for larger primes, and add back.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    primes = _primes(N + 1)
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]
    
    # Precompute arange for numpy
    arr = np.arange(N + 1)
    
    best_set = []
    best_size = 0
    
    # Generate all combinations of excluded residues for small primes
    # 2 choices for 2, 3 for 3, 5 for 5, 7 for 7, 11 for 11 -> 2310 combos
    import itertools
    
    for combo in itertools.product(range(2), range(3), range(5), range(7), range(11)):
        if time.time() > deadline - 5:
            break
        
        # Apply sieve
        mask = np.ones(N + 1, dtype=bool)
        for p, r in zip(small_primes, combo):
            mask[arr % p == r] = False
        
        # Get candidate points
        candidates = np.nonzero(mask)[0]
        
        if len(candidates) == 0:
            continue
        
        # Convert to list for tracker
        init_points = candidates.tolist()
        
        # Phase 1: Repair for large primes
        # We only need to check primes > 11 because small primes are already handled by the sieve
        # However, the tracker needs to know about all primes to be correct if we add back later?
        # Actually, if we add back, we might violate small prime constraints?
        # No, the sieve ensured that NO point in the candidate set is in the excluded residue for small primes.
        # So any subset of this candidate set is automatically safe for small primes.
        # Therefore, we only need to worry about large primes during repair.
        # But for add-back, we need to be careful.
        # If we add a point from the full range [0, N], it might violate small prime constraints.
        # So we should restrict add-back candidates to the masked set?
        # If we restrict to the masked set, we are guaranteed to be safe for small primes.
        # This is a valid strategy: "Sieve-Add-Back".
        
        # Let's refine the add-back to only consider points that passed the sieve.
        # This makes the add-back much faster as the candidate pool is smaller.
        
        # Repair
        repaired = _greedy_removal(init_points, large_primes, deadline, strategy="min_class")
        
        # Add-back restricted to the sieve mask
        # We need a modified add_back that only considers points in the mask.
        # Or we can just pass the mask as a filter.
        
        # Let's implement a specialized add_back for this case.
        
        # For now, let's just use the standard add_back but it might try to add points that violate small primes.
        # If we use standard add_back, we must include small primes in the tracker.
        # Let's use all primes for the tracker to be safe, but the initial set is already safe for small primes.
        
        # To be efficient, we'll use all primes for the tracker in add_back.
        # But we can speed it up by pre-filtering candidates.
        
        # Let's just run the standard pipeline on the repaired set.
        # The standard add_back checks `can_add` which checks active_primes.
        # Since the initial set is safe for small primes, active_primes for small primes will be empty
        # or not blocking valid additions?
        # Wait, if a small prime p has 1 empty class (the one we sieved), and we have p-1 occupied classes,
        # it is "active" (p-1 nonzero). The empty class is the sieved one.
        # So `can_add` will correctly reject any point in the sieved class.
        # So standard add_back is safe!
        
        final_set = _fast_add_back(repaired, primes, deadline)
        
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = final_set
    
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Run the branch and sieve strategy
    best = _branch_and_sieve()
    
    # If we have time left, do a final polish on the best set
    if time.time() < deadline - 2 and best:
        primes = _primes(N + 1)
        # One more add-back pass
        best = _fast_add_back(best, primes, deadline)
        
    return best
# EVOLVE-BLOCK-END
