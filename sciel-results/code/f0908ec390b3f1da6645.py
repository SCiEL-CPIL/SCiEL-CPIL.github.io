import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 305.0  # seconds, keep safety margin under 330s

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple w.r.t. given primes."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


class _ResidueTracker:
    """
    Maintains residue counts for a set of points across relevant primes.
    Supports fast add/remove and admissibility check.
    """
    def __init__(self, primes, n_max):
        self.primes = [p for p in primes if p <= n_max + 1]
        self.n_max = n_max
        self.counts = {}
        self.full_count = {}
        for p in self.primes:
            self.counts[p] = [0] * p
            self.full_count[p] = 0
        self._pts = set()
    
    def add(self, x):
        if x in self._pts:
            return
        self._pts.add(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            r = x % p
            old = self.counts[p][r]
            new = old + 1
            if old == 0:
                self.full_count[p] += 1
            self.counts[p][r] = new
    
    def remove(self, x):
        if x not in self._pts:
            return
        self._pts.discard(x)
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            r = x % p
            old = self.counts[p][r]
            new = old - 1
            if new == 0:
                self.full_count[p] -= 1
            self.counts[p][r] = new
    
    def is_admissible(self):
        """Check if current set is admissible."""
        n_pts = len(self._pts)
        for p in self.primes:
            if p > n_pts:
                break
            if self.full_count[p] == p:
                return False
        return True


def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    Uses conflict-aware heuristic: remove the class with minimum new removals
    (i.e., maximum overlap with already-removed elements).
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Track which elements have been removed
    removed = set()
    
    for p in small_primes:
        if p > len(points):
            break
        
        # Check if all p residue classes are present
        class_counts = {}
        for v in points:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        if len(class_counts) < p:
            continue  # Already missing at least one class
        
        # All p classes present. Choose which class to remove.
        # Heuristic: remove the class whose elements have the most overlap
        # with already-removed elements (minimizes new removals)
        
        best_r = None
        best_new_removals = float('inf')
        
        for r in range(p):
            if r not in class_counts:
                continue
            # Count how many elements in this class are NOT already removed
            # (i.e., would be newly removed)
            new_removals = 0
            for v in points:
                if v % p == r and v not in removed:
                    new_removals += 1
            
            if new_removals < best_new_removals:
                best_new_removals = new_removals
                best_r = r
        
        # Remove the chosen class
        if best_r is not None:
            for v in list(points):
                if v % p == best_r:
                    points.discard(v)
                    removed.add(v)
    
    return sorted(points)


def _conflict_greedy(primes, n_max=N):
    """
    Conflict-graph greedy heuristic for maximum independent set.
    Models the problem as sequential constraint satisfaction:
    for each prime p, choose the residue class to block that minimizes
    cumulative removals considering interactions with other primes.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # For each prime, track which residue class we've decided to block
    blocked_classes = {}  # p -> r (the blocked residue class mod p)
    
    for p in small_primes:
        if p > len(points):
            break
        
        # Check if all p residue classes are present
        class_counts = {}
        for v in points:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        if len(class_counts) < p:
            continue  # Already missing at least one class
        
        # All p classes present. Choose which class to remove.
        # For each candidate residue class r, compute the "cost":
        # how many elements would be removed, weighted by how many other
        # primes' coverage would be affected.
        
        best_r = None
        best_cost = float('inf')
        
        for r in range(p):
            if r not in class_counts:
                continue
            
            # Elements in this class
            elements_in_class = [v for v in points if v % p == r]
            
            # Cost = number of elements + penalty for primes that would
            # still be fully covered after this removal
            cost = len(elements_in_class)
            
            # Check impact on other primes
            for q in small_primes:
                if q == p or q > len(points):
                    continue
                
                # Check if q is currently fully covered
                q_classes = set()
                for v in points:
                    q_classes.add(v % q)
                
                if len(q_classes) < q:
                    continue  # q already not fully covered
                
                # If we remove elements_in_class, would q lose full coverage?
                # For each residue s mod q, check if all elements with v%q==s
                # are in elements_in_class
                for s in range(q):
                    all_in_class = True
                    for v in points:
                        if v % q == s and v % p != r:
                            all_in_class = False
                            break
                    if all_in_class:
                        # Removing class r would save prime q
                        cost -= 10  # Reward for saving another prime
                        break
            
            if cost < best_cost:
                best_cost = cost
                best_r = r
        
        # Remove the chosen class
        if best_r is not None:
            blocked_classes[p] = best_r
            points = {v for v in points if v % p != best_r}
    
    return sorted(points)


def _multi_start_greedy(primes, n_max=N, deadline=None):
    """
    Multi-start greedy with different random removals to find larger admissible sets.
    Uses conflict-aware selection in the main loop.
    """
    best = None
    best_size = 0
    small_primes = [p for p in primes if p <= n_max + 1]
    
    if deadline is not None:
        start_time = time.time()
    
    # Start with the standard conflict-aware greedy
    greedy_result = _greedy_construct(primes, n_max)
    best = greedy_result
    best_size = len(best)
    
    if deadline is not None and time.time() > deadline - 15:
        return best
    
    # Try multiple random initial removals
    num_attempts = 100
    for attempt in range(num_attempts):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        # Start with full set, but randomly remove a few elements first
        points = set(range(n_max + 1))
        # Randomly remove some elements to break symmetry
        for _ in range(random.randint(1, 5)):
            if points:
                to_remove = random.choice(list(points))
                points.discard(to_remove)
        
        # Now run conflict-aware greedy repair
        removed = set(range(n_max + 1)) - points
        
        for p in small_primes:
            if p > len(points):
                break
            class_counts = Counter(v % p for v in points)
            if len(class_counts) == p:
                # Choose class with minimum new removals
                best_r = None
                best_new = float('inf')
                for r in range(p):
                    if r not in class_counts:
                        continue
                    new_removals = sum(1 for v in points if v % p == r and v not in removed)
                    if new_removals < best_new:
                        best_new = new_removals
                        best_r = r
                
                if best_r is not None:
                    for v in list(points):
                        if v % p == best_r:
                            points.discard(v)
                            removed.add(v)
        
        if len(points) > best_size:
            best = sorted(points)
            best_size = len(points)
    
    return best


def _local_search(points, primes, n_max=N, max_iters=100000, deadline=None):
    """
    Local search: try to add back removed elements to increase size.
    Uses efficient residue tracking and prioritizes candidates that
    don't complete any prime's residue coverage.
    """
    points_set = set(points)
    best = sorted(points_set)
    best_size = len(best)
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Build tracker
    tracker = _ResidueTracker(primes, n_max)
    for x in points_set:
        tracker.add(x)
    
    if deadline is not None:
        start_time = time.time()
    
    # Pre-compute which elements are not in points_set
    all_elements = set(range(n_max + 1))
    candidates = all_elements - points_set
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        if not candidates:
            break
        
        # Sample candidates to keep it fast
        sample_size = min(len(candidates), 30)
        sample = random.sample(list(candidates), sample_size)
        
        best_candidate = None
        best_score = float('inf')
        
        for candidate in sample:
            n_after = len(points_set) + 1
            would_violate = False
            critical_primes = 0
            
            for p in small_primes:
                if p > n_after:
                    break
                if tracker.full_count[p] == p:
                    would_violate = True
                    break
                r = candidate % p
                if tracker.counts[p][r] == 0:
                    if tracker.full_count[p] == p - 1:
                        would_violate = True
                        break
                    else:
                        critical_primes += 1
            
            if not would_violate:
                if critical_primes < best_score:
                    best_score = critical_primes
                    best_candidate = candidate
        
        if best_candidate is not None:
            points_set.add(best_candidate)
            tracker.add(best_candidate)
            candidates.discard(best_candidate)
            best = sorted(points_set)
            best_size = len(best)
    
    return best


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    An admissible tuple satisfies: for every prime p <= len(tuple),
    at least one residue class mod p is missed.
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    # First, get a baseline conflict-aware greedy solution
    baseline = _greedy_construct(primes, N)
    best = baseline
    best_size = len(best)
    
    deadline = start_time + DEADLINE
    
    # Try conflict greedy variant
    if time.time() < deadline - 60:
        conflict_result = _conflict_greedy(primes, N)
        if len(conflict_result) > best_size:
            best = conflict_result
            best_size = len(best)
    
    # Try multi-start greedy with conflict-aware selection
    if time.time() < deadline - 45:
        multi_result = _multi_start_greedy(primes, N, deadline=deadline)
        if len(multi_result) > best_size:
            best = multi_result
            best_size = len(best)
    
    # Local search to try to add elements back
    if time.time() < deadline - 20:
        local_result = _local_search(
            best, primes, N,
            max_iters=50000,
            deadline=deadline
        )
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
    
    # Second round of local search
    if time.time() < deadline - 10:
        local_result2 = _local_search(
            best, primes, N,
            max_iters=30000,
            deadline=deadline
        )
        if len(local_result2) > best_size:
            best = local_result2
            best_size = len(best)
    
    # Final verification
    n = len(best)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in best)
        if len(residues) == p:
            # Not admissible! Remove from the smallest class
            class_counts = Counter(v % p for v in best)
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = min_classes[0]
            best = [v for v in best if v % p != remove_class]
    
    return best

# EVOLVE-BLOCK-END
