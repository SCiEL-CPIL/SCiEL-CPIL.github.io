"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
TIME_LIMIT = 280.0
M = N * N  # total grid points

# EVOLVE-BLOCK-START


def _build_distance_structure():
    """
    Precompute squared distances for all pairs in the N x N grid.
    Returns:
        d2: (M, M) array of squared distances
        pts: (M, 2) array of point coordinates
    """
    x = np.arange(N, dtype=np.float64)
    y = np.arange(N, dtype=np.float64)
    gx, gy = np.meshgrid(x, y, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2)
    
    # Compute all pairwise squared distances
    # Using broadcasting: (M, 1, 2) - (1, M, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = (diff ** 2).sum(axis=2)  # (M, M)
    
    return d2, pts


class NoIsoscelesTracker:
    """
    Efficient tracker for the no-isosceles constraint.
    
    A set S is valid iff:
    1. For every p in S, all distances from p to other points in S are distinct.
    2. For every pair (p, q) in S, dist(p,q) does not appear as a distance from q to any other point in S.
    
    Actually the constraint is simpler: no three points form an isosceles triangle.
    Three points p, q, r form an isosceles triangle iff:
    - dist(p,q) == dist(p,r) and p != q, r
    - dist(q,p) == dist(q,r) and q != p, r
    - dist(r,p) == dist(r,q) and r != p, q
    (with non-degeneracy, but on integer grid we just check the distance equalities)
    
    We maintain:
    - active: boolean array of length M
    - dist_sets[p]: set of squared distances from p to all other active points
    """
    
    def __init__(self):
        self.active = None
        self.dist_sets = None
    
    def init_with_greedy(self, d2, order, deadline):
        """Build initial set using greedy order."""
        M = d2.shape[0]
        self.active = np.zeros(M, dtype=bool)
        self.dist_sets = [set() for _ in range(M)]
        
        count = 0
        for idx in order:
            if time.time() > deadline:
                break
            
            # Check if idx can be added
            valid = True
            seen_dists = set()
            for j in range(M):
                if self.active[j]:
                    d = d2[idx, j]
                    if d in seen_dists:
                        valid = False
                        break
                    seen_dists.add(d)
                    if d in self.dist_sets[j]:
                        valid = False
                        break
            
            if valid:
                self.active[idx] = True
                for j in range(M):
                    if self.active[j]:
                        d = d2[idx, j]
                        self.dist_sets[idx].add(d)
                        self.dist_sets[j].add(d)
                count += 1
        
        return count
    
    def can_add(self, idx, d2):
        """Check if idx can be added to the current set."""
        if self.active[idx]:
            return False
        
        seen_dists = set()
        active_indices = np.where(self.active)[0]
        for j in active_indices:
            d = d2[idx, j]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            if d in self.dist_sets[j]:
                return False
        
        return True
    
    def add(self, idx, d2):
        """Add point idx (assumes can_add returned True)."""
        self.active[idx] = True
        active_indices = np.where(self.active)[0]
        for j in active_indices:
            d = d2[idx, j]
            self.dist_sets[idx].add(d)
            self.dist_sets[j].add(d)
    
    def remove(self, idx, d2):
        """Remove point idx from the set."""
        self.active[idx] = False
        active_indices = np.where(self.active)[0]
        for j in active_indices:
            d = d2[idx, j]
            if d in self.dist_sets[j]:
                self.dist_sets[j].discard(d)
        self.dist_sets[idx] = set()
    
    def rebuild(self, d2, active):
        """Rebuild from an active boolean array."""
        M = d2.shape[0]
        self.active = active.copy()
        self.dist_sets = [set() for _ in range(M)]
        
        active_indices = np.where(self.active)[0]
        for i, ai in enumerate(active_indices):
            for j in range(i + 1, len(active_indices)):
                aj = active_indices[j]
                d = d2[ai, aj]
                self.dist_sets[ai].add(d)
                self.dist_sets[aj].add(d)
    
    def size(self):
        return int(self.active.sum())


def _greedy_init(d2, seed, deadline):
    """Greedy initialization with random ordering."""
    M = d2.shape[0]
    tracker = NoIsoscelesTracker()
    
    rng = random.Random(seed)
    order = list(range(M))
    rng.shuffle(order)
    
    count = tracker.init_with_greedy(d2, order, deadline)
    return tracker.active.copy(), count


def _simulated_annealing(d2, initial_active, seed, deadline):
    """
    Simulated annealing with incremental violation tracking.
    
    State: active boolean array
    Move: add or remove a random point
    Cost: -number of points (we maximize)
    Constraint: no isosceles triangles
    
    We use a penalty-based approach:
    - If the move creates a violation, we reject unless at high temperature
    - Track violations incrementally for speed
    
    Actually, let's use a different approach:
    - Only consider moves that keep the set valid
    - Try to add a random inactive point; if valid, accept
    - Try to remove a random active point and add a different one
    """
    M = d2.shape[0]
    tracker = NoIsoscelesTracker()
    tracker.rebuild(d2, initial_active)
    current_size = tracker.size()
    best_size = current_size
    best_active = initial_active.copy()
    
    rng = random.Random(seed)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    n_iters = int((deadline - time.time()) * 500)  # estimate
    n_iters = max(n_iters, 1000)
    
    T = T_start
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iters)
        
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try to add a random inactive point
            inactive = np.where(~tracker.active)[0]
            if len(inactive) == 0:
                continue
            idx = inactive[rng.randrange(len(inactive))]
            
            if tracker.can_add(idx, d2):
                tracker.add(idx, d2)
                new_size = tracker.size()
                if new_size > best_size:
                    best_size = new_size
                    best_active = tracker.active.copy()
            else:
                # Reject (no penalty in this approach)
                pass
        
        else:
            # Try a swap: remove one, add another
            active = np.where(tracker.active)[0]
            if len(active) < 2:
                continue
            
            # Remove a random active point
            rem_idx = active[rng.randrange(len(active))]
            tracker.remove(rem_idx, d2)
            
            # Try to add a random inactive point
            inactive = np.where(~tracker.active)[0]
            if len(inactive) > 0:
                added = False
                for attempt in range(10):
                    add_idx = inactive[rng.randrange(len(inactive))]
                    if tracker.can_add(add_idx, d2):
                        tracker.add(add_idx, d2)
                        new_size = tracker.size()
                        if new_size > best_size:
                            best_size = new_size
                            best_active = tracker.active.copy()
                        added = True
                        break
                if not added:
                    # Restore removed point
                    tracker.add(rem_idx, d2)
    
    return best_active


def _local_search(d2, initial_active, seed, deadline):
    """
    Local search: repeatedly try random swaps (remove one, add another).
    Accept if size increases or stays the same.
    """
    M = d2.shape[0]
    tracker = NoIsoscelesTracker()
    tracker.rebuild(d2, initial_active)
    current_size = tracker.size()
    best_size = current_size
    best_active = initial_active.copy()
    
    rng = random.Random(seed)
    
    stall = 0
    max_stall = 1000
    
    while time.time() < deadline:
        if stall > max_stall:
            break
        
        # Try a random swap
        active = np.where(tracker.active)[0]
        if len(active) < 1:
            break
        
        rem_idx = active[rng.randrange(len(active))]
        tracker.remove(rem_idx, d2)
        
        # Try to find a replacement
        inactive = np.where(~tracker.active)[0]
        if len(inactive) == 0:
            tracker.add(rem_idx, d2)
            continue
        
        found = False
        # Try up to 50 random candidates
        for attempt in range(50):
            add_idx = inactive[rng.randrange(len(inactive))]
            if tracker.can_add(add_idx, d2):
                tracker.add(add_idx, d2)
                found = True
                new_size = tracker.size()
                if new_size > best_size:
                    best_size = new_size
                    best_active = tracker.active.copy()
                    stall = 0
                else:
                    stall += 1
                break
        
        if not found:
            # Restore removed point
            tracker.add(rem_idx, d2)
            stall += 1
    
    return best_active


def _multi_start(d2, seed, deadline, num_starts=5):
    """
    Multi-start: run multiple greedy initializations and local searches.
    """
    best_active = None
    best_size = 0
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Greedy init with different seed
        init_deadline = deadline - 10
        active, count = _greedy_init(d2, seed + i * 100, init_deadline)
        
        if count > best_size:
            best_size = count
            best_active = active.copy()
        
        # Local search
        ls_deadline = deadline - 5
        improved = _local_search(d2, active, seed + i * 100 + 50, ls_deadline)
        if improved.sum() > best_size:
            best_size = int(improved.sum())
            best_active = improved.copy()
    
    return best_active, best_size


def _simulated_annealing_v2(d2, initial_active, seed, deadline):
    """
    Improved SA with better move selection and incremental tracking.
    
    Key insight: Instead of checking all active points for can_add,
    we maintain a "conflict count" for each inactive point.
    """
    M = d2.shape[0]
    tracker = NoIsoscelesTracker()
    tracker.rebuild(d2, initial_active)
    current_size = tracker.size()
    best_size = current_size
    best_active = initial_active.copy()
    
    rng = random.Random(seed)
    
    # For speed, precompute which points are "near" each other
    # Actually, let's just use the tracker but with faster can_add
    
    n_iters = int((deadline - time.time()) * 200)
    n_iters = max(n_iters, 500)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random inactive point
        inactive = np.where(~tracker.active)[0]
        if len(inactive) == 0:
            break
        
        idx = inactive[rng.randrange(len(inactive))]
        
        if tracker.can_add(idx, d2):
            tracker.add(idx, d2)
            new_size = tracker.size()
            if new_size > best_size:
                best_size = new_size
                best_active = tracker.active.copy()
        else:
            # Try a swap instead
            active = np.where(tracker.active)[0]
            if len(active) > 0:
                rem_idx = active[rng.randrange(len(active))]
                tracker.remove(rem_idx, d2)
                
                # Try to add idx or another point
                if tracker.can_add(idx, d2):
                    tracker.add(idx, d2)
                    new_size = tracker.size()
                    if new_size > best_size:
                        best_size = new_size
                        best_active = tracker.active.copy()
                else:
                    # Try a few other candidates
                    for attempt in range(5):
                        alt_idx = inactive[rng.randrange(len(inactive))]
                        if tracker.can_add(alt_idx, d2):
                            tracker.add(alt_idx, d2)
                            new_size = tracker.size()
                            if new_size > best_size:
                                best_size = new_size
                                best_active = tracker.active.copy()
                            break
                    else:
                        # Restore removed point
                        tracker.add(rem_idx, d2)
    
    return best_active


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 15
    
    # Precompute distance structure
    d2, pts = _build_distance_structure()
    M = N * N
    
    best_active = None
    best_size = 0
    
    # Phase 1: Multiple greedy initializations
    for seed in range(8):
        if time.time() > deadline - 60:
            break
        active, count = _greedy_init(d2, seed, deadline - 60)
        if count > best_size:
            best_size = count
            best_active = active.copy()
    
    if best_active is None:
        best_active = np.zeros(M, dtype=bool)
        best_size = 0
    
    # Phase 2: Local search from best
    if best_size > 0 and time.time() < deadline - 20:
        improved = _local_search(d2, best_active, 0, deadline - 20)
        if improved.sum() > best_size:
            best_size = int(improved.sum())
            best_active = improved.copy()
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        improved = _simulated_annealing_v2(d2, best_active, 100, deadline - 5)
        if improved.sum() > best_size:
            best_size = int(improved.sum())
            best_active = improved.copy()
    
    # Phase 4: More local search if time remains
    if time.time() < deadline - 5:
        improved = _local_search(d2, best_active, 200, deadline - 3)
        if improved.sum() > best_size:
            best_size = int(improved.sum())
            best_active = improved.copy()
    
    # Convert to list of (x, y) tuples
    active_indices = np.where(best_active)[0]
    result = [(int(idx // N), int(idx % N)) for idx in active_indices]
    
    return result

# EVOLVE-BLOCK-END
