"""
Odd covering systems with moduli dividing 51975.
Structural change: model as weighted maximum bipartite matching (Hungarian algorithm)
where left nodes are divisors > 1 of 51975, right nodes are residues 0..51974,
and the weight of edge (m, r) is the number of residues in class r mod m that
would be covered (i.e., the size of the set {x in [0, 51975) : x ≡ r (mod m)}).

Since each modulus can only be used once, and each residue class mod m covers
a fixed number of residues, we want to assign each divisor to a residue class
to maximize total coverage. This is an assignment problem solvable by the
Hungarian algorithm. However, since |divisors| may be smaller than |residues|,
we pad the problem to make it square.

Actually, a simpler and more direct approach: since each (m, r) pair covers
exactly floor(51975/m) or ceil(51975/m) residues, and we want to maximize
coverage, we should pick for each divisor m the residue r that covers the
most residues. But since all residue classes mod m cover approximately the
same number of residues (differing by at most 1), the key is which divisors
to use and which residues to assign.

Better structural approach: Use scipy.optimize.linear_sum_assignment to solve
the assignment problem where we assign each divisor to a unique residue class
pattern. But since residues are 51975 and divisors are fewer, we need to think
differently.

Let me reconsider: The number of divisors of 51975 = 3^3 * 5^2 * 7 * 11 is
(3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48 divisors total, minus 1 (for divisor 1)
= 47 divisors > 1.

So we have 47 divisors to assign to residue classes. For each divisor m, choosing
residue r covers either floor(51975/m) or ceil(51975/m) residues. The exact count
depends on r: residues r where r < 51975 % m cover ceil(51975/m) residues, others
cover floor(51975/m).

So for each m, we can compute the "gain" of each residue choice. We want to pick
one residue per divisor to maximize total coverage. This is a maximum weight
matching in a bipartite graph where left side is divisors, right side is residues,
but since each divisor picks exactly one residue and residues can be shared (multiple
divisors can cover the same residue), it's not a standard matching.

Actually, this is a set cover / maximum coverage problem, not a matching problem.
The Hungarian algorithm doesn't directly apply because multiple divisors can cover
the same residue.

Let me use a different structural approach: simulated annealing with a good
initialization from greedy, plus efficient neighborhood evaluation.
"""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = 51975


def _divisors():
    """Get all divisors > 1 of BASE."""
    ds = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            ds.append(d)
    return sorted(ds)


def _coverage_count(m, r):
    """Number of residues in [0, BASE) that are ≡ r (mod m)."""
    # Count x in [0, BASE) with x ≡ r (mod m)
    # x = r + k*m for k = 0, 1, ...
    # Need r + k*m < BASE, so k < (BASE - r)/m
    # k ranges from 0 to floor((BASE-1-r)/m) if r < BASE, else 0
    if r >= BASE:
        return 0
    return (BASE - 1 - r) // m + 1


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s wall-clock
    
    divs = _divisors()
    n_divs = len(divs)
    
    # Precompute coverage counts for each (m, r)
    # cov[i][r] = number of residues covered by divisor divs[i] with residue r
    cov = np.zeros((n_divs, BASE), dtype=np.int32)
    for i, m in enumerate(divs):
        for r in range(m):
            cov[i, r] = _coverage_count(m, r)
    
    # Greedy initialization: pick (m, r) that covers most uncovered residues
    uncovered = np.ones(BASE, dtype=np.uint8)
    selected = [None] * n_divs  # selected[i] = residue for divs[i], or None
    used_divs = set()
    
    # For greedy, we need to evaluate which (m, r) covers the most uncovered
    # We'll use a priority-based approach
    
    order = list(range(n_divs))
    # Sort divisors by size descending (larger moduli cover fewer residues per class,
    # but have more classes; smaller moduli cover more per class)
    # Actually, let's just do greedy by best gain
    
    remaining = set(range(n_divs))
    
    while remaining and time.time() < deadline:
        best_gain = 0
        best_i = None
        best_r = None
        for i in remaining:
            m = divs[i]
            # For this m, find r that maximizes coverage of uncovered residues
            # uncovered[r::m] gives the uncovered residues in class r mod m
            # We need to sum uncovered[x] for x ≡ r (mod m)
            # Efficient: use slicing
            counts = np.array([uncovered[r::m].sum() for r in range(m)], dtype=np.int64)
            idx = int(np.argmax(counts))
            c = int(counts[idx])
            if c > best_gain:
                best_gain = c
                best_i = i
                best_r = idx
        if best_gain == 0:
            break
        selected[best_i] = best_r
        remaining.discard(best_i)
        uncovered[best_r::divs[best_i]] = 0
    
    # Build initial classes
    classes = []
    for i, r in enumerate(selected):
        if r is not None:
            classes.append([divs[i], r])
    
    # Local search: try to improve by changing residue assignments
    # For each selected divisor, try alternative residues
    polish_iters = 0
    max_polish = 200
    while time.time() < deadline and polish_iters < max_polish:
        polish_iters += 1
        improved = False
        for i, (m, r) in enumerate(classes):
            # Try each alternative residue
            current_uncovered = int(uncovered.sum())
            for r2 in range(m):
                if r2 == r:
                    continue
                # Tentatively change residue
                # First, unmark old class, then mark new
                # But we need to be careful: other classes may also cover these residues
                # Simpler: recompute uncovered from scratch (expensive but correct)
                # For speed, do incremental:
                # Save current state
                old_mask = uncovered[r::m].copy()
                new_mask = uncovered[r2::m].copy()
                
                # Remove old coverage
                # Residues in r::m that are only covered by this class become uncovered
                # This is complex; just do full recompute for correctness
                pass
            
            # Full recompute approach for correctness
            best_r2 = r
            best_uncovered = current_uncovered
            for r2 in range(m):
                if r2 == r:
                    continue
                # Build new uncovered
                u2 = np.ones(BASE, dtype=np.uint8)
                for j, (mj, rj) in enumerate(classes):
                    if j == i:
                        continue
                    u2[rj::mj] = 0
                u2[r2::m] = 0
                nu = int(u2.sum())
                if nu < best_uncovered:
                    best_uncovered = nu
                    best_r2 = r2
            
            if best_r2 != r:
                classes[i][1] = best_r2
                # Update uncovered
                uncovered = np.ones(BASE, dtype=np.uint8)
                for mj, rj in classes:
                    uncovered[rj::mj] = 0
                improved = True
        
        if not improved:
            break
    
    return classes
