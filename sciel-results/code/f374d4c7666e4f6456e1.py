"""
Rewritten program: find 20 integer points (|x|,|y|<=100) with no four collinear,
minimizing the largest subset with no three collinear.

Strategy:
1. Start with a known good configuration (e.g., parabola, but that gives score 20).
2. Use local search / simulated annealing to perturb points while maintaining
   the "no four collinear" constraint, trying to reduce the maximum no-three-collinear subset.
3. Use a fast scoring function: for a given set of 20 points, compute the largest
   subset with no three collinear via branch-and-bound (exact, but 20 is small enough
   with good pruning).

Key insight: The parabola y=x^2 gives score 20 (no three collinear at all).
We want to introduce some collinear triples to force the score down, while
ensuring no four are collinear.

Approach:
- Generate candidate configurations by placing points on a few lines (but not 4 on any line),
  or use a structured grid-like pattern with perturbations.
- Use simulated annealing: randomly move a point to a nearby integer location,
  check validity (no four collinear), compute score, accept if better or with prob exp(-delta/T).
- Deadline guard: track elapsed time, stop at ~110s.

For scoring: exact branch-and-bound on 20 points to find largest 3-collinear-free subset.
We can use the fact that a subset is 3-collinear-free iff no three of its points are collinear.
We search for the maximum size of such a subset.

Branch and bound:
- Sort points.
- Recursively try including/excluding points.
- Prune if current_size + remaining <= best.
- Also prune if adding a point creates a collinear triple with two already-included points.

This is exponential in worst case but 20 is manageable with good ordering and pruning.
Actually, worst case 2^20 = 1M which is fine. But we need to check collinearity with pairs.

Better: use the fact that if we have a set S, and we add point p, we need to check
for every pair (a,b) in S whether p, a, b are collinear. That's O(|S|^2) per addition.

We can precompute for each point p, the set of pairs (i,j) such that p, i, j are collinear.
Then during search, when we include p, we need to ensure that no pair (i,j) from the
current set is in the collinear-pairs-of-p set.

Let's implement this.
"""

import time
import random
import math

DEADLINE = 110.0  # seconds

def collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    # For each pair, count how many points lie on the line through them
    # But that's O(n^3). For n=20, that's 8000 checks, fine.
    # Better: for each pair (i,j), count points k such that i,j,k collinear.
    # If any count >= 4, return True.
    for i in range(n):
        for j in range(i + 1, n):
            count = 2  # i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                if collinear(points[i], points[j], points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def max_no_three_collinear(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound.
    """
    n = len(points)
    
    # Precompute: for each point index, the set of pairs (i,j) with i<j
    # such that points[i], points[j], points[k] are collinear.
    # coll_pairs[k] = set of frozenset({i,j}) for i<j, i!=k, j!=k, collinear
    coll_pairs = [set() for _ in range(n)]
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            for j in range(i + 1, n):
                if j == k:
                    continue
                if collinear(points[i], points[j], points[k]):
                    coll_pairs[k].add(frozenset((i, j)))
    
    # Branch and bound
    best = [0]
    
    def search(idx, current_set, current_pairs):
        """
        current_set: set of indices included so far
        current_pairs: set of frozenset({i,j}) for pairs in current_set
        """
        if len(current_set) + (n - idx) <= best[0]:
            return
        
        if idx == n:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
            return
        
        # Option 1: exclude point idx
        search(idx + 1, current_set, current_pairs)
        
        # Option 2: include point idx if it doesn't create a collinear triple
        # Check if any pair in coll_pairs[idx] is in current_pairs
        can_include = True
        for pair in coll_pairs[idx]:
            if pair in current_pairs:
                can_include = False
                break
        
        if can_include:
            new_set = current_set | {idx}
            # New pairs are (idx, j) for j in current_set
            new_pairs = set(current_pairs)
            for j in current_set:
                new_pairs.add(frozenset((idx, j)))
            search(idx + 1, new_set, new_pairs)
    
    search(0, set(), set())
    return best[0]

def construct():
    start_time = time.time()
    
    # Initial configuration: try a few structured patterns
    # Pattern 1: parabola (score 20)
    # Pattern 2: points on a few lines with perturbations
    # Let's start with a random valid configuration
    
    def random_valid_config():
        """Generate a random configuration with no four collinear."""
        for _ in range(1000):
            points = []
            for _ in range(20):
                for _attempt in range(100):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = (x, y)
                    if p in points:
                        continue
                    # Check no four collinear with existing points
                    new_points = points + [p]
                    if not has_four_collinear(new_points):
                        points.append(p)
                        break
                else:
                    return None
            if len(points) == 20:
                return points
        return None
    
    # Try to get an initial valid config
    points = random_valid_config()
    if points is None:
        # Fallback: use a structured pattern
        # Place points on 5 lines, 4 per line, but perturb to avoid 4 collinear
        points = []
        for line_idx in range(5):
            for t in range(4):
                # Line: y = line_idx * x + (line_idx * 10)
                x = t * 25 - 50
                y = line_idx * x + line_idx * 10
                # Perturb slightly
                x += random.randint(-2, 2)
                y += random.randint(-2, 2)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                points.append((x, y))
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        points = unique_points[:20]
        # If we don't have 20, add random ones
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in points:
                points.append(p)
    
    # Ensure no four collinear
    if has_four_collinear(points):
        points = random_valid_config()
        if points is None:
            # Last resort: use a known valid config
            # Points on parabola but with some perturbations
            points = []
            for x in range(-10, 10):
                y = x * x
                points.append((x, y))
    
    current_score = max_no_three_collinear(points)
    best_points = points[:]
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        # Randomly pick a point and move it
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        new_x = old_x + random.randint(-5, 5)
        new_y = old_y + random.randint(-5, 5)
        new_x = max(-100, min(100, new_x))
        new_y = max(-100, min(100, new_y))
        
        if (new_x, new_y) in points:
            continue
        
        points[idx] = (new_x, new_y)
        
        if has_four_collinear(points):
            points[idx] = (old_x, old_y)
            continue
        
        new_score = max_no_three_collinear(points)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = points[:]
        else:
            points[idx] = (old_x, old_y)
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return [[p[0], p[1]] for p in best_points]
