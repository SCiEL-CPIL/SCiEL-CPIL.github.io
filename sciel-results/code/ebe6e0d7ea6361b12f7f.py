"""Program for the no-isosceles set task using a greedy independent set heuristic."""
import numpy as np

N = 64

def construct():
    """
    Construct a set of points in [0, N)^2 with no three forming an isosceles triangle.
    
    Strategy:
    1. Generate a candidate set of points (e.g., all grid points or a sparse subset).
    2. For each candidate point, determine which other points it 'conflicts' with.
       Two points conflict if adding a third point would form an isosceles triangle.
       However, the constraint is that NO three points in the final set form an isosceles triangle.
       
       A more direct approach:
       We want a set S such that for any distinct a, b, c in S, the triangle abc is not isosceles.
       This means for any pair (a, b) in S, there cannot be a c in S such that:
       - dist(a,c) == dist(a,b)
       - dist(b,c) == dist(a,b)
       - dist(a,c) == dist(b,c)
       
       This is a global constraint. A simple greedy approach:
       Iterate through points in a specific order. For each point p, check if adding p to the current set S
       creates any isosceles triangle with any two existing points a, b in S.
       
       Checking all pairs (a,b) in S for each new p is O(|S|^2) per point, which is O(N^4) total for N^2 points.
       With N=64, N^2=4096 points. 4096^2 is ~16M pairs per point? No, |S| grows.
       Worst case |S| ~ 100. 100^2 = 10k pairs per point. 4096 * 10k = 40M operations. This is feasible in Python if optimized?
       Maybe too slow in pure Python.
       
       Let's try a different heuristic:
       Points on a line y=0 only have isosceles triangles if they are collinear?
       If all points are on a line, any three distinct points form a degenerate triangle.
       Usually "triangle" implies non-collinear. If the problem allows degenerate triangles, then any 3 points on a line form an isosceles triangle (with base 0? No, distances are d1, d2, d1+d2. Isosceles if d1=d2 or d1=d1+d2 (impossible) or d2=d1+d2 (impossible). So if d1=d2, it's isosceles).
       The seed program uses a Sidon-like set on a line. This ensures no three points have equal pairwise distances?
       Actually, the seed ensures distinct pairwise differences.
       If we have points x1, x2, x3 on a line. Distances are |x1-x2|, |x2-x3|, |x1-x3|.
       Isosceles if any two are equal.
       Since |x1-x3| = |x1-x2| + |x2-x3|, it can only equal one of them if the other is 0.
       So the only way to be isosceles on a line is |x1-x2| = |x2-x3|.
       The seed program's logic:
       `diffs = {abs(x - y) for y in xs}`
       `if not (diffs & used):`
       This ensures that the distance from the new point to any existing point has not been seen before as a distance between any two existing points?
       Wait, `used` accumulates differences.
       If I add x, the new differences are |x-y| for all y in xs.
       If any of these is in `used`, it means there exist y1, y2 in xs such that |y1-y2| == |x-y| for some y.
       This prevents x from being equidistant to y as y1 is to y2.
       This seems to prevent isosceles triangles where the new point is the apex?
       
       Let's stick to the 2D grid but use a faster heuristic.
       
       Alternative: Randomized Greedy with Backtracking? No, must be fast.
       
       Let's try a greedy approach on the full grid, but check conflicts efficiently.
       For a new point p, it conflicts with a pair (a,b) in S if:
       1. dist(p,a) == dist(p,b)  (p is apex)
       2. dist(p,a) == dist(a,b)  (a is apex)
       3. dist(p,b) == dist(a,b)  (b is apex)
       
       We can maintain for each point a in S:
       - Set of distances from a to all other points in S.
       
       When adding p:
       For each a in S:
         d_pa = dist(p, a)
         If d_pa is in distances_from_a, then there exists b in S such that dist(a,b) == d_pa.
         This creates an isosceles triangle (a is apex? No, dist(a,p)==dist(a,b), so a is the apex).
         So if d_pa is in `distances_from[a]`, we cannot add p.
         
         Also, we need to check if p is the apex.
         This requires checking if there are two points b, c in S such that dist(p,b) == dist(p,c).
         This is equivalent to: for each distance d, if there are at least two points in S at distance d from p, then p is the apex of an isosceles triangle.
         
       So the check for adding p to S is:
       1. For each a in S: if dist(p,a) in distances_from[a], return False.
       2. Count distances from p to all points in S. If any distance count >= 2, return False.
       
       Complexity: O(|S|) for step 1 (with hash sets) and O(|S|) for step 2.
       Total O(|S|^2) for all points? No, O(|S|) per point, so O(N^2 * |S|) total.
       If |S| is small (~100), and N^2=4096, then 4096 * 100 = 409,600 operations. Very fast.
       
       We need to order the points to maximize the size of S.
       Random order? Or spatial order?
       Let's try random order with a fixed seed for reproducibility, or multiple random restarts.
       Since the score is deterministic, we should use a deterministic heuristic or a fixed seed.
       
       Let's implement this greedy algorithm.
    """
    # Create a list of all grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    
    # Shuffle points deterministically? Or try different orders?
    # Let's try a few different deterministic orderings and pick the best.
    # Orderings:
    # 1. Row-major
    # 2. Column-major
    # 3. Diagonal
    # 4. Random with seed 42
    
    import random
    
    def greedy_construct(order):
        S = []
        distances_from = {}  # point -> set of distances to other points in S
        
        for p in order:
            can_add = True
            
            # Check condition 1: p is not the base of an isosceles triangle with apex a in S
            # i.e., dist(p, a) == dist(a, b) for some b in S
            # This is equivalent to dist(p, a) being in distances_from[a]
            for a in S:
                d = (p[0] - a[0])**2 + (p[1] - a[1])**2
                if d in distances_from[a]:
                    can_add = False
                    break
            
            if not can_add:
                continue
                
            # Check condition 2: p is not the apex of an isosceles triangle
            # i.e., there are no two points b, c in S such that dist(p, b) == dist(p, c)
            # We need to count distances from p to all points in S
            dist_counts = {}
            for a in S:
                d = (p[0] - a[0])**2 + (p[1] - a[1])**2
                dist_counts[d] = dist_counts.get(d, 0) + 1
                if dist_counts[d] >= 2:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Add p to S
            S.append(p)
            distances_from[p] = set()
            
            # Update distances_from for all existing points in S
            for a in S[:-1]:  # Exclude p itself
                d = (p[0] - a[0])**2 + (p[1] - a[1])**2
                distances_from[a].add(d)
                distances_from[p].add(d)
            
        return S

    # Try multiple orderings
    best_set = []
    
    # Ordering 1: Row-major
    order1 = [(x, y) for y in range(N) for x in range(N)]
    s1 = greedy_construct(order1)
    if len(s1) > len(best_set):
        best_set = s1
        
    # Ordering 2: Column-major
    order2 = [(x, y) for x in range(N) for y in range(N)]
    s2 = greedy_construct(order2)
    if len(s2) > len(best_set):
        best_set = s2
        
    # Ordering 3: Random with seed 0
    order3 = points[:]
    random.seed(0)
    random.shuffle(order3)
    s3 = greedy_construct(order3)
    if len(s3) > len(best_set):
        best_set = s3
        
    # Ordering 4: Random with seed 1
    order4 = points[:]
    random.seed(1)
    random.shuffle(order4)
    s4 = greedy_construct(order4)
    if len(s4) > len(best_set):
        best_set = s4
        
    # Ordering 5: Random with seed 2
    order5 = points[:]
    random.seed(2)
    random.shuffle(order5)
    s5 = greedy_construct(order5)
    if len(s5) > len(best_set):
        best_set = s5

    return best_set
