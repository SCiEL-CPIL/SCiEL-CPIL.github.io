"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    max_d2 = int(dist2.max())
    
    # used_dists[i] is a set of distances used by point i to other points in S
    # Using sets for O(1) lookup, though bytearrays are faster for dense data.
    # Given max_d2 ~ 8000, bytearrays are good.
    
    S = []
    S_set = set()
    used_dists = [set() for _ in range(num_points)]
    
    def is_valid_add(P):
        """Check if P can be added to S."""
        if P in S_set:
            return False
        
        # Check 1: Distances from P to all Q in S must be unique
        # And Check 2: Distance d must not be already used by Q
        seen_dists = set()
        for Q in S:
            d = int(dist2[P, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            if d in used_dists[Q]:
                return False
                
        return True

    def add_point(P):
        """Add P to S and update used_dists."""
        S.append(P)
        S_set.add(P)
        for Q in S:
            if Q != P:
                d = int(dist2[P, Q])
                used_dists[P].add(d)
                used_dists[Q].add(d)

    def remove_point(P):
        """Remove P from S and update used_dists."""
        S.remove(P)
        S_set.discard(P)
        for Q in S:
            d = int(dist2[P, Q])
            used_dists[P].discard(d)
            used_dists[Q].discard(d)
        used_dists[P].clear()

    def try_swap(P_out, P_in):
        """Try to swap P_out (in S) with P_in (not in S). Returns True if successful."""
        if P_out not in S_set or P_in in S_set:
            return False
            
        # Check if P_in can be added to S \ {P_out}
        # This is slightly different from is_valid_add(P_in) because P_out is removed.
        # We need to check:
        # 1. Distances from P_in to S\{P_out} are unique.
        # 2. For each Q in S\{P_out}, d(P_in, Q) is not in used_dists[Q] \ {d(P_out, Q)}.
        
        seen_dists = set()
        for Q in S:
            if Q == P_out:
                continue
            d = int(dist2[P_in, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # d(P_out, Q) is currently in used_dists[Q].
            # After swap, d(P_out, Q) is removed from used_dists[Q].
            # So if d == d(P_out, Q), it's fine (it will be freed).
            # If d != d(P_out, Q), it must not be in used_dists[Q].
            if d in used_dists[Q]:
                if d != int(dist2[P_out, Q]):
                    return False
                    
        # If valid, perform swap
        remove_point(P_out)
        add_point(P_in)
        return True

    def greedy_construct(order):
        """Build a set using greedy addition in given order."""
        nonlocal S, S_set, used_dists
        S.clear()
        S_set.clear()
        used_dists = [set() for _ in range(num_points)]
        
        for P in order:
            if is_valid_add(P):
                add_point(P)
        
        return len(S)

    # Generate different orderings
    rng = random.Random(42)
    
    # 1. Random
    order_rand = list(range(num_points))
    rng.shuffle(order_rand)
    
    # 2. Radial (Center out)
    cx, cy = N/2.0, N/2.0
    order_radial = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    
    # 3. Diagonal
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    
    # 4. Corner out (Far from center)
    order_corner = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2, reverse=True)

    orders = [order_rand, order_radial, order_diag, order_corner]
    
    best_size = 0
    best_S = []
    
    for order in orders:
        size = greedy_construct(order)
        if size > best_size:
            best_size = size
            best_S = S[:]
            # Save state
            best_used = [s.copy() for s in used_dists]
            best_S_set = S_set.copy()

    # Local Search / Simulated Annealing
    S = best_S[:]
    S_set = best_S_set.copy()
    used_dists = best_used
    
    current_size = len(S)
    best_size_sa = current_size
    best_S_sa = S[:]
    best_used_sa = [s.copy() for s in used_dists]
    best_S_set_sa = S_set.copy()
    
    # Parameters for local search
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline - 10:
        if not S:
            break
            
        # Random move selection
        move_type = rng.choice(['add', 'swap', 'remove'])
        
        if move_type == 'add':
            # Try to add a random point
            candidates = [P for P in rng.sample(range(num_points), 50) if P not in S_set]
            for P in candidates:
                if is_valid_add(P):
                    add_point(P)
                    if len(S) > best_size_sa:
                        best_size_sa = len(S)
                        best_S_sa = S[:]
                        best_used_sa = [s.copy() for s in used_dists]
                        best_S_set_sa = S_set.copy()
                    break
        elif move_type == 'swap':
            # Try to swap a random point in S with a random point outside
            P_out = rng.choice(S)
            P_in = rng.randint(0, num_points - 1)
            if P_in not in S_set:
                if try_swap(P_out, P_in):
                    if len(S) > best_size_sa:
                        best_size_sa = len(S)
                        best_S_sa = S[:]
                        best_used_sa = [s.copy() for s in used_dists]
                        best_S_set_sa = S_set.copy()
        elif move_type == 'remove':
            # Remove a random point to escape local minima
            P_out = rng.choice(S)
            remove_point(P_out)
            
        # Cooling
        temperature *= cooling
        
        # Occasionally reset to best found to avoid drifting too far
        if rng.random() < 0.01:
             # Restore best state
             S = best_S_sa[:]
             S_set = best_S_set_sa.copy()
             used_dists = [s.copy() for s in best_used_sa]

    # Final check
    if best_size_sa > best_size:
        best_size = best_size_sa
        best_S = best_S_sa

    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
