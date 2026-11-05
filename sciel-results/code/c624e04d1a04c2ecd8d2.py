"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin for 300s limit
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # State variables
    # S: list of indices in the current set
    # in_S: boolean array for O(1) lookup
    # used_dists: list of sets, where used_dists[i] contains squared distances
    #             from i to other points in S
    
    best_S = []
    best_size = 0
    
    def is_valid_add(P, S, in_S, used_dists):
        """
        Check if P can be added to S.
        Assumes P is not in S.
        """
        if in_S[P]:
            return False
            
        seen_dists = set()
        # Check distances from P to all Q in S
        for Q in S:
            d = int(dist2[P, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Check if this distance is already used by Q
            if d in used_dists[Q]:
                return False
                
        return True

    def add_point(P, S, in_S, used_dists):
        """Add P to S and update state."""
        S.append(P)
        in_S[P] = True
        for Q in S:
            if Q != P:
                d = int(dist2[P, Q])
                used_dists[P].add(d)
                used_dists[Q].add(d)

    def remove_point(P, S, in_S, used_dists):
        """Remove P from S and update state."""
        S.remove(P)
        in_S[P] = False
        for Q in S:
            d = int(dist2[P, Q])
            used_dists[P].discard(d)
            used_dists[Q].discard(d)
        used_dists[P].clear()

    def try_swap(P_out, P_in, S, in_S, used_dists):
        """
        Try to swap P_out (in S) with P_in (not in S).
        Returns True if successful.
        """
        if not in_S[P_out] or in_S[P_in]:
            return False
            
        # Temporarily remove P_out to check if P_in can be added to S \ {P_out}
        # We don't actually remove it yet, we just simulate the check.
        # The set of points to check against is S \ {P_out}.
        
        seen_dists = set()
        for Q in S:
            if Q == P_out:
                continue
            d = int(dist2[P_in, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # d(P_out, Q) is currently in used_dists[Q].
            # After swap, d(P_out, Q) is removed.
            # So if d == d(P_out, Q), it's fine.
            # Otherwise, d must not be in used_dists[Q].
            if d in used_dists[Q]:
                if d != int(dist2[P_out, Q]):
                    return False
                    
        # If valid, perform swap
        remove_point(P_out, S, in_S, used_dists)
        add_point(P_in, S, in_S, used_dists)
        return True

    def greedy_construct(order, S, in_S, used_dists):
        """Build a set using greedy addition in given order."""
        S.clear()
        in_S.fill(False)
        for i in range(num_points):
            used_dists[i].clear()
            
        for P in order:
            if is_valid_add(P, S, in_S, used_dists):
                add_point(P, S, in_S, used_dists)
        
        return len(S)

    # Initialize state containers
    S = []
    in_S = np.zeros(num_points, dtype=bool)
    used_dists = [set() for _ in range(num_points)]
    
    rng = random.Random(42)
    
    # Phase 1: Multi-Start Greedy
    # Generate diverse orderings
    orders = []
    
    # 1. Random
    o1 = list(range(num_points))
    rng.shuffle(o1)
    orders.append(o1)
    
    # 2. Radial (Center out)
    cx, cy = N/2.0, N/2.0
    o2 = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    orders.append(o2)
    
    # 3. Diagonal
    o3 = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    orders.append(o3)
    
    # 4. Corner out
    o4 = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2, reverse=True)
    orders.append(o4)
    
    # 5. Random with bias towards corners?
    o5 = list(range(num_points))
    rng.shuffle(o5)
    orders.append(o5)
    
    for order in orders:
        size = greedy_construct(order, S, in_S, used_dists)
        if size > best_size:
            best_size = size
            best_S = S[:]
            best_in_S = in_S.copy()
            best_used = [s.copy() for s in used_dists]

    # Phase 2: Simulated Annealing / Local Search
    # Start from the best greedy solution
    S = best_S[:]
    in_S = best_in_S.copy()
    used_dists = best_used
    
    current_size = len(S)
    best_size_sa = current_size
    best_S_sa = S[:]
    best_in_S_sa = in_S.copy()
    best_used_sa = [s.copy() for s in used_dists]
    
    temperature = 1.0
    cooling = 0.995
    min_temperature = 0.01
    
    while time.time() < deadline - 5:
        if not S:
            break
            
        # Random move selection
        # Prioritize adds if possible, but we need to find them.
        # We'll just pick a random move type.
        move_type = rng.choice(['add', 'swap', 'remove'])
        
        if move_type == 'add':
            # Try to add a random point
            # To make this efficient, we might want to check a batch of candidates
            # But simple random sampling is okay for now
            candidates = rng.sample(range(num_points), 100)
            added = False
            for P in candidates:
                if not in_S[P] and is_valid_add(P, S, in_S, used_dists):
                    add_point(P, S, in_S, used_dists)
                    added = True
                    if len(S) > best_size_sa:
                        best_size_sa = len(S)
                        best_S_sa = S[:]
                        best_in_S_sa = in_S.copy()
                        best_used_sa = [s.copy() for s in used_dists]
                    break
        elif move_type == 'swap':
            # Try to swap a random point in S with a random point outside
            if S:
                P_out = rng.choice(S)
                # Sample P_in
                P_in = rng.randint(0, num_points - 1)
                if not in_S[P_in]:
                    if try_swap(P_out, P_in, S, in_S, used_dists):
                        # Size is unchanged, but configuration changed
                        # We don't update best_size_sa unless size increases (which it doesn't here)
                        # But we might want to keep track of best configuration?
                        # No, we only care about size.
                        pass
        elif move_type == 'remove':
            # Remove a random point to escape local minima
            if S:
                P_out = rng.choice(S)
                remove_point(P_out, S, in_S, used_dists)
                
        # Cooling
        if temperature > min_temperature:
            temperature *= cooling
            
        # Occasionally reset to best found to avoid drifting too far into bad local minima
        if rng.random() < 0.005:
             S = best_S_sa[:]
             in_S = best_in_S_sa.copy()
             used_dists = best_used_sa

    # Final check
    if best_size_sa > best_size:
        best_size = best_size_sa
        best_S = best_S_sa

    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
