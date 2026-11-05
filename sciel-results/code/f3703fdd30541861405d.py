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
    
    # used_dists[i][d] = 1 if distance d is used by point i
    # This is a list of bytearrays for memory efficiency and fast access
    used_dists = [bytearray(max_d2 + 1) for _ in range(num_points)]
    
    S = []
    S_set = set()
    
    def is_valid_add(P):
        """Check if P can be added to S."""
        if P in S_set:
            return False
        
        # Check 1: Distances from P to all Q in S must be unique
        seen_dists = set()
        for Q in S:
            d = int(dist2[P, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Check 2: Distance d must not be already used by Q
            if used_dists[Q][d]:
                return False
                
        return True

    def add_point(P):
        """Add P to S and update used_dists."""
        S.append(P)
        S_set.add(P)
        for Q in S:
            if Q != P:
                d = int(dist2[P, Q])
                used_dists[P][d] = 1
                used_dists[Q][d] = 1

    def remove_point(P):
        """Remove P from S and update used_dists."""
        S.remove(P)
        S_set.discard(P)
        for Q in S:
            d = int(dist2[P, Q])
            used_dists[P][d] = 0
            used_dists[Q][d] = 0
        # Clear P's used dists (optional, but good for cleanliness)
        used_dists[P] = bytearray(max_d2 + 1)

    def try_swap(P_out, P_in):
        """Try to swap P_out (in S) with P_in (not in S). Returns True if successful."""
        if P_out not in S_set or P_in in S_set:
            return False
            
        # Temporarily remove P_out
        # We need to check if P_in can be added to S \ {P_out}
        
        # 1. Check if distances from P_in to S\{P_out} are unique
        seen_dists = set()
        valid = True
        for Q in S:
            if Q == P_out:
                continue
            d = int(dist2[P_in, Q])
            if d in seen_dists:
                valid = False
                break
            seen_dists.add(d)
            
            # 2. Check if d is used by Q in S\{P_out}
            # Note: used_dists[Q] currently includes d(Q, P_out).
            # If d == d(Q, P_out), it's "free" after removal.
            # If d != d(Q, P_out), it must be 0 in used_dists[Q].
            if used_dists[Q][d]:
                # If the used distance is the one from P_out, it will be freed.
                # But we are checking if P_in can be added.
                # The condition is: for all R in S\{P_out}, d(P_in, R) is not in UsedDists[R] \ {d(R, P_out)}
                # So if used_dists[Q][d] is 1, we must check if that 1 comes from P_out.
                # If d == d(Q, P_out), then it's fine (it will be 0 after removal).
                # If d != d(Q, P_out), then it's a conflict.
                if d != int(dist2[P_out, Q]):
                    valid = False
                    break
                    
        if not valid:
            return False
            
        # Perform the swap
        remove_point(P_out)
        add_point(P_in)
        return True

    # Phase 1: Greedy Construction
    # Try to add points in a random order
    rng = random.Random(42)
    order = list(range(num_points))
    rng.shuffle(order)
    
    for P in order:
        if is_valid_add(P):
            add_point(P)
    
    best_size = len(S)
    best_S = S[:]
    
    # Phase 2: Local Search with Swaps
    # We try to improve the set by swapping points.
    # A swap doesn't change size, but it might allow further additions.
    
    def local_search():
        nonlocal S, S_set, used_dists, best_size, best_S
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 10:
            # Try to add any point
            added = False
            # Randomly sample some candidates
            candidates = [P for P in rng.sample(range(num_points), 100) if P not in S_set]
            for P in candidates:
                if is_valid_add(P):
                    add_point(P)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                    no_improve_count = 0
                    break
            
            if not added:
                # Try a swap
                # Pick a random P_out from S and P_in from complement
                if S:
                    P_out = rng.choice(S)
                    # Find a P_in
                    # Try a few random P_in
                    for _ in range(10):
                        P_in = rng.randint(0, num_points - 1)
                        if P_in in S_set:
                            continue
                        if try_swap(P_out, P_in):
                            break
                    
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Restart or break
                    break
        
    local_search()
    
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
