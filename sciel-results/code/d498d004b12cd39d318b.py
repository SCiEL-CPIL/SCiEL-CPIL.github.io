"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_valid_addition(P, S, D_dict):
        """Check if P can be added to S."""
        if not S:
            return True
        dists_from_P = set()
        for Q in S:
            d = sq_dist(P, Q)
            if d in dists_from_P:
                return False
            dists_from_P.add(d)
            if d in D_dict[Q]:
                return False
        return True

    def is_valid_swap(P, Q, S, D_dict):
        """
        Check if swapping Q for P keeps the set valid.
        S' = S \ {Q} U {P}
        """
        # If P is already in S, it's not a swap, it's a no-op or invalid
        if P in D_dict:
            return False
            
        # We need to check if P is valid in S \ {Q}
        # Conditions:
        # 1. Distances from P to R in S \ {Q} are unique.
        # 2. For each R in S \ {Q}, dist(P, R) is not in D'_R.
        #    D'_R = D_R \ {dist(R, Q)}
        
        dists_from_P = set()
        for R in S:
            if R == Q:
                continue
            d_PR = sq_dist(P, R)
            
            # Check uniqueness among P to others in S'
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            # Check if d_PR is in D'_R
            # D'_R contains d_PR if d_PR in D_R AND d_PR != dist(R, Q)
            if d_PR in D_dict[R]:
                d_RQ = sq_dist(R, Q)
                if d_PR != d_RQ:
                    return False
                # If d_PR == d_RQ, then d_PR was removed from D_R to form D'_R, 
                # so it is NOT in D'_R. Thus valid.
            # If d_PR not in D_R, it is not in D'_R. Valid.
            
        return True

    def add_point(P, S, D_dict, S_set, complement):
        """Add P to S."""
        S.append(P)
        S_set.add(P)
        complement.remove(P)
        
        dists_from_P = set()
        for Q in S:
            if Q == P:
                continue
            d = sq_dist(P, Q)
            dists_from_P.add(d)
            D_dict[Q].add(d)
        D_dict[P] = dists_from_P

    def remove_point(P, S, D_dict, S_set, complement):
        """Remove P from S."""
        S.remove(P)
        S_set.discard(P)
        complement.append(P)
        
        dists_from_P = D_dict[P]
        for Q in S:
            d = sq_dist(P, Q)
            D_dict[Q].discard(d)
        del D_dict[P]

    def swap_point(P, Q, S, D_dict, S_set, complement):
        """Swap Q with P."""
        # Remove Q
        remove_point(Q, S, D_dict, S_set, complement)
        # Add P
        add_point(P, S, D_dict, S_set, complement)

    # Main Search
    best_S = []
    best_D = {}
    best_S_set = set()
    best_complement = points[:]
    
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    # Try to build a large set quickly using a heuristic
    # Sort points by distance to center, or random
    S = []
    D_dict = {}
    S_set = set()
    complement = points[:]
    
    # Heuristic: Add points that are far from existing ones
    # We'll just do random greedy first, then refine
    for _ in range(100): # Try 100 times to get a good start? No, just one good run
        break
        
    # Better init: Random shuffle, greedy add
    comp_copy = complement[:]
    rng.shuffle(comp_copy)
    for P in comp_copy:
        if is_valid_addition(P, S, D_dict):
            add_point(P, S, D_dict, S_set, complement)
            
    best_S = S[:]
    best_D = {k: v.copy() for k, v in D_dict.items()}
    best_S_set = S_set.copy()
    best_complement = complement[:]
    
    # Phase 2: Local Search with Swaps
    # We operate on the current best state
    S = best_S[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 10:
        if not complement:
            break
            
        # Choose action: Add (50%) or Swap (50%)
        # If S is small, bias towards Add
        if len(S) < 15 and rng.random() < 0.7:
            action = 'add'
        else:
            action = 'swap' if rng.random() < 0.5 else 'add'
            
        if action == 'add':
            # Pick random candidate
            P = rng.choice(complement)
            if is_valid_addition(P, S, D_dict):
                add_point(P, S, D_dict, S_set, complement)
                # Update best if improved
                if len(S) > len(best_S):
                    best_S = S[:]
                    best_D = {k: v.copy() for k, v in D_dict.items()}
                    best_S_set = S_set.copy()
                    best_complement = complement[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Swap
            P = rng.choice(complement)
            Q = rng.choice(S)
            if is_valid_swap(P, Q, S, D_dict):
                swap_point(P, Q, S, D_dict, S_set, complement)
                # Swap keeps size same, so no size improvement.
                # But it might allow future additions.
                # We don't update 'best' based on size for swaps, 
                # but we keep the state.
                # To prevent cycling, we might want to track visited states? 
                # For now, just let it run.
                
                # Check if we can now add a point?
                # Sometimes a swap opens up space.
                # Let's try to add a point after a swap with some probability
                if rng.random() < 0.2 and complement:
                    P_add = rng.choice(complement)
                    if is_valid_addition(P_add, S, D_dict):
                        add_point(P_add, S, D_dict, S_set, complement)
                        if len(S) > len(best_S):
                            best_S = S[:]
                            best_D = {k: v.copy() for k, v in D_dict.items()}
                            best_S_set = S_set.copy()
                            best_complement = complement[:]
                            no_improve_count = 0
            else:
                no_improve_count += 1
                
        # Reset state if stuck for too long?
        # If no_improve_count > max_no_improve, maybe restart from best_S?
        if no_improve_count > max_no_improve:
            # Revert to best
            S = best_S[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            # Perturb best state slightly to escape local optimum
            # Remove 2 random points
            if len(S) > 2:
                idx1 = rng.randrange(len(S))
                idx2 = rng.randrange(len(S))
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(S)
                P1 = S[idx1]
                P2 = S[idx2]
                # Remove them
                remove_point(P1, S, D_dict, S_set, complement)
                remove_point(P2, S, D_dict, S_set, complement)
                
    return best_S
# EVOLVE-BLOCK-END
