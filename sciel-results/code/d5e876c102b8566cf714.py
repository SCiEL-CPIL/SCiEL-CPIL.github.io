import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix D[i,j] = squared distance
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)

    def can_add(p, S, D_S):
        """
        Check if point p can be added to set S.
        S: list of indices
        D_S: set of squared distances between all pairs in S
        """
        if not S:
            return True
        
        # Get distances from p to all points in S
        d_p_S = D[p, S]
        
        # Condition 1: All distances from p to S must be distinct
        # If there is a duplicate, p forms an isosceles triangle with the two points having the same distance
        if len(np.unique(d_p_S)) != len(S):
            return False
            
        # Condition 2: No distance d(p, s) should be in D_S
        # If d(p, s) == d(u, v) for some u, v in S, then (p, s, u) or (p, s, v) is isosceles?
        # Wait. If d(p, s) = d(u, v), does that form an isosceles triangle?
        # The triangle is (p, s, u) or (p, s, v)?
        # If d(p, s) = d(u, v), the sides of triangle (p, s, u) are d(p,s), d(s,u), d(u,p).
        # We don't know if d(p,s) equals d(s,u) or d(u,p).
        # The condition for NO isosceles triangle among ANY three points in S U {p} is:
        # For any distinct a, b, c in S U {p}, d(a,b) != d(a,c) != d(b,c) != d(a,b).
        
        # Let's re-verify the logic.
        # If we add p, new triples are (p, s1, s2) for s1, s2 in S.
        # Triangle (p, s1, s2) is isosceles if:
        # 1. d(p, s1) == d(p, s2)  -> Covered by Condition 1 (distinct d_p_S)
        # 2. d(p, s1) == d(s1, s2) -> d(p, s1) is in D_S (since d(s1, s2) is in D_S)
        # 3. d(p, s2) == d(s1, s2) -> d(p, s2) is in D_S
        
        # So yes, if any d(p, s) is in D_S, it creates an isosceles triangle.
        
        # Check if any value in d_p_S is in D_S
        # Since D_S is a set, we can do a quick check
        for d in d_p_S:
            if d in D_S:
                return False
                
        return True

    def add_point(p, S, D_S):
        """Add point p to S and update D_S."""
        S.append(p)
        # Update D_S with distances from p to all other points in S
        # Note: S already contains p at the end
        for s in S:
            if s == p:
                continue
            D_S.add(D[p, s])

    def remove_point(p, S, D_S):
        """Remove point p from S and update D_S."""
        S.remove(p)
        # We need to remove distances involving p from D_S.
        # But D_S is a set of values, not counts. If another pair has the same distance, we can't just remove it.
        # So we need to maintain counts of distances in D_S.
        pass

    # Since remove_point is tricky with a simple set, let's use a Counter for D_S
    # Or, simpler: just rebuild D_S when we need to remove? 
    # Rebuilding D_S is O(|S|^2). If |S| is ~10-20, this is fast.
    # But in ILS, we do many removals/additions.
    # Let's use a Counter for D_S to allow efficient removals.

    from collections import Counter
    
    def can_add_counter(p, S, D_S_counter):
        if not S:
            return True
        
        d_p_S = D[p, S]
        
        # Condition 1
        if len(np.unique(d_p_S)) != len(S):
            return False
            
        # Condition 2
        for d in d_p_S:
            if D_S_counter[d] > 0:
                return False
                
        return True

    def add_point_counter(p, S, D_S_counter):
        S.append(p)
        for s in S:
            if s == p:
                continue
            d_val = D[p, s]
            D_S_counter[d_val] += 1

    def remove_point_counter(p, S, D_S_counter):
        S.remove(p)
        for s in S:
            d_val = D[p, s]
            D_S_counter[d_val] -= 1
            if D_S_counter[d_val] == 0:
                del D_S_counter[d_val]

    # Phase 1: Greedy
    best_S = []
    best_D_S = Counter()
    best_size = 0

    for restart in range(100):
        if time.time() > deadline - 60:
            break

        S = []
        D_S_counter = Counter()
        candidates = list(range(M))
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline - 60:
                break
            if can_add_counter(p, S, D_S_counter):
                add_point_counter(p, S, D_S_counter)

        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_D_S = D_S_counter.copy()

    # Phase 2: Iterated Local Search
    current_S = best_S.copy()
    current_D_S = best_D_S.copy()
    
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break

        if not current_S:
            break

        # Perturbation
        k = random.randint(1, min(5, len(current_S)))
        to_remove = random.sample(current_S, k)
        for p in to_remove:
            remove_point_counter(p, current_S, current_D_S)

        # Greedy reconstruction
        S_set = set(current_S)
        candidates = [p for p in range(M) if p not in S_set]
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline - 10:
                break
            if can_add_counter(p, current_S, current_D_S):
                add_point_counter(p, current_S, current_D_S)

        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = current_S.copy()
            best_D_S = current_D_S.copy()
        else:
            # Revert to best
            current_S = best_S.copy()
            current_D_S = best_D_S.copy()

        # Local Swap Search
        # Try to swap one point in current_S with one point not in current_S
        if len(current_S) > 0:
            p_out = random.choice(current_S)
            remove_point_counter(p_out, current_S, current_D_S)
            
            S_set = set(current_S)
            candidates = [p for p in range(M) if p not in S_set and p != p_out]
            random.shuffle(candidates)
            
            improved = False
            for p_in in candidates[:100]: # Limit search for swap
                if time.time() > deadline - 10:
                    break
                if can_add_counter(p_in, current_S, current_D_S):
                    add_point_counter(p_in, current_S, current_D_S)
                    improved = True
                    break
            
            if improved and len(current_S) > best_size:
                best_size = len(current_S)
                best_S = current_S.copy()
                best_D_S = current_D_S.copy()
            else:
                # Revert swap
                if improved:
                    remove_point_counter(p_in, current_S, current_D_S)
                    add_point_counter(p_out, current_S, current_D_S)
                else:
                    add_point_counter(p_out, current_S, current_D_S)
                current_S = best_S.copy()
                current_D_S = best_D_S.copy()

    return [tuple(points[i]) for i in best_S]
