import random
import time

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]
ALL_MASK = (1 << NUM_POINTS) - 1

def precompute():
    # iso_mask[i][j] is a bitmask of points k such that {i, j, k} is isosceles
    iso_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(i + 1, NUM_POINTS):
            xj, yj = POINTS[j]
            d_ij2 = (xi - xj) ** 2 + (yi - yj) ** 2
            mask = 0
            for k in range(NUM_POINTS):
                if k == i or k == j:
                    continue
                xk, yk = POINTS[k]
                d_ik2 = (xi - xk) ** 2 + (yi - yk) ** 2
                d_jk2 = (xj - xk) ** 2 + (yj - yk) ** 2
                if d_ik2 == d_jk2 or d_ik2 == d_ij2 or d_jk2 == d_ij2:
                    mask |= (1 << k)
            iso_mask[i][j] = mask
            iso_mask[j][i] = mask
    return iso_mask

ISO_MASK = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = float('inf')
    best_set = None
    
    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        
        S = set()
        S_mask = 0
        block_count = [0] * NUM_POINTS
        blocked_mask = 0
        
        # Candidates: points that can be added (not in S, not blocked)
        # Initially all points are candidates
        candidates = set(range(NUM_POINTS))
        
        # Greedy Construction
        while candidates:
            # Check if we are already maximal
            # Maximal if all points not in S are blocked
            # i.e., (blocked_mask | S_mask) == ALL_MASK
            if (blocked_mask | S_mask) == ALL_MASK:
                break
            
            # Pick a candidate
            # Heuristic: pick one that blocks the most new points
            best_cand = None
            best_score = -1
            
            # To speed up, we can sample or just iterate if candidates is small
            # Candidates size decreases as we add points
            cand_list = list(candidates)
            # Randomize to avoid bias
            random.shuffle(cand_list)
            
            for p in cand_list:
                # Calculate score: how many NEW points does p block?
                # p blocks points in OR_{q in S} ISO_MASK[p][q]
                # New points are those not already in blocked_mask
                
                new_blocked = 0
                for q in S:
                    new_blocked |= ISO_MASK[p][q]
                
                # Count bits in (new_blocked & ~blocked_mask)
                score = bin(new_blocked & ~blocked_mask).count('1')
                
                if score > best_score:
                    best_score = score
                    best_cand = p
                    if best_score > 50: # Early exit if score is high
                        break
            
            if best_cand is None:
                # Should not happen if candidates is not empty
                break
                
            p = best_cand
            
            # Add p to S
            S.add(p)
            S_mask |= (1 << p)
            candidates.remove(p)
            
            # Update block counts and blocked_mask
            # For each q in S (excluding p, but S already has p, so iterate S \ {p})
            # Actually, we need to update for pairs (p, q) where q is in OLD S
            # Let's iterate over old_S
            # We can just iterate over S and skip p? No, S is updated.
            # Let's store old_S before adding? Or just iterate S and skip p.
            
            for q in S:
                if q == p:
                    continue
                mask = ISO_MASK[p][q]
                # Iterate bits in mask
                m = mask
                while m:
                    lsb = m & -m
                    x_idx = lsb.bit_length() - 1
                    block_count[x_idx] += 1
                    if block_count[x_idx] == 1:
                        blocked_mask |= lsb
                    m ^= lsb
            
            # Remove newly blocked points from candidates
            # We need to know which points became newly blocked
            # We can recompute or track it.
            # Simpler: just filter candidates
            # candidates = {c for c in candidates if (blocked_mask >> c) & 1 == 0}
            # This is O(|candidates|).
            
            # More efficient:
            # We know the new points are in (new_blocked & ~old_blocked_mask)
            # Let's just iterate candidates and remove if blocked
            # Since candidates shrinks, this is fine.
            
            new_candidates = set()
            for c in candidates:
                if (blocked_mask >> c) & 1:
                    continue
                new_candidates.add(c)
            candidates = new_candidates

        # Now S is maximal.
        # Reduction Phase
        improved = True
        while improved:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p in s_list:
                if p not in S:
                    continue
                
                # Simulate removal of p
                # Decrement block_count for x in OR_{q in S \ {p}} ISO_MASK[p][q]
                changes = []
                for q in S:
                    if q == p:
                        continue
                    mask = ISO_MASK[p][q]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        block_count[x_idx] -= 1
                        changes.append(x_idx)
                        m ^= lsb
                
                # Check if S \ {p} is maximal
                # 1. p must be blocked by S \ {p}
                if block_count[p] == 0:
                    # Revert
                    for x_idx in changes:
                        block_count[x_idx] += 1
                    continue
                
                # 2. All x not in S \ {p} must be blocked
                # x not in S \ {p} means x not in S OR x == p
                valid = True
                for x_idx in range(NUM_POINTS):
                    if x_idx in S and x_idx != p:
                        continue
                    if block_count[x_idx] == 0:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    for x_idx in changes:
                        block_count[x_idx] += 1
                    continue
                
                # Valid removal
                S.remove(p)
                S_mask &= ~(1 << p)
                # Update blocked_mask
                # We need to clear bits where count became 0
                # We can just recalculate blocked_mask from block_count?
                # Or update it.
                # Since we only decrement, bits might turn off.
                # Let's just recalculate blocked_mask for simplicity in this step?
                # Or iterate changes and check if count is 0.
                for x_idx in changes:
                    if block_count[x_idx] == 0:
                        blocked_mask &= ~(1 << x_idx)
                
                improved = True
                break
        
        return len(S), S

    restarts = 0
    while time.time() < deadline:
        size, s = run_greedy_reduce(restarts)
        if size < best_size:
            best_size = size
            best_set = s
        restarts += 1
        
        # If we find a very small set, we might want to break early?
        # Theoretical min is likely around 10-15 for 16x16.
        # Let's just keep running.

    if best_set is None:
        # Fallback
        S = set(range(NUM_POINTS))
        best_set = S

    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
