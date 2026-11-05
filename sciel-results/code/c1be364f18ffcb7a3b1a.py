"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    M = len(all_points)
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Precompute squared distances
    # dist[i][j]
    dist = [[0]*M for _ in range(M)]
    for i in range(M):
        for j in range(i+1, M):
            dx = all_points[i][0] - all_points[j][0]
            dy = all_points[i][1] - all_points[j][1]
            d = dx*dx + dy*dy
            dist[i][j] = d
            dist[j][i] = d
            
    # Precompute conflict masks
    # conflict[i][j] is a bitmask of points k such that (i, j, k) is isosceles
    conflict = [[0]*M for _ in range(M)]
    for i in range(M):
        for j in range(i+1, M):
            mask = 0
            d_ij = dist[i][j]
            for k in range(M):
                if k == i or k == j:
                    continue
                d_ik = dist[i][k]
                d_jk = dist[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    mask |= (1 << k)
            conflict[i][j] = mask
            conflict[j][i] = mask
            
    mask_all = (1 << M) - 1
    
    def get_blocked_mask(S_indices):
        """Calculate the bitmask of all points blocked by the set S_indices."""
        mask = 0
        n = len(S_indices)
        for i in range(n):
            a = S_indices[i]
            for j in range(i+1, n):
                b = S_indices[j]
                mask |= conflict[a][b]
        return mask

    def is_maximal(S_indices, mask_S, mask_blocked):
        """Check if S is maximal."""
        # Points not in S
        mask_not_S = mask_all ^ mask_S
        # Maximal if all points not in S are blocked
        return (mask_not_S & ~mask_blocked) == 0

    def greedy_construct():
        """Greedy construction with max-blocking heuristic."""
        S = []
        mask_S = 0
        mask_blocked = 0
        
        for _ in range(M):
            # Find addable points
            addable = []
            # Points not in S and not blocked
            candidates_mask = mask_all ^ mask_S ^ mask_blocked
            # Iterate through set bits of candidates_mask
            # Since M=256, we can just iterate 0..M-1
            for p in range(M):
                if (candidates_mask >> p) & 1:
                    # Check if p is actually addable (it is by definition of candidates_mask)
                    # Calculate how many new points it blocks
                    new_block = 0
                    for q in S:
                        new_block |= conflict[p][q]
                    # Count new bits
                    # new_bits = new_block & ~mask_blocked
                    # count = bin(new_bits).count('1')
                    # We want to maximize this
                    # Store (count, p)
                    new_bits = new_block & ~mask_blocked
                    cnt = bin(new_bits).count('1')
                    addable.append((cnt, p))
            
            if not addable:
                break
                
            # Pick best
            # Tie-break randomly
            max_cnt = max(c for c, _ in addable)
            best_candidates = [p for c, p in addable if c == max_cnt]
            p = random.choice(best_candidates)
            
            S.append(p)
            mask_S |= (1 << p)
            # Update mask_blocked
            for q in S[:-1]: # q is the ones added before p
                mask_blocked |= conflict[p][q]
                
        return S, mask_S, mask_blocked

    def minimize_deletion(S, mask_S, mask_blocked):
        """Remove points if set remains maximal."""
        changed = True
        while changed:
            changed = False
            # Try removing each point
            # Randomize order
            indices = list(range(len(S)))
            random.shuffle(indices)
            for idx in indices:
                p = S[idx]
                # Remove p
                S_new = S[:idx] + S[idx+1:]
                mask_S_new = mask_S ^ (1 << p)
                
                # Recalculate blocked mask for S_new
                mask_blocked_new = get_blocked_mask(S_new)
                
                # Check maximality
                if is_maximal(S_new, mask_S_new, mask_blocked_new):
                    S = S_new
                    mask_S = mask_S_new
                    mask_blocked = mask_blocked_new
                    changed = True
                    break # Restart loop
                    
        return S, mask_S, mask_blocked

    def local_search_swap(S, mask_S, mask_blocked, deadline):
        """Try swaps to find smaller maximal sets."""
        current_S = S
        current_mask_S = mask_S
        current_mask_blocked = mask_blocked
        current_size = len(S)
        
        while time.time() < deadline:
            improved = False
            indices = list(range(len(current_S)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                u = current_S[idx]
                
                # Calculate blocked mask without u
                S_wo_u = current_S[:idx] + current_S[idx+1:]
                mask_blocked_wo_u = get_blocked_mask(S_wo_u)
                
                # Candidates for v: not in S, not blocked by S_wo_u
                # Note: v must not be in S
                mask_not_S = mask_all ^ current_mask_S
                candidates_mask = mask_not_S & ~mask_blocked_wo_u
                
                # Pick a few random candidates
                # To save time, pick up to 5 random candidates
                candidates = []
                temp_mask = candidates_mask
                while temp_mask and len(candidates) < 5:
                    # Pick a random bit
                    # This is slow in Python. Just iterate if M is small?
                    # M=256. Iteration is fine.
                    pass
                
                # Efficient way to pick random set bits:
                # Just iterate all and collect, then sample
                cand_list = []
                for v in range(M):
                    if (candidates_mask >> v) & 1:
                        cand_list.append(v)
                
                if not cand_list:
                    continue
                    
                random.shuffle(cand_list)
                
                for v in cand_list[:5]:
                    # Try swap u -> v
                    S_new = S_wo_u + [v]
                    mask_S_new = (current_mask_S ^ (1 << u)) | (1 << v)
                    
                    # Check independence: v not in mask_blocked_wo_u (already ensured)
                    # Check maximality
                    mask_blocked_new = get_blocked_mask(S_new)
                    
                    if is_maximal(S_new, mask_S_new, mask_blocked_new):
                        # Try to minimize
                        S_min, mask_S_min, mask_blocked_min = minimize_deletion(S_new, mask_S_new, mask_blocked_new)
                        
                        if len(S_min) < current_size:
                            current_S = S_min
                            current_mask_S = mask_S_min
                            current_mask_blocked = mask_blocked_min
                            current_size = len(S_min)
                            improved = True
                            break # Restart outer loop? Or continue?
                            # Break inner, continue outer with new S
                        # If not improved, revert (we don't change current_S)
                
                if improved:
                    break
                    
            if not improved:
                # No improvement in this pass
                # Maybe try a different heuristic or break?
                # To avoid infinite loop, we can break or shuffle more aggressively
                # Let's break to try a new greedy restart if we are stuck
                break
                
        return current_S

    start_time = time.time()
    deadline = start_time + 290 # Safety margin
    
    best_S = None
    best_size = M + 1
    
    # Initial greedy
    S, mask_S, mask_blocked = greedy_construct()
    S, mask_S, mask_blocked = minimize_deletion(S, mask_S, mask_blocked)
    if len(S) < best_size:
        best_size = len(S)
        best_S = S[:]
        
    # Main search loop
    while time.time() < deadline:
        # Randomized Greedy
        S, mask_S, mask_blocked = greedy_construct()
        S, mask_S, mask_blocked = minimize_deletion(S, mask_S, mask_blocked)
        
        # Local Search
        S_ls = local_search_swap(S, mask_S, mask_blocked, deadline)
        
        if len(S_ls) < best_size:
            best_size = len(S_ls)
            best_S = S_ls[:]
            
        # If we found a very small set, we might want to spend more time on it?
        # The loop structure handles this by continuing to search.
        
    # Return best_S as list of (x,y)
    return [all_points[i] for i in best_S]

# EVOLVE-BLOCK-END
