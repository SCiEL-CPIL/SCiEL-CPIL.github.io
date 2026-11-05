# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

N = 16
N2 = N * N

# Precompute point coordinates and index mapping
points = list(itertools.product(range(N), repeat=2))
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

def _is_iso_idx(i, j, k):
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute iso_with[i][j] as a bitset of k such that i, j, k form an isosceles triangle
iso_with = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        mask = 0
        for k in range(N2):
            if k == i or k == j:
                continue
            if _is_iso_idx(i, j, k):
                mask |= (1 << k)
        iso_with[i][j] = mask

def is_maximal(S_list):
    """
    S_list: list of indices in the set
    Returns True if S_list is a maximal no-isosceles set
    """
    S_bits = 0
    for idx in S_list:
        S_bits |= (1 << idx)
    
    # Check each point not in S
    for q in range(N2):
        if (S_bits >> q) & 1:
            continue
        
        blocked = False
        for a in S_list:
            if (iso_with[q][a] & S_bits):
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_size = float('inf')
    best_set = []
    
    def greedy_blocking():
        """
        Greedy construction that prioritizes points that block the most 'addable' points.
        """
        S = []
        S_bits = 0
        # Initially all points are addable
        addable = set(range(N2))
        
        while addable:
            best_p = -1
            best_score = -1
            
            # Heuristic: pick p that blocks the most other addable points
            # To speed up, we can sample or just iterate if N2 is small enough (256 is small)
            for p in addable:
                # p is in addable, so it's valid to add to S
                # Calculate how many q in addable (q != p) would become blocked
                score = 0
                for q in addable:
                    if q == p:
                        continue
                    # q becomes blocked if there exists u in S such that {p, q, u} is iso
                    # This is equivalent to: (iso_with[p][q] & S_bits) != 0
                    if (iso_with[p][q] & S_bits) != 0:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p == -1:
                # Should not happen if addable is not empty
                break
            
            # Add best_p to S
            S.append(best_p)
            S_bits |= (1 << best_p)
            addable.remove(best_p)
            
            # Update addable: remove any q that is now blocked by best_p
            to_remove = []
            for q in addable:
                if (iso_with[best_p][q] & S_bits) != 0:
                    to_remove.append(q)
            
            for q in to_remove:
                addable.remove(q)
        
        return S

    def shrink(S_list):
        """Try removing single points while maintaining maximality."""
        S_list = list(S_list)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            indices = list(range(len(S_list)))
            random.shuffle(indices)
            for idx in indices:
                p = S_list[idx]
                candidate = S_list[:idx] + S_list[idx + 1:]
                if is_maximal(candidate):
                    S_list = candidate
                    changed = True
                    break
        return S_list

    def shrink_pairs(S_list):
        """Try removing pairs of points to escape local minima."""
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n = len(S_list)
            if n < 2:
                break
            for _ in range(min(30, n * (n - 1) // 2)):
                i, j = random.sample(range(n), 2)
                candidate = [S_list[k] for k in range(n) if k != i and k != j]
                if is_maximal(candidate):
                    S_list = candidate
                    improved = True
                    break
        return S_list

    def local_search(S_list):
        """Combined local search: shrink, shrink_pairs, and swaps."""
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try to remove single points
            new_S = shrink(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try to remove pairs
            new_S = shrink_pairs(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try swaps
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            if len(S_list) == 0:
                break
                
            p = random.choice(S_list)
            q = random.randint(0, N2 - 1)
            while (S_bits >> q) & 1:
                q = random.randint(0, N2 - 1)
                
            S_minus_p_bits = S_bits ^ (1 << p)
            S_minus_p_list = [x for x in S_list if x != p]
            
            valid_swap = True
            for a in S_minus_p_list:
                if (iso_with[q][a] & S_minus_p_bits):
                    valid_swap = False
                    break
                    
            if valid_swap:
                new_S = S_minus_p_list + [q]
                new_S = shrink(new_S)
                new_S = shrink_pairs(new_S)
                if len(new_S) < len(S_list):
                    S_list = new_S
                    improved = True
                    break
        return S_list

    # Run the new greedy blocking strategy
    S = greedy_blocking()
    S = local_search(S)
    
    best_size = len(S)
    best_set = S[:]
    
    # Try a few random restarts with the old greedy to see if it finds something better
    # Or just stick to the new one if it's fast
    # The new greedy is O(N^3) roughly? 256^3 is ~16M, which is fast in Python if optimized, but the inner loop is slow.
    # Let's optimize the greedy blocking inner loop.
    
    # Actually, let's just run the new strategy a few times with slight randomization in tie-breaking
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Randomized greedy blocking
        S = []
        S_bits = 0
        addable = set(range(N2))
        
        while addable:
            best_p = -1
            best_score = -1
            
            # Add some randomness
            candidates = list(addable)
            random.shuffle(candidates)
            
            for p in candidates:
                score = 0
                for q in addable:
                    if q == p:
                        continue
                    if (iso_with[p][q] & S_bits) != 0:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p == -1:
                break
            
            S.append(best_p)
            S_bits |= (1 << best_p)
            addable.remove(best_p)
            
            to_remove = []
            for q in addable:
                if (iso_with[best_p][q] & S_bits) != 0:
                    to_remove.append(q)
            
            for q in to_remove:
                addable.remove(q)
        
        S = local_search(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
    
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
