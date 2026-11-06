import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 256
    # Precompute conflict bitmasks
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = i + j
            p = i * j
            if p % s == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add_mask(v, smask):
        return (conflict_mask[v] & smask) == 0
    
    def greedy_fill(S_mask):
        """Greedily add all non-conflicting elements"""
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if (S_mask >> v) & 1:
                    continue
                if can_add_mask(v, S_mask):
                    S_mask |= (1 << v)
                    changed = True
        return S_mask
    
    def local_search(S_mask, deadline):
        """Perform local search with 1-for-2 and 2-for-1 moves"""
        best_size = popcount(S_mask)
        best_mask = S_mask
        
        # Precompute conflict counts for fast 1-for-2 detection
        # conflict_count[v] = number of elements in S that conflict with v
        # We'll maintain this incrementally
        
        while time.time() < deadline:
            improved = False
            S_list = [v for v in range(1, N + 1) if (S_mask >> v) & 1]
            random.shuffle(S_list)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                
                # Find all v not in S where u is the ONLY conflict
                cands_1for2 = []
                for v in range(1, N + 1):
                    if (S_mask >> v) & 1:
                        continue
                    if (conflict_mask[v] & S_mask) == (1 << u):
                        cands_1for2.append(v)
                
                # Try 1-for-2: remove u, add two non-conflicting candidates
                if len(cands_1for2) >= 2:
                    found = False
                    for i in range(len(cands_1for2)):
                        for j in range(i + 1, len(cands_1for2)):
                            v1, v2 = cands_1for2[i], cands_1for2[j]
                            if not ((conflict_mask[v1] >> v2) & 1):
                                S_mask &= ~(1 << u)
                                S_mask |= (1 << v1) | (1 << v2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        if popcount(S_mask) > best_size:
                            best_size = popcount(S_mask)
                            best_mask = S_mask
                        break  # Restart with new S
                
                # Try 2-for-1: remove u and w from S, add one v
                # v must conflict only with u and w in S
                # And v must not conflict with itself (trivial)
                # We need to find w in S and v not in S such that:
                #   (conflict_mask[v] & S_mask) == (1 << u) | (1 << w)
                #   and v doesn't conflict with any other element in S
                # This is harder; try a limited search
                if time.time() < deadline - 2:
                    for w in S_list:
                        if w == u:
                            continue
                        # Look for v with exactly conflicts {u, w}
                        for v in range(1, N + 1):
                            if (S_mask >> v) & 1:
                                continue
                            if (conflict_mask[v] & S_mask) == (1 << u) | (1 << w):
                                # Valid 2-for-1 move
                                S_mask &= ~(1 << u) & ~(1 << w)
                                S_mask |= (1 << v)
                                improved = True
                                break
                        if improved:
                            if popcount(S_mask) > best_size:
                                best_size = popcount(S_mask)
                                best_mask = S_mask
                            break
                    if improved:
                        break
            
            if not improved:
                # Try random 1-for-1 swaps to escape local minima
                if time.time() < deadline - 10:
                    for _ in range(30):
                        if time.time() > deadline - 5:
                            break
                        u = random.choice(S_list)
                        cands = [v for v in range(1, N + 1) 
                                 if not ((S_mask >> v) & 1) 
                                 and (conflict_mask[v] & S_mask) == (1 << u)]
                        if cands:
                            v = random.choice(cands)
                            S_mask &= ~(1 << u)
                            S_mask |= (1 << v)
                            break
        
        return best_mask, best_size
    
    # Try multiple different initial strategies
    results = []
    
    # Strategy 1: Start with all odds
    S_mask1 = 0
    for i in range(1, N + 1, 2):
        S_mask1 |= (1 << i)
    S_mask1 = greedy_fill(S_mask1)
    best1, size1 = local_search(S_mask1, deadline - 20)
    results.append((size1, best1))
    
    # Strategy 2: Start with all evens
    if time.time() < deadline - 15:
        S_mask2 = 0
        for i in range(2, N + 1, 2):
            S_mask2 |= (1 << i)
        S_mask2 = greedy_fill(S_mask2)
        best2, size2 = local_search(S_mask2, deadline - 10)
        results.append((size2, best2))
    
    # Strategy 3: Random initialization
    if time.time() < deadline - 10:
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            S_mask3 = 0
            # Random subset of ~100 elements
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for v in candidates[:100]:
                if can_add_mask(v, S_mask3):
                    S_mask3 |= (1 << v)
            S_mask3 = greedy_fill(S_mask3)
            best3, size3 = local_search(S_mask3, deadline - 5)
            results.append((size3, best3))
    
    # Return the best result
    if not results:
        return [1]
    
    best_size, best_mask = max(results, key=lambda x: x[0])
    result = [v for v in range(1, N + 1) if (best_mask >> v) & 1]
    if not result:
        result = [1]
    return result

if __name__ == "__main__":
    print(construct())
