"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL = list(range(1, N + 1))

def _build_conflict():
    """
    Build a conflict structure.
    Two elements x, y are 'compatible' if for all z in ALL (z != x, z != y),
    gcd(x,y), gcd(x,z), gcd(y,z) are not all equal.
    Actually, the condition is: no three distinct a,b,c such that gcd(a,b)=gcd(a,c)=gcd(b,c).
    This is equivalent to: for any pair (x,y), there is no z such that
    gcd(x,y) == gcd(x,z) == gcd(y,z).
    Let d = gcd(x,y).
    We need to check if there exists z (z!=x, z!=y) such that d|x, d|y, d|z AND
    gcd(x,z)=d AND gcd(y,z)=d.
    Note: if d|x and d|y, then gcd(x,y) is a multiple of d. Wait, d IS gcd(x,y).
    So x = d*x', y = d*y' with gcd(x',y')=1.
    gcd(x,z) = d implies d|z and gcd(x', z/d) = 1.
    gcd(y,z) = d implies d|z and gcd(y', z/d) = 1.
    So we need z = d*k such that gcd(x',k)=1 and gcd(y',k)=1.
    Also z != x, z != y.
    If such a z exists, then (x,y,z) is a forbidden triple.
    So x and y are 'conflicting' if there exists z in [1,128] (z!=x, z!=y) such that
    gcd(x,z)=gcd(y,z)=gcd(x,y).
    """
    # Precompute gcds for all pairs? 128*128 is small.
    # But we can be smarter.
    # For each pair (x,y), let d = gcd(x,y).
    # We want to know if there is z such that:
    # 1. d | z
    # 2. gcd(x/d, z/d) = 1
    # 3. gcd(y/d, z/d) = 1
    # 4. z != x, z != y
    # Let x' = x/d, y' = y/d.
    # We need k = z/d such that:
    # 1. k is integer (implied by d|z)
    # 2. gcd(x', k) = 1
    # 3. gcd(y', k) = 1
    # 4. d*k != x, d*k != y  =>  k != x', k != y'
    # 5. 1 <= d*k <= 128  =>  1 <= k <= 128/d
    
    # Conflict[x][y] = True if such k exists.
    
    # Optimization:
    # If there is any k in [1, 128/d] \ {x', y'} coprime to both x' and y', then conflict.
    
    # Let's precompute for each d, the list of k's? No, d varies.
    # Let's just iterate.
    
    # Precompute primes? Or just use math.gcd.
    
    # For each pair (x,y) in ALL:
    #   d = gcd(x,y)
    #   x_prime = x // d
    #   y_prime = y // d
    #   limit = 128 // d
    #   for k in range(1, limit + 1):
    #       if k == x_prime or k == y_prime:
    #           continue
    #       if gcd(x_prime, k) == 1 and gcd(y_prime, k) == 1:
    #           conflict = True
    #           break
    #   else:
    #           conflict = False
            
    # This is O(N^2 * N/d). Sum of 128/d for d|gcd(x,y).
    # Worst case d=1, limit=128. N^2 * 128 = 16384 * 128 ~ 2M. Very fast.
    
    conflict = [[False]*N for _ in range(N)]
    for i in range(N):
        x = i + 1
        for j in range(i+1, N):
            y = j + 1
            d = gcd(x, y)
            xp = x // d
            yp = y // d
            limit = N // d
            is_conflict = False
            for k in range(1, limit + 1):
                if k == xp or k == yp:
                    continue
                if gcd(xp, k) == 1 and gcd(yp, k) == 1:
                    is_conflict = True
                    break
            conflict[i][j] = is_conflict
            conflict[j][i] = is_conflict
            
    return conflict

def _is_valid_subset(S, conflict):
    """Check if subset S (list of 0-indexed values) is valid."""
    # For any triple in S, they must not be a forbidden triple.
    # Wait, the conflict graph defined above: (x,y) is an edge if there EXISTS z such that (x,y,z) is forbidden.
    # If we pick a set S, is it sufficient that no pair in S is an edge in this conflict graph?
    # No.
    # If (x,y) is an edge, it means there exists SOME z (maybe not in S) that forms a triple.
    # The condition is: NO three DISTINCT elements in S form a triple.
    # So if x,y in S, and the only z that forms a triple with x,y is z0 not in S, then x,y can be in S together.
    # My conflict graph is too strict. It forbids x,y if ANY z in [1,128] creates a triple.
    # But we only care about z IN S.
    
    # So I need to check triples directly or use a different structure.
    # Given N=128, checking all triples is O(|S|^3).
    # If |S| is around 30, 30^3 = 27000 checks. Very fast.
    # If |S| is 100, 100^3 = 1,000,000. Still fast in Python if optimized? Maybe.
    # Let's stick to direct triple checking for validity, but use the conflict graph idea for search?
    # Actually, local search needs fast validity updates.
    # Adding x to S: check all pairs (a,b) in S. For each pair, check if gcd(a,b)==gcd(a,x)==gcd(b,x).
    # This is O(|S|^2).
    # Removing x from S: no check needed, just remove.
    # Swapping x out, y in: Remove x, Add y.
    
    # Let's implement a class or just use lists and helper functions.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute gcds for all pairs?
    # gcd_matrix[i][j] = gcd(i+1, j+1)
    gcd_mat = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = gcd(i+1, j+1)
            
    def check_triple(i, j, k):
        """Check if indices i,j,k form a forbidden triple."""
        # Values are i+1, j+1, k+1
        g_ij = gcd_mat[i][j]
        g_ik = gcd_mat[i][k]
        g_jk = gcd_mat[j][k]
        return g_ij == g_ik == g_jk

    def is_valid_addition(candidate_idx, S_idx_set, S_list):
        """Check if candidate_idx can be added to S."""
        c = candidate_idx
        for i in S_list:
            for j in S_list:
                if i >= j:
                    continue
                if check_triple(i, j, c):
                    return False
        return True

    # Greedy initialization
    S = []
    S_set = set()
    # Sort by number of prime factors? Or just 1..N?
    # Let's try 1..N first.
    for i in range(N):
        if is_valid_addition(i, S_set, S):
            S.append(i)
            S_set.add(i)
            
    best_S = S[:]
    best_size = len(S)
    
    # Local Search: 1-for-1 swaps
    # Try to swap an element in S with an element not in S.
    # Randomized restarts?
    
    random.seed(42)
    
    def local_search(S_init, max_steps=5000):
        S = S_init[:]
        S_set = set(S)
        current_size = len(S)
        
        # Precompute conflicts with current S?
        # For each x not in S, how many conflicts does it have with S?
        # If we remove y from S, and add x, we need to check:
        # 1. x is valid with S \ {y}
        # 2. No new triples formed.
        
        # Simpler:
        # Pick y in S.
        # Pick x not in S.
        # Tentatively remove y, add x.
        # Check validity.
        
        # To speed up, we can precompute for each x not in S, the set of pairs (a,b) in S
        # that form a forbidden triple with x.
        # If this set is empty, x can be added (if S was valid).
        # But we are swapping, so S changes.
        
        # Let's just do the O(|S|^2) check for each swap candidate.
        # |S| ~ 20-30. |S|^2 ~ 900.
        # Candidates: 128 * 30 = 3840 swaps.
        # 3840 * 900 = 3.4M ops per step? No, per step we try ONE swap or BEST swap.
        # If we try best swap, we evaluate all 3840 swaps.
        # 3840 * 900 = 3.4M gcd lookups.
        # In Python, this might take ~0.5s.
        # So we can do ~100 steps in 50s.
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            
            improved = False
            best_swap = None
            best_score = 0 # We want to minimize conflicts? Or just find ANY valid swap?
            
            # Since we maintain validity, any valid swap keeps size same.
            # We want to escape local optima.
            # Maybe try swaps that reduce "potential" conflicts?
            # Or just random valid swap?
            
            # Let's try to find a swap that allows adding MORE elements later?
            # Hard to measure.
            
            # Let's just do random restarts with greedy, and local search with random valid swaps.
            
            # For now, try all swaps, pick first valid one?
            # Or pick random valid one.
            
            valid_swaps = []
            
            # Optimization: Only consider x not in S that has low conflict count with S?
            
            # Let's just iterate.
            for y in S:
                for x in range(N):
                    if x in S_set:
                        continue
                    # Check if S - {y} + {x} is valid
                    # Equivalent to:
                    # 1. For all a,b in S, a!=y, b!=y, a<b: not check_triple(a,b,x)
                    # 2. For all a in S, a!=y: not check_triple(a,x,?) -> wait, triples are 3 elements.
                    #    The new set is S' = (S \ {y}) U {x}.
                    #    Triples in S' are:
                    #    - Triples entirely in S \ {y} (already valid)
                    #    - Triples involving x and two from S \ {y}
                    #    - Triples involving x and one from S \ {y} and ... no, only 3 distinct.
                    #    So we just need to check triples (a,b,x) where a,b in S \ {y}, a!=b.
                    
                    valid = True
                    for a in S:
                        if a == y: continue
                        for b in S:
                            if b == y or b <= a: continue
                            if check_triple(a, b, x):
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        valid_swaps.append((x, y))
            
            if valid_swaps:
                # Pick random swap
                x, y = random.choice(valid_swaps)
                S.remove(y)
                S_set.remove(y)
                S.append(x)
                S_set.add(x)
                improved = True
            
            if not improved:
                break
            steps += 1
            
        return S

    # Run local search from greedy
    current_S = local_search(best_S, max_steps=100)
    if len(current_S) > best_size:
        best_S = current_S
        best_size = len(current_S)
        
    # Random restarts
    while time.time() < deadline:
        # Randomized greedy
        random_S = []
        random_set = set()
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            if is_valid_addition(i, random_set, random_S):
                random_S.append(i)
                random_set.add(i)
                
        # Local search
        refined_S = local_search(random_S, max_steps=50)
        
        if len(refined_S) > best_size:
            best_S = refined_S
            best_size = len(refined_S)
            # Try to improve further?
            # current_S = local_search(best_S, max_steps=100)
            # if len(current_S) > best_size:
            #     best_S = current_S
            #     best_size = len(current_S)

    # Convert indices back to values
    return sorted([i+1 for i in best_S])

# EVOLVE-BLOCK-END
