import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Deterministic Construction: Quadratic Residues
    # A set of quadratic residues modulo a prime p is a Sidon set (B2[1]) 
    # in the sense that differences are distinct, but for sums, 
    # the set {x^2 mod p | x in Z_p} is a B2 set? 
    # Actually, the set of quadratic residues modulo p is NOT necessarily a Sidon set for sums.
    # However, a known construction for B2 sets is based on finite fields.
    # Let's use a simpler, verified construction: 
    # S = { i * M + j } is bad as shown.
    
    # Let's use the "modular" construction for Sidon sets:
    # Let p be a prime. S = { x^2 + p*x mod p^2 ? } No.
    
    # A standard construction for a Sidon set in [1, N] is:
    # Take a prime p such that p^2 <= N < (p+1)^2.
    # S = { i * p + j : 1 <= i <= p, 1 <= j <= p, i+j < p }? 
    # No, let's use the Bose-Chowla construction simplified.
    # Or simply: 
    # It is known that the set of squares modulo a prime p (lifted to integers) 
    # has good sum properties if we map them carefully.
    
    # Let's try a different deterministic base:
    # "Golf" construction:
    # We want a set S such that a+b=c+d implies {a,b}={c,d}.
    # Consider S = { x * K + y } where x,y are from a Sidon set in a smaller range?
    
    # Let's stick to a robust randomized approach but with a MUCH better initial seed.
    # Instead of empty start, start with a "dense" random subset that is likely B2[2].
    # Or, use the fact that a random set of size ~ c * sqrt(N) is Sidon with high prob.
    # For B2[2], we can go up to ~ 2 * sqrt(N) or more?
    # Theoretical max size for B2[k] in [1,N] is ~ k * sqrt(N).
    # For N=1000, sqrt(N)=31.6.
    # So we expect size around 60-100 for B2[1] (Sidon).
    # For B2[2], maybe 100-150?
    
    # The current program's score 0.9375 suggests it finds a set of size ~ 94 (if seed is 100).
    # Let's aim for higher.
    
    # Strategy:
    # 1. Generate a candidate pool.
    # 2. Use a fast greedy adder with a pre-computed conflict matrix?
    #    No, N=1000 is small enough for O(N^2) precomputation of potential conflicts?
    #    Actually, let's optimize the greedy loop heavily using NumPy for the check.
    
    # We will maintain `counts` as a numpy array.
    # When checking if x can be added:
    # We need to check if any y in S causes count[x+y] to exceed 2.
    # This is equivalent to: for all y in S, counts[x+y] < 2.
    # And counts[2*x] < 2.
    
    # We can vectorize the check if we keep S as a numpy array.
    # But S changes.
    
    # Let's try a "Local Search" approach starting from a random set of size ~ 80.
    # 1. Start with random set S of size M.
    # 2. Compute counts.
    # 3. Identify "bad" sums (count > 2).
    # 4. Remove an element involved in a bad sum.
    # 5. Try to add a new random element.
    # 6. Repeat.
    
    # This is a standard heuristic for such packing problems.
    
    def check_validity(S):
        """Returns True if S is a valid B2[2] set."""
        counts = np.zeros(2001, dtype=np.int8)
        S_list = list(S)
        for i, x in enumerate(S_list):
            counts[2*x] += 1
            for y in S_list[i+1:]:
                counts[x+y] += 1
        return np.all(counts <= 2)
    
    def get_bad_elements(S):
        """Returns a list of elements that are part of a sum with count > 2."""
        counts = np.zeros(2001, dtype=np.int8)
        pairs = []
        S_list = list(S)
        for i, x in enumerate(S_list):
            pairs.append((2*x, x, x))
            for j, y in enumerate(S_list[i+1:]):
                pairs.append((x+y, x, y))
        
        for s, a, b in pairs:
            counts[s] += 1
            
        bad_elements = set()
        for s, a, b in pairs:
            if counts[s] > 2:
                bad_elements.add(a)
                bad_elements.add(b)
        return bad_elements

    best_set = set()
    best_size = 0
    
    # Try a few different initial sizes and random seeds
    # We will run a local search loop.
    
    # To speed up, we can pre-compute a "conflict graph" or just rely on the fast loop.
    # The bottleneck is the Python loop in `get_bad_elements` and `check_validity`.
    # Let's optimize `check_validity` and the add/remove logic.
    
    # Alternative: 
    # Use a deterministic construction for a large Sidon set as a seed.
    # A known large Sidon set in [1, N] is constructed using finite fields.
    # For N=1000, we can find a prime p=31 (31^2=961).
    # The set of quadratic residues mod 31 is {1, 4, 9, 16, 25, 5, 18, 2, 19, 7, 28, 14, 10, 20, 8, 29, 15, 11, 21, 3, 26, 13, 6, 30, 17, 1, ...}
    # Wait, the set of squares mod p has size (p-1)/2.
    # Is it a Sidon set? 
    # The set of squares mod p is a difference set, but not necessarily a Sidon set for sums in integers.
    # However, the set { i * p + j } where i, j are from a small Sidon set is a Sidon set.
    # Let's use:
    # Let A be a Sidon set in [1, k].
    # Let B be a Sidon set in [1, m].
    # S = { a * (m+1) + b } is a Sidon set if k*(m+1)+m <= N.
    # We want to maximize |A|*|B|.
    # Let k=m. Size ~ k^2. Max element ~ k^2.
    # If k=31, size ~ 961. Max element ~ 31*32+31 = 1023 > 1000.
    # So k=30. Size 900? 
    # Wait, if A is Sidon in [1,30], B is Sidon in [1,30].
    # S = { a * 31 + b }.
    # Max element: 30*31 + 30 = 960.
    # Is S a Sidon set?
    # Sum: (a1*31+b1) + (a2*31+b2) = (a1+a2)*31 + (b1+b2).
    # If (a1+a2)*31 + (b1+b2) = (a3+a3)*31 + (b3+b4).
    # Since b1+b2 <= 60 and 31 < 60, there is a carry risk.
    # If b1+b2 >= 31, it carries over.
    # So this is NOT a Sidon set.
    
    # We need b1+b2 < 31 to avoid carry.
    # So we need B to be a set such that all sums are < 31.
    # If B is Sidon in [1, 15], max sum 15+15=30 < 31.
    # So let A be Sidon in [1, 30], B be Sidon in [1, 15].
    # Size = 30 * 15 = 450.
    # Max element: 30*31 + 15 = 945.
    # This is a valid Sidon set (B2[1]) of size 450!
    # Why did the current program only get ~94?
    # Because it used random greedy which is very inefficient for finding large structured sets.
    # A deterministic construction gives 450 immediately.
    
    # Let's verify this construction.
    # A = [1..30]. Is [1..30] a Sidon set? No! 1+4 = 2+3.
    # We need A to be a Sidon set.
    # Max size of Sidon set in [1, 30] is ~ sqrt(30) ~ 5.
    # So we can't use [1..30] as A.
    # We need to find a large Sidon set A in [1, K] and B in [1, L].
    # Let's use the modular construction for A and B.
    # For range [1, 30], a Sidon set can be {1, 2, 4, 8, 16}? No, 1+8=2+7? 7 not in set.
    # {1, 2, 4, 8, 16} sums:
    # 1+2=3, 1+4=5, 1+8=9, 1+16=17
    # 2+4=6, 2+8=10, 2+16=18
    # 4+8=12, 4+16=20
    # 8+16=24
    # All distinct? Yes. Size 5.
    # Can we do better in [1,30]? 
    # Known max Sidon set size in [1,30] is around 7-8.
    # Let's just use a random Sidon set finder for small ranges to get A and B.
    
    def find_sidon_set(n, k):
        """Find a Sidon set of size k in [1, n] using greedy."""
        # Greedy doesn't always find max, but for small n it's okay.
        # We'll use a simple backtracking or greedy with restarts.
        best = set()
        for _ in range(100):
            S = set()
            counts = np.zeros(2*n+1, dtype=np.int8)
            # Random order
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
            for x in candidates:
                if counts[2*x] >= 1: continue # Sidon means max 1 rep
                ok = True
                for y in S:
                    if counts[x+y] >= 1:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    counts[2*x] += 1
                    for y in S:
                        if y != x:
                            counts[x+y] += 1
            if len(S) > len(best):
                best = S
                if len(best) == k:
                    break
        return best
    
    # We want to maximize |A| * |B| subject to:
    # A is Sidon in [1, K], B is Sidon in [1, L].
    # Max element in S = K * (L+1) + L <= 1000.
    # And we need L+1 > max_sum_B = 2*max(B).
    # So L+1 > 2*max(B). Since B is in [1,L], max(B)<=L.
    # So L+1 > 2L is impossible for L>1.
    # Wait, the condition for no carry is max_sum_B < L+1.
    # max_sum_B = b1+b2. Max b is L. So max sum is 2L.
    # We need 2L < L+1 => L < 1. Impossible.
    
    # So the simple product construction S = { a * M + b } fails for Sidon sets unless we restrict B heavily.
    # If we restrict B to [1, M//2], then max sum is M. Still carry if sum == M.
    # We need max sum < M.
    # So B must be a Sidon set in [1, (M-1)//2].
    # Let M = L+1.
    # B is Sidon in [1, L//2].
    # A is Sidon in [1, K].
    # Max element: K * (L+1) + L//2 <= 1000.
    # Size = |A| * |B|.
    
    # Let's choose L such that L//2 is large enough to have a decent Sidon set.
    # Let L = 20. B in [1, 10]. Max Sidon size in [1,10] is ~ 4-5.
    # Let K = 1000 // 21 ~ 47. A in [1, 47]. Max Sidon size in [1,47] is ~ 7-8.
    # Size ~ 5 * 8 = 40.
    # This is worse than the random greedy which got ~94.
    
    # Why is random greedy better?
    # Because it doesn't force the rigid structure.
    # The random greedy in the current program is actually not that bad, it just needs more time/restarts.
    # The score 0.9375 implies size ~ 94 (if seed=100).
    # Let's assume the seed program gets 100.
    # We need to beat 100.
    
    # Let's improve the local search.
    # 1. Start with a random set of size 100 that is "almost" valid?
    #    Or start with empty and greedy add.
    # 2. Use a "swap" move: remove x, add y.
    
    # Let's implement a fast local search.
    
    best_set = set()
    best_size = 0
    
    def try_local_search(initial_set, time_limit):
        S = set(initial_set)
        counts = np.zeros(2001, dtype=np.int8)
        for x in S:
            counts[2*x] += 1
            for y in S:
                if y != x:
                    counts[x+y] += 1
        
        # Verify initial set
        if np.any(counts > 2):
             # Remove bad elements
             bad = get_bad_elements(S)
             for b in bad:
                 S.remove(b)
             # Recalculate counts
             counts = np.zeros(2001, dtype=np.int8)
             for x in S:
                 counts[2*x] += 1
                 for y in S:
                     if y != x:
                         counts[x+y] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add
            for _ in range(50):
                x = random.randint(1, 1000)
                if x in S: continue
                if counts[2*x] >= 2: continue
                ok = True
                for y in S:
                    if counts[x+y] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    counts[2*x] += 1
                    for y in S:
                        if y != x:
                            counts[x+y] += 1
                    improved = True
                    break
            
            if not improved:
                # Try to swap
                # Pick a random element x in S, remove it, then try to add a new one.
                # This is O(|S|^2) if we do it naively.
                # Let's try 10 random swaps.
                for _ in range(10):
                    if not S: break
                    x_to_remove = random.choice(list(S))
                    # Remove x_to_remove
                    S.remove(x_to_remove)
                    # Update counts
                    counts[2*x_to_remove] -= 1
                    for y in S:
                        counts[x_to_remove+y] -= 1
                    
                    # Try to add a new element
                    added = False
                    for _ in range(20):
                        x_new = random.randint(1, 1000)
                        if x_new in S: continue
                        if counts[2*x_new] >= 2: continue
                        ok = True
                        for y in S:
                            if counts[x_new+y] >= 2:
                                ok = False
                                break
                        if ok:
                            S.add(x_new)
                            counts[2*x_new] += 1
                            for y in S:
                                if y != x_new:
                                    counts[x_new+y] += 1
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Re-add the removed element to restore state
                        S.add(x_to_remove)
                        counts[2*x_to_remove] += 1
                        for y in S:
                            if y != x_to_remove:
                                counts[x_to_remove+y] += 1
                        
        return S

    # Run local search from several random starts
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start with a random set of size 50
        initial_set = set(random.sample(range(1, 1001), 50))
        # Clean it up first
        S = try_local_search(initial_set, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Also try greedy from scratch
        S_greedy = set()
        counts = np.zeros(2001, dtype=np.int8)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if counts[2*x] >= 2: continue
            ok = True
            for y in S_greedy:
                if counts[x+y] >= 2:
                    ok = False
                    break
            if ok:
                S_greedy.add(x)
                counts[2*x] += 1
                for y in S_greedy:
                    if y != x:
                        counts[x+y] += 1
        
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_set = S_greedy

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
