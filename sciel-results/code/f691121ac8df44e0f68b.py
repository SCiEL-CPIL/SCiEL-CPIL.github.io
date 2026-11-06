import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def is_smooth(n, max_prime):
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p > max_prime:
                return False
            while n % p == 0:
                n //= p
        return True

    def find_clique(P, N):
        """Find a clique of size 12 in graph of 1..N where edges connect u,v if u+v is P-smooth."""
        # Build adjacency
        adj = [set() for _ in range(N + 1)]
        for u in range(1, N + 1):
            for v in range(u + 1, N + 1):
                if is_smooth(u + v, P):
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Greedy clique search with random restarts
        best_clique = []
        for _ in range(500): # 500 restarts
            current_clique = []
            candidates = set(range(1, N + 1))
            random.shuffle(list(candidates)) # Randomize order
            
            while len(current_clique) < 12:
                if not candidates:
                    break
                # Pick node with max degree in candidates
                best_node = None
                max_deg = -1
                # To speed up, we can just pick first, but max degree is better
                # N is small (100-200), so this is fine
                for node in candidates:
                    deg = len(adj[node] & candidates)
                    if deg > max_deg:
                        max_deg = deg
                        best_node = node
                
                if best_node is None:
                    break
                    
                current_clique.append(best_node)
                candidates = candidates & adj[best_node]
                
            if len(current_clique) > len(best_clique):
                best_clique = current_clique
                if len(best_clique) >= 12:
                    return best_clique[:12]
        
        return best_clique

    # Try to find clique for P=2 (Score 1)
    # Unlikely to succeed, but fast to check
    # N=50 is enough for P=2
    clique = find_clique(2, 50)
    if len(clique) >= 12:
        max_val = max(clique)
        # Scale by power of 2
        k = 1
        while k * max_val <= 10000:
            k *= 2
        # Actually, we want largest k? No, any k works. 
        # Using k=1 is fine if values <= 10000.
        # If max_val <= 10000, return clique.
        # Else scale down? No, we need distinct ints.
        # If max_val > 10000, we can't scale up. We must have chosen N small enough.
        # For P=2, N=50, max sum 99. 50 < 10000.
        return [x * 1 for x in clique[:12]] # Just return them, they are <= 50

    # Try P=3 (Score 2)
    clique = find_clique(3, 100)
    if len(clique) >= 12:
        max_val = max(clique)
        # Scale to use range up to 10000? 
        # Not necessary, just return them if they fit.
        # If they fit, primes are {2,3}.
        # Check if they fit.
        if max_val <= 10000:
            return clique[:12]
        # If not, we need to scale DOWN? No, we can't scale down and keep integers distinct easily without breaking smoothness?
        # Actually, if u+v is smooth, (u/2)+(v/2) is not necessarily integer.
        # So we should have chosen N such that max_val <= 10000.
        # N=100 is safe.

    # Try P=5 (Score 3)
    clique = find_clique(5, 100)
    if len(clique) >= 12:
        return clique[:12]

    # Try P=7 (Score 4)
    clique = find_clique(7, 100)
    if len(clique) >= 12:
        return clique[:12]
    
    # Fallback: Local search from seed
    # Use the seed's logic if clique search fails
    def factorize(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    
    random.seed(42)
    candidate_pool = list(set(random.randint(1, 10000) for _ in range(5000)))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if new_val == old_val or new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break

    return best
