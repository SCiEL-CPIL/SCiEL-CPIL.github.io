import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback: Known Paley tournament for n=83
    # p = 83 is a prime of form 4k+3? 83 = 4*20 + 3. Yes.
    # Paley tournament: i -> j if (j-i) is a quadratic residue mod p.
    def get_paley_83():
        p = 83
        # Compute quadratic residues
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # 0 means i -> j, 1 means j -> i
                # i -> j iff (j - i) is a square
                diff = (j - i) % p
                if squares[diff]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    # Fast validity check using NumPy
    # A 4-set {a,b,c,d} is valid if any vertex dominates the other 3.
    # Invalid if it is isomorphic to the (2,2,1,1) score sequence tournament.
    # We can count the number of invalid 4-sets efficiently.
    
    def count_invalid(adj):
        """
        adj: numpy array (n, n) of bools. adj[i,j] is True if i -> j.
        Returns count of 4-sets with no dominator.
        """
        n = len(adj)
        if n < 4:
            return 0
        
        # A 4-set is invalid if the induced subgraph has max out-degree 2.
        # This happens for the two non-transitive tournaments on 4 vertices?
        # No, only one isomorphism class is invalid: the one with score sequence (2,2,1,1).
        # The other non-transitive one is (3,1,1,1) which has a dominator (score 3).
        # Wait, (3,1,1,1) has a vertex with out-degree 3, so it is valid.
        # (2,2,1,1) has max out-degree 2, so it is invalid.
        # (2,2,2,0) is impossible for a tournament (sum 6, but 0 is sink, others must point to it.
        # If 3 is sink, 0,1,2 form a tournament. Scores 2,1,0. Total 3,2,1. So (3,2,1,0) is transitive/valid).
        
        # So we just need to count 4-sets where NO vertex has out-degree 3.
        
        # Method:
        # For each vertex v, let S_v be the set of vertices v dominates.
        # A 4-set {a,b,c,d} is valid if it contains a vertex v such that the other 3 are in S_v.
        # This is hard to count directly with inclusion-exclusion without double counting.
        
        # Alternative: Count all 4-sets, subtract valid ones?
        # Or count invalid ones directly.
        # An invalid 4-set is a "cyclic" type or similar?
        # Actually, let's just check all 4-sets using vectorized operations.
        # n is at most ~60. C(60,4) ~ 480k.
        # We can generate all indices and check them in batches.
        
        # Generate all 4-combinations
        # This might be memory heavy if done all at once for n=60?
        # 480k * 4 integers = 1.9M integers ~ 15MB. Acceptable.
        
        # However, generating combinations in Python is slow.
        # Let's use a different check:
        # A tournament is valid iff for every 4-set, max out-degree is 3.
        
        # Let's try a heuristic check for small n, but for the search we need speed.
        # Let's use the property:
        # Number of invalid 4-sets = Sum over all 4-sets of [MaxOutDeg < 3]
        
        # We can compute this by iterating over pairs of "non-domination"?
        # Actually, for the local search, we can just use a random sample of 4-sets
        # to estimate the cost function, and do a full check only when cost is low.
        
        # For now, let's implement a reasonably fast full check using NumPy broadcasting
        # on a subset of vertices if n is large, or full if n is small.
        
        # Since n <= 60 in our search, let's just do the full check.
        # To avoid generating 480k tuples in Python, we can use np.indices? No.
        
        # Let's stick to a Python loop but optimized?
        # No, the previous program failed because of Python loops.
        
        # Let's use a different approach for the cost function in the search:
        # Randomly sample 1000 4-sets. If any is invalid, cost += 1.
        # This is fast and correlates with total violations.
        
        pass 

    def sample_invalid_count(adj, n, num_samples=200):
        """
        Estimate number of invalid 4-sets by sampling.
        Returns 0 if no invalid sets found in sample (likely valid or very close).
        """
        invalid_count = 0
        # Precompute out-neighbors for fast check?
        # adj is a bool matrix.
        
        for _ in range(num_samples):
            # Pick 4 distinct random vertices
            # Using random.sample is fast enough for 200 samples
            verts = random.sample(range(n), 4)
            a, b, c, d = verts
            
            # Check if any dominates the other 3
            # a dominates if adj[a,b] and adj[a,c] and adj[a,d]
            if adj[a, b] and adj[a, c] and adj[a, d]:
                continue
            if adj[b, a] and adj[b, c] and adj[b, d]:
                continue
            if adj[c, a] and adj[c, b] and adj[c, d]:
                continue
            if adj[d, a] and adj[d, b] and adj[d, c]:
                continue
            
            invalid_count += 1
        
        return invalid_count

    def is_fully_valid(adj, n):
        """
        Exhaustive check. Only call when sample count is 0.
        """
        # For n up to 60, C(n,4) is ~500k.
        # We can do this in a few seconds in Python if optimized, or use NumPy tricks.
        # Let's use a NumPy trick:
        # Create a 4D array of all 4-sets? Too big.
        
        # Let's use a recursive or iterative check with early exit.
        # Since we only call this when sample is 0, it's rare.
        
        # Actually, let's just do a full Python loop here. It's the final verification.
        # If it takes 5 seconds, that's fine.
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if (adj[i, j] and adj[i, k] and adj[i, l]):
                            continue
                        if (adj[j, i] and adj[j, k] and adj[j, l]):
                            continue
                        if (adj[k, i] and adj[k, j] and adj[k, l]):
                            continue
                        if (adj[l, i] and adj[l, j] and adj[l, k]):
                            continue
                        return False
        return True

    def to_orientation(adj, n):
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                # 0 means i -> j
                orientation.append(0 if adj[i, j] else 1)
        return orientation

    # Search Strategy:
    # Try n from 31 to 67.
    # For each n, run Simulated Annealing.
    
    best_n = 83
    best_orient = None
    
    # We will prioritize smaller n.
    # However, finding a valid tournament for n=31 is extremely hard (prob ~ 0).
    # The density of valid tournaments drops rapidly.
    # Let's start from a higher n where random tournaments are closer to valid,
    # or use a construction that is "almost" valid.
    
    # Actually, let's try n=60 down to 40. Why?
    # Because random tournaments for n=60 have many violations, but local search can fix them?
    # Or maybe start from n=67 (known to exist) and try to reduce? No, that's hard.
    
    # Let's try random restarts for n in [40, 60].
    # If we find one, return it.
    
    # To improve the chance, we can use a "greedy" initialization:
    # Start with a random tournament.
    
    for n in range(40, 68): # Try 40 to 67
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this n
        max_restarts = 5
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize random tournament
            adj = np.random.rand(n, n) < 0.5
            # Make it a tournament: adj[i,j] = not adj[j,i]
            # Ensure diagonal is 0
            np.fill_diagonal(adj, False)
            # Symmetrize: if adj[i,j] is True, adj[j,i] must be False
            # We can do: adj = (adj > 0.5) | (~adj.T > 0.5) ? No.
            # Simplest:
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        adj[j, i] = False
                    else:
                        adj[i, j] = False
                        adj[j, i] = True
            
            # Local Search / Simulated Annealing
            current_cost = sample_invalid_count(adj, n)
            
            # If cost is 0 in sample, check full
            if current_cost == 0:
                if is_fully_valid(adj, n):
                    best_n = n
                    best_orient = to_orientation(adj, n)
                    found = True
                    break
            
            # Annealing
            T = 1.0
            for step in range(5000): # Max steps per restart
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Flip edge i-j
                old_val = adj[i, j]
                adj[i, j] = not old_val
                adj[j, i] = not adj[j, i] # Wait, if i->j becomes j->i, then adj[i,j] becomes False, adj[j,i] becomes True.
                # My flip logic:
                # If adj[i,j] was True (i->j), it becomes False (j->i).
                # So adj[i,j] = False, adj[j,i] = True.
                # If adj[i,j] was False (j->i), it becomes True (i->j).
                # So adj[i,j] = True, adj[j,i] = False.
                # This is equivalent to:
                # adj[i,j] = not old_val
                # adj[j,i] = old_val
                
                # Re-evaluate cost
                new_cost = sample_invalid_count(adj, n)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    # Accept move
                else:
                    # Revert
                    adj[i, j] = old_val
                    adj[j, i] = not old_val
                
                # Cool down
                T *= 0.999
                
                if current_cost == 0:
                    if is_fully_valid(adj, n):
                        best_n = n
                        best_orient = to_orientation(adj, n)
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break

    if best_orient is not None:
        return {'n': best_n, 'orientation': best_orient}
    else:
        return get_paley_83()

# EVOLVE-BLOCK-END
