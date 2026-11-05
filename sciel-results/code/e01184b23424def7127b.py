import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Candidate generation
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    candidate_sets = []
    
    # Generate subsets of small primes
    for r in range(len(small_primes) + 1):
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(1.0 / p for p in subset)
            if current_sum > 1.0:
                continue
            
            remaining_budget = 1.0 - current_sum
            current_set = list(subset)
            
            # Add large primes in ascending order (smallest first) to maximize count and coverage density
            for p in large_primes:
                if current_sum + 1.0 / p <= 1.0:
                    current_set.append(p)
                    current_sum += 1.0 / p
            
            # Avoid duplicates (though subsets are unique, the filling might result in same set? No, subset is unique)
            # But different subsets might lead to same final set if small primes are not included? 
            # No, if 2 is not in subset, it won't be in current_set.
            candidate_sets.append(current_set)
            
    # Remove duplicates just in case
    unique_sets = []
    seen = set()
    for s in candidate_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
            
    best_L = 0
    best_classes = []
    
    MAX_L = 100000
    
    def evaluate_set(P):
        # P is a list of primes
        P = sorted(P)
        
        # Greedy assignment with lookahead
        covered = [False] * (MAX_L + 1)
        residues = {}
        assigned = set()
        
        n = 1
        while n <= MAX_L:
            if time.time() > deadline - 5:
                break
                
            if covered[n]:
                n += 1
                continue
                
            # n is a gap. Find best prime to cover it.
            best_p = None
            best_score = -1
            best_r = None
            
            # Consider all unassigned primes
            for p in P:
                if p in assigned:
                    continue
                    
                r = n % p
                
                # Simulate: if we assign r to p, what is the first gap in [n, n+100]?
                # We check coverage of k in [n, n+100]
                # A number k is covered if:
                # 1. It was already covered by previous assignments
                # 2. It is covered by the new assignment (k % p == r)
                
                first_gap = n + 101 # Default if no gap in window
                for k in range(n, min(n + 101, MAX_L + 1)):
                    is_covered = covered[k] or (k % p == r)
                    if not is_covered:
                        first_gap = k
                        break
                        
                # We want to maximize the first gap
                if first_gap > best_score:
                    best_score = first_gap
                    best_p = p
                    best_r = r
            
            if best_p is None:
                # No primes left to assign
                break
                
            # Assign
            residues[best_p] = best_r
            assigned.add(best_p)
            
            # Update covered array
            for k in range(best_r if best_r > 0 else best_p, MAX_L + 1, best_p):
                covered[k] = True
                
            n += 1
            
        # Calculate L
        L = 0
        for i in range(1, MAX_L + 1):
            if covered[i]:
                L = i
            else:
                break
                
        # Local Search
        # Find first gap
        G = L + 1
        if G > MAX_L:
            return L, [[p, residues[p]] for p in P if p in residues]
            
        # Maintain cover_count for efficiency in local search?
        # For now, simple check: try to change residue of each assigned prime
        # to cover G, and verify no k < G is uncovered.
        
        # Precompute for each k < G, the list of primes covering it?
        # That's expensive. 
        # Alternative: Just check the specific numbers that would be affected.
        # If we change r_p to r'_p, the numbers covered by p change from {k | k% p == r_p} to {k | k% p == r'_p}.
        # We need to ensure that for all k < G:
        # if k was covered ONLY by p (with old r_p), then k must still be covered by new r'_p or by some other prime.
        
        # To check "covered ONLY by p", we need to know if any other prime q covers k.
        
        # Let's just try a few iterations of local search with a simple validation.
        
        improved = True
        while improved and G <= MAX_L:
            improved = False
            # Try each assigned prime
            for p in list(assigned):
                old_r = residues[p]
                new_r = G % p
                
                if old_r == new_r:
                    continue
                
                # Check if changing p to new_r breaks any k < G
                valid = True
                # We only need to check k that are covered by old_r but not new_r, and k < G
                # k = old_r, old_r+p, ... < G
                # And we must ensure k is not covered by new_r (it won't be, as old_r != new_r mod p)
                # So we just need to check if k is covered by ANY OTHER prime.
                
                for k in range(old_r if old_r > 0 else p, G, p):
                    # Check if k is covered by any q in assigned \ {p}
                    is_covered_by_other = False
                    for q in assigned:
                        if q == p:
                            continue
                        if k % q == residues[q]:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        valid = False
                        break
                
                if valid:
                    # Accept change
                    residues[p] = new_r
                    # Update covered array? 
                    # We don't really need the covered array for the next iteration of local search 
                    # if we re-calculate G from scratch or update it carefully.
                    # Let's just re-calculate L after a successful change? 
                    # Or update covered array.
                    # Unmark old_r multiples, mark new_r multiples.
                    for k in range(old_r if old_r > 0 else p, MAX_L + 1, p):
                        # This is O(MAX_L/p). 
                        # We only care if k < G was the only one covered by p?
                        # We already validated k < G.
                        # For k >= G, we just update the array for future gap finding.
                        pass # We'll just recompute L at the end of the while loop if we accept?
                    
                    # Actually, let's just recompute L from scratch after each accepted change? 
                    # That is O(MAX_L * |P|). 
                    # MAX_L=10^5, |P|=50. 5*10^6 ops. 
                    # If we do 100 iterations, 5*10^8. Might be too slow.
                    
                    # Better: Update covered array incrementally.
                    # But we need to know if a number becomes uncovered.
                    # This requires maintaining cover_count.
                    
                    # Let's skip the complex incremental update and just accept the change 
                    # and re-calculate the first gap G by scanning from G-1?
                    # No, changing residues can create gaps BEFORE G? 
                    # We validated that no k < G is broken. So gaps before G are safe.
                    # So we just need to find the new first gap starting from G.
                    
                    # Update covered array for p
                    # We need to unmark old_r and mark new_r.
                    # But we don't have cover_count. 
                    # So we can't easily unmark.
                    
                    # Let's stick to the simple validation and just re-run the "Find L" loop 
                    # but starting from G?
                    # No, we need to know if G is covered now.
                    # G is covered by new_r by definition.
                    # So we just need to find the next gap after G.
                    
                    # We can do this by scanning k = G+1, G+2... 
                    # and checking coverage using the updated residues dict.
                    # This is O(1) per k if we just check all primes.
                    # O(|P|) per k.
                    
                    # So:
                    improved = True
                    # Find new G
                    k = G
                    while k <= MAX_L:
                        is_covered = False
                        for q in assigned:
                            if k % q == residues[q]:
                                is_covered = True
                                break
                        if not is_covered:
                            G = k
                            break
                        k += 1
                    else:
                        G = MAX_L + 1
                        break
                        
                    # If G didn't change? It must have changed or G is still a gap?
                    # We covered G. So G must increase.
                    
                    break # Restart the "for p in assigned" loop with new G
        
        # Final L calculation
        L = 0
        for i in range(1, MAX_L + 1):
            is_covered = False
            for p in assigned:
                if i % p == residues[p]:
                    is_covered = True
                    break
            if is_covered:
                L = i
            else:
                break
                
        return L, [[p, residues[p]] for p in P if p in residues]

    for P in unique_sets:
        if time.time() > deadline - 10:
            break
            
        L, classes = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # Verify sum
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback
        return []
        
    return best_classes
