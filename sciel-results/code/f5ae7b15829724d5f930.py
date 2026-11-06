"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    prime_set = set(primes)
    
    def compute_coverage_fast(classes, max_L):
        """Compute coverage using a boolean array for speed."""
        covered = bytearray(max_L + 1)
        for p, r in classes:
            # Mark all n in [1, max_L] where n % p == r
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, max_L + 1, p):
                covered[n] = 1
        # Find largest L such that 1..L all covered
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, covered
    
    def compute_coverage(classes, max_L):
        """Legacy interface, uses fast version."""
        return compute_coverage_fast(classes, max_L)
    
    # Start with empty classes and build greedily
    best_L = 0
    best_classes = []
    
    def greedy_build(target_L, initial_classes=None):
        """Build a covering of 1..target_L using greedy set cover with efficiency scoring."""
        if initial_classes is not None:
            classes = [c[:] for c in initial_classes]
            used_primes = set(c[0] for c in classes)
            budget_remaining = Fraction(1, 1) - sum(Fraction(1, c[0]) for c in classes)
        else:
            classes = []
            used_primes = set()
            budget_remaining = Fraction(1, 1)
        
        # Compute current coverage
        covered_arr = bytearray(target_L + 1)
        for p, r in classes:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, target_L + 1, p):
                covered_arr[n] = 1
        
        # Count uncovered
        uncovered_count = sum(1 for i in range(1, target_L + 1) if not covered_arr[i])
        
        if uncovered_count == 0:
            return classes, target_L
        
        # Greedy: repeatedly pick the (p, r) that maximizes new_coverage * p
        # (efficiency = coverage per unit budget)
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            best_score = -1
            best_p = None
            best_r = None
            best_new_count = 0
            best_new_positions = None
            
            # For each unused prime, try all residues
            for p in primes:
                if p in used_primes:
                    continue
                if Fraction(1, p) > budget_remaining:
                    continue
                
                # For each residue r, compute how many currently-uncovered positions it covers
                # We can do this efficiently by iterating through uncovered positions mod p
                # But for speed, let's just try each residue and count
                
                for r in range(p):
                    # Count how many n in [1, target_L] with n % p == r are currently uncovered
                    count = 0
                    if r == 0:
                        start = p
                    else:
                        start = r
                    for n in range(start, target_L + 1, p):
                        if not covered_arr[n]:
                            count += 1
                    
                    if count == 0:
                        continue
                    
                    # Score: new_coverage * p (value per unit budget)
                    score = count * p
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r
                        best_new_count = count
                        best_new_positions = list(range(start, target_L + 1, p))
            
            if best_p is None:
                break
            
            # Add this class
            used_primes.add(best_p)
            classes.append([best_p, best_r])
            budget_remaining -= Fraction(1, best_p)
            
            # Update coverage
            for n in best_new_positions:
                covered_arr[n] = 1
            
            uncovered_count -= best_new_count
        
        # Compute final L
        L = 0
        for i in range(1, target_L + 1):
            if covered_arr[i]:
                L = i
            else:
                break
        
        return classes, L
    
    def local_search(classes, current_L, max_time=30.0):
        """Try to improve coverage via local search (swaps and moves)."""
        if time.time() > deadline - max_time:
            return classes, current_L
        
        best_classes = [c[:] for c in classes]
        best_L = current_L
        best_budget = sum(Fraction(1, c[0]) for c in best_classes)
        
        check_L = max(current_L + 200, 1500)
        
        # Local search: try replacing or adding classes
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try removing a class and re-adding a different one
            for i in range(len(best_classes)):
                if time.time() > deadline - 10.0:
                    break
                old_p, old_r = best_classes[i]
                
                # Try replacing with each unused prime
                for new_p in primes:
                    if new_p == old_p:
                        continue
                    if any(c[0] == new_p for c in best_classes):
                        continue
                    
                    # Check budget: new_p must not increase budget
                    if Fraction(1, new_p) > Fraction(1, old_p):
                        continue
                    
                    # Try all residues for new_p
                    # Compute coverage without the old class
                    temp_classes = best_classes[:i] + best_classes[i+1:]
                    L_no_old, _ = compute_coverage_fast(temp_classes, check_L)
                    
                    # For each residue, compute new coverage
                    best_new_r = None
                    best_new_L = L_no_old
                    
                    for r in range(new_p):
                        test_classes = temp_classes + [[new_p, r]]
                        L_test, _ = compute_coverage_fast(test_classes, check_L)
                        if L_test > best_new_L:
                            best_new_L = L_test
                            best_new_r = r
                    
                    if best_new_r is not None and best_new_L > best_L:
                        best_classes[i] = [new_p, best_new_r]
                        best_L = best_new_L
                        improved = True
                        break
                
                if improved:
                    break
            
            # Try adding a new class if budget allows
            current_budget = sum(Fraction(1, c[0]) for c in best_classes)
            if current_budget < Fraction(1, 1):
                for p in primes:
                    if any(c[0] == p for c in best_classes):
                        continue
                    if Fraction(1, p) > Fraction(1, 1) - current_budget:
                        continue
                    
                    # Try all residues
                    best_new_r = None
                    best_new_L = best_L
                    
                    for r in range(p):
                        test_classes = best_classes + [[p, r]]
                        L_test, _ = compute_coverage_fast(test_classes, check_L)
                        if L_test > best_new_L:
                            best_new_L = L_test
                            best_new_r = r
                    
                    if best_new_r is not None and best_new_L > best_L:
                        best_classes.append([p, best_new_r])
                        best_L = best_new_L
                        improved = True
                        break
        
        return best_classes, best_L
    
    # Phase 1: Greedy construction
    # Try building from scratch with different target sizes
    for target in [100, 200, 500, 1000, 1500, 2000, 3000, 5000]:
        if time.time() > deadline - 20.0:
            break
        
        classes, L = greedy_build(target)
        if L > best_L:
            best_L = L
            best_classes = classes[:]
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 30.0:
        best_classes, best_L = local_search(best_classes, best_L, max_time=30.0)
    
    # Phase 3: Try greedy from current solution with higher target
    if time.time() < deadline - 20.0:
        for target in [best_L + 100, best_L + 500, best_L + 1000, 5000]:
            if time.time() > deadline - 10.0:
                break
            if target <= best_L:
                continue
            
            classes, L = greedy_build(target, initial_classes=best_classes)
            if L > best_L:
                best_L = L
                best_classes = classes[:]
        
        # Another local search round
        if time.time() < deadline - 10.0:
            best_classes, best_L = local_search(best_classes, best_L, max_time=10.0)
    
    # Final verification with a larger check
    final_L, _ = compute_coverage_fast(best_classes, max(best_L + 500, 10000))
    if final_L > best_L:
        best_L = final_L
    
    return best_classes

# EVOLVE-BLOCK-END
