"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

def _is_smooth(n, primes):
    """Check if n is S-smooth (all prime factors in primes)."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of prime
    factors of all pairwise sums (distinct indices) is minimized.
    
    Strategy: Try small sets of primes S = {2,3,5,7,...}. For each S,
    precompute all S-smooth numbers <= 10000. Then find a clique of size 12
    in the compatibility graph where two numbers are compatible if their sum
    is also S-smooth. Return the set with the smallest |S|.
    """
    max_val = 10000
    n_needed = 12
    
    # Try increasing sets of primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    best_result = None
    best_prime_count = float('inf')
    
    # Try all subsets of base_primes in order of size
    for size in range(1, len(base_primes) + 1):
        for prime_subset in combinations(base_primes, size):
            if best_result is not None and size >= best_prime_count:
                break
            
            # Generate all S-smooth numbers <= max_val
            smooth_numbers = set()
            primes_list = list(prime_subset)
            
            # Use DFS to generate smooth numbers
            def gen_smooth(idx, current):
                if current > max_val:
                    return
                if current > 0:
                    smooth_numbers.add(current)
                if idx >= len(primes_list):
                    return
                p = primes_list[idx]
                val = current
                while val * p <= max_val:
                    val *= p
                    gen_smooth(idx + 1, val)
            
            gen_smooth(0, 1)
            
            if len(smooth_numbers) < n_needed:
                continue
            
            smooth_list = sorted(smooth_numbers)
            
            # Build compatibility graph: edge between i,j if smooth_list[i] + smooth_list[j] is S-smooth
            # We need a clique of size n_needed
            # Use backtracking with ordering
            
            # Precompute compatibility
            # For efficiency, only consider numbers that have enough compatible neighbors
            # But first, let's just try backtracking on the full set if it's small enough
            
            if len(smooth_list) > 500:
                # Too many, skip this prime set or use smarter approach
                # Actually, let's still try but with pruning
                pass
            
            # Backtracking to find clique of size n_needed
            # Order nodes by degree (number of compatible neighbors)
            
            compat = [set() for _ in range(len(smooth_list))]
            
            # This could be O(n^2) which is fine for n up to a few thousand
            for i in range(len(smooth_list)):
                for j in range(i + 1, len(smooth_list)):
                    s = smooth_list[i] + smooth_list[j]
                    if _is_smooth(s, primes_list):
                        compat[i].add(j)
                        compat[j].add(i)
            
            # Sort indices by degree (descending) for better pruning
            indices_by_degree = sorted(range(len(smooth_list)), key=lambda x: -len(compat[x]))
            
            # Backtracking
            found_clique = []
            
            def backtrack(start_idx, current_clique, available):
                if len(current_clique) == n_needed:
                    return True
                if start_idx >= len(indices_by_degree):
                    return False
                
                # Pruning: not enough nodes left
                remaining = len(available) - start_idx + 1
                if remaining < n_needed - len(current_clique):
                    return False
                
                for i in range(start_idx, len(indices_by_degree)):
                    node = indices_by_degree[i]
                    if node not in available:
                        continue
                    
                    new_available = available & compat[node]
                    if len(new_available) < n_needed - len(current_clique) - 1:
                        continue
                    
                    current_clique.append(node)
                    if backtrack(i + 1, current_clique, new_available):
                        return True
                    current_clique.pop()
                
                return False
            
            available = set(range(len(smooth_list)))
            if backtrack(0, [], available):
                result = [smooth_list[i] for i in found_clique] if found_clique else None
                # Wait, the backtrack function returns True but doesn't populate found_clique properly
                # Let me fix this
                
                # Actually, let me redo the backtracking to properly capture the result
                pass
            
            # Let me redo with proper result capture
            result_clique = []
            
            def backtrack2(start_idx, current_clique, available):
                if len(current_clique) == n_needed:
                    result_clique.extend(current_clique)
                    return True
                if start_idx >= len(indices_by_degree):
                    return False
                
                remaining = len(available) - start_idx + 1
                if remaining < n_needed - len(current_clique):
                    return False
                
                for i in range(start_idx, len(indices_by_degree)):
                    node = indices_by_degree[i]
                    if node not in available:
                        continue
                    
                    new_available = available & compat[node]
                    if len(new_available) < n_needed - len(current_clique) - 1:
                        continue
                    
                    current_clique.append(node)
                    if backtrack2(i + 1, current_clique, new_available):
                        return True
                    current_clique.pop()
                
                return False
            
            available = set(range(len(smooth_list)))
            if backtrack2(0, [], available):
                result = [smooth_list[i] for i in result_clique]
                if len(result) == n_needed and len(set(result)) == n_needed:
                    # Verify
                    primes_used = set()
                    for a, b in combinations(result, 2):
                        s = a + b
                        temp = s
                        for p in primes_list:
                            while temp % p == 0:
                                primes_used.add(p)
                                temp //= p
                        if temp != 1:
                            # Shouldn't happen
                            break
                    else:
                        if len(primes_used) <= size:
                            best_result = result
                            best_prime_count = size
                            return result
                break  # Found a solution for this size, move to next size (but we want smallest, so return)
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a reasonable set
    # Use numbers that are powers of 2, so all sums are even, and hopefully few prime factors
    # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# EVOLVE-BLOCK-END
