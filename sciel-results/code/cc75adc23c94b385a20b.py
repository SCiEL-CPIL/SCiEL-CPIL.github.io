"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import numpy as np

def _score_set_fast(U):
    """Compute the score for a given set U using vectorized operations."""
    if not U or 0 not in U:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    if n < 2:
        return 0.0
    
    # Compute all pairwise sums and differences
    # Using broadcasting: U_arr[:, None] + U_arr[None, :]
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    # Get unique values
    sum_unique = np.unique(sums)
    diff_unique = np.unique(diffs)
    
    sum_count = len(sum_unique)
    diff_count = len(diff_unique)
    
    max_u = int(np.max(U_arr))
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / log_denom
    return score

def _generate_primes(limit):
    """Generate primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def _construct_from_primes(primes, k):
    """
    Construct U = {0, p_1, p_1+p_2, ..., sum of first k primes}
    from a list of primes.
    """
    if k <= 0:
        return [0]
    
    selected = primes[:k]
    U = [0]
    cumulative = 0
    for p in selected:
        cumulative += p
        U.append(cumulative)
    
    return U

def _local_search_gaps(U, primes, max_iterations=500, num_candidates=20):
    """
    Local search on gap sequences.
    Tries to replace individual gaps with better candidates from primes or nearby values.
    """
    if len(U) < 2:
        return U, _score_set_fast(U)
    
    current_gaps = [U[i+1] - U[i] for i in range(len(U)-1)]
    current_score = _score_set_fast(U)
    
    # Create a pool of candidate gap values
    # Include primes near each current gap and some random primes
    candidate_pool = set()
    for g in current_gaps:
        # Add primes within a range of current gap
        for p in primes:
            if abs(p - g) < max(10, g // 5):
                candidate_pool.add(p)
        # Add some nearby integer values
        for delta in range(-3, 4):
            if g + delta > 0:
                candidate_pool.add(g + delta)
    
    # Also add a broader set of primes for diversity
    candidate_pool.update(primes[:num_candidates])
    candidate_pool = list(candidate_pool)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try modifying each gap
        for i in range(len(current_gaps)):
            old_g = current_gaps[i]
            best_candidate = old_g
            best_score = current_score
            
            # Shuffle candidates for diversity
            random.shuffle(candidate_pool)
            
            # Try a subset of candidates
            for new_g in candidate_pool[:30]:
                if new_g == old_g or new_g <= 0:
                    continue
                
                new_gaps = current_gaps[:]
                new_gaps[i] = new_g
                
                # Reconstruct U
                new_U = [0]
                cumsum = 0
                for g in new_gaps:
                    cumsum += g
                    new_U.append(cumsum)
                
                # Check for duplicates
                if len(set(new_U)) != len(new_U):
                    continue
                
                new_score = _score_set_fast(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = new_g
            
            if best_candidate != old_g:
                current_gaps[i] = best_candidate
                current_score = best_score
                improved = True
        
        # Try adding a new element
        if len(current_gaps) < 60:
            best_add_score = current_score
            best_add_gap = None
            
            for p in primes[:30]:
                new_gaps = current_gaps + [p]
                new_U = [0]
                cumsum = 0
                for g in new_gaps:
                    cumsum += g
                    new_U.append(cumsum)
                
                if len(set(new_U)) != len(new_U):
                    continue
                
                new_score = _score_set_fast(new_U)
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add_gap = p
            
            if best_add_gap is not None:
                current_gaps.append(best_add_gap)
                current_score = best_add_score
                improved = True
        
        # Try removing the last element
        if len(current_gaps) > 2:
            new_gaps = current_gaps[:-1]
            new_U = [0]
            cumsum = 0
            for g in new_gaps:
                cumsum += g
                new_U.append(cumsum)
            
            new_score = _score_set_fast(new_U)
            if new_score > current_score:
                current_gaps = new_gaps
                current_score = new_score
                improved = True
        
        if not improved:
            break
    
    # Reconstruct final U
    final_U = [0]
    cumsum = 0
    for g in current_gaps:
        cumsum += g
        final_U.append(cumsum)
    
    return final_U, current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses multiple prime-based constructions with local search optimization.
    """
    # Generate primes up to a reasonable limit
    primes = _generate_primes(1000)
    
    best_U = [0]
    best_score = 0.0
    
    # Try different sizes and prime selections
    for k in [10, 15, 20, 25, 30]:
        # Use first k primes
        U_candidate = _construct_from_primes(primes, k)
        U_candidate, score = _local_search_gaps(U_candidate, primes, max_iterations=200, num_candidates=15)
        
        if score > best_score:
            best_score = score
            best_U = U_candidate
    
    # Also try with shifted prime indices for diversity
    for offset in [5, 10, 20]:
        for k in [15, 20, 25]:
            if offset + k < len(primes):
                shifted_primes = primes[offset:offset+k]
                U_candidate = [0]
                cumulative = 0
                for p in shifted_primes:
                    cumulative += p
                    U_candidate.append(cumulative)
                
                U_candidate, score = _local_search_gaps(U_candidate, primes, max_iterations=200, num_candidates=15)
                
                if score > best_score:
                    best_score = score
                    best_U = U_candidate
    
    return best_U
# EVOLVE-BLOCK-END
