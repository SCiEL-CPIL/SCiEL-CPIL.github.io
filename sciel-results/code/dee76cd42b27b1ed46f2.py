"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _squarefree_kernel(n):
    """Return the squarefree part of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # Precompute squarefree kernels for all numbers in (210, 10210]
    # and collect the set of relevant primes
    kernels = {}
    all_primes = set()
    for n in range(211, MAX_VAL + 1):
        k = _squarefree_kernel(n)
        kernels[n] = k
        # Factor k to get primes
        temp = k
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                all_primes.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            all_primes.add(temp)
    
    primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Target: primes 2,3,5,7 need odd count, others even
    target = [0] * num_primes
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target[prime_to_idx[p]] = 1
    
    # For each number, create its vector
    num_vectors = {}
    for n in range(211, MAX_VAL + 1):
        k = kernels[n]
        vec = [0] * num_primes
        temp = k
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                vec[prime_to_idx[d]] = 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            vec[prime_to_idx[temp]] = 1
        num_vectors[n] = vec
    
    # Search by increasing upper bound
    # For a given upper bound M, we have numbers 211..M
    # We need to find a subset (size 1..100) whose vector sum = target
    
    # Use Gaussian elimination to check feasibility and find solution
    def find_solution(max_val):
        """Find a subset of numbers in (210, max_val] with XOR sum = target, size <= 100."""
        if time.time() > deadline:
            return None
        
        # Collect vectors for numbers 211..max_val
        vecs = []
        nums = []
        for n in range(211, max_val + 1):
            vecs.append(num_vectors[n])
            nums.append(n)
        
        # Gaussian elimination to find if target is in the span
        # and to find a particular solution
        # We'll do row-reduction on the vectors
        
        m = len(vecs)
        if m == 0:
            return None
        
        # Build augmented matrix [vecs | target]
        # Actually, we want to solve: sum of subset = target
        # This is: find x in {0,1}^m such that A*x = target (mod 2)
        
        # Use Gaussian elimination
        # rows are the vectors, we'll reduce them
        rows = [v[:] for v in vecs]
        n_cols = num_primes
        
        # Forward elimination
        rank = 0
        pivot_cols = []
        for col in range(n_cols):
            if rank >= m:
                break
            # Find pivot
            pivot = -1
            for i in range(rank, m):
                if rows[i][col] == 1:
                    pivot = i
                    break
            if pivot == -1:
                continue
            # Swap
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            # Eliminate
            for i in range(m):
                if i != rank and rows[i][col] == 1:
                    for j in range(col, n_cols):
                        rows[i][j] ^= rows[rank][j]
            pivot_cols.append(col)
            rank += 1
        
        # Check if target is in the column space
        # Reduce target using the same pivots
        t = target[:]
        for i in range(rank):
            col = pivot_cols[i]
            if t[col] == 1:
                for j in range(col, n_cols):
                    t[j] ^= rows[i][j]
        
        if any(t):
            return None  # Not solvable
        
        # Back-substitute to find a solution
        # The rows after elimination: for each pivot row i, it has a 1 at pivot_cols[i]
        # and the target was reduced to 0, so the solution is:
        # x[pivot_cols[i]] = (original_target at that position after considering other pivots)
        # Actually, let's redo: we want A*x = b where b = target
        # After elimination, we have R*x = b' where b' = 0 (since solvable)
        # Free variables can be 0, pivot variables determined
        
        # Let me redo this more carefully
        # Start over with proper Gaussian elimination
        
        # Build matrix A (m x n_cols) and b = target
        A = [v[:] for v in vecs]
        b = target[:]
        
        # Forward elimination with backtracking for solution
        pivot_row = 0
        pivot_col_map = {}  # col -> row
        
        for col in range(n_cols):
            if pivot_row >= m:
                break
            # Find row with 1 in this column
            found = -1
            for i in range(pivot_row, m):
                if A[i][col] == 1:
                    found = i
                    break
            if found == -1:
                continue
            # Swap
            A[pivot_row], A[found] = A[found], A[pivot_row]
            b[pivot_row], b[found] = b[found], b[pivot_row]
            # Eliminate this column from all other rows
            for i in range(m):
                if i != pivot_row and A[i][col] == 1:
                    for j in range(col, n_cols):
                        A[i][j] ^= A[pivot_row][j]
                    b[i] ^= b[pivot_row]
            pivot_col_map[col] = pivot_row
            pivot_row += 1
        
        # Check consistency
        for i in range(pivot_row, m):
            if b[i] == 1:
                return None  # Inconsistent
        
        # Back-substitute: free variables = 0, pivot variables from b
        x = [0] * m
        for col, row in pivot_col_map.items():
            x[row] = b[row]
        
        # Verify
        result_vec = [0] * n_cols
        for i in range(m):
            if x[i] == 1:
                for j in range(n_cols):
                    result_vec[j] ^= vecs[i][j]
        
        if result_vec != target:
            # Try to fix - maybe my elimination was wrong
            # Let's just return the selected numbers and verify externally
            pass
        
        selected = [nums[i] for i in range(m) if x[i] == 1]
        if len(selected) == 0:
            return None
        if len(selected) > MAX_SIZE:
            return None
        
        # Verify the solution
        prod_kernels = 1
        for n in selected:
            prod_kernels *= kernels[n]
        # Check if prod_kernels * 210 is a perfect square
        val = prod_kernels * 210
        s = isqrt(val)
        if s * s != val:
            return None
        
        return selected
    
    # Binary search or linear search for minimum max_val
    best_solution = None
    best_max = MAX_VAL + 1
    
    # Try increasing upper bounds
    # Start from small values and increase
    # But first, let's try a few strategic values
    
    # The minimum possible max is 211 (if 211 alone works, but 211*210 = 211*2*3*5*7, not a square)
    # Let's search from 211 upward
    
    # For efficiency, try in increasing order but skip ahead
    current_max = 211
    while current_max <= MAX_VAL:
        if time.time() > deadline:
            break
        
        sol = find_solution(current_max)
        if sol is not None:
            best_solution = sol
            best_max = current_max
            break
        
        current_max += 1
    
    if best_solution is None:
        # Fallback: use the seed solution
        return [4 * 210]
    
    # Ensure types are correct
    result = [int(x) for x in best_solution]
    assert isinstance(result, list)
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result)
    assert len(result) >= 1 and len(result) <= 100
    assert all(x > 210 and x <= 10210 for x in result)
    assert len(set(result)) == len(result)  # distinct
    
    return result
# EVOLVE-BLOCK-END
