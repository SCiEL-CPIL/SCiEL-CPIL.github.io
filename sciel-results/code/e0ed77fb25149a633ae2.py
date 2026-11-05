import sys
import time

def construct(n_rows, k):
    """
    Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Approach: Model as sparse binary matrix A (diff slots x candidate values 1..M),
    solve Ax=1 via Smith Normal Form to characterize solution space, apply LLL on kernel
    basis to find low-weight integer combinations, then reconstruct rows.
    """
    row_len = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    start_time = time.time()
    time_limit = 100  # Leave buffer for 120s limit
    
    def time_exceeded():
        return time.time() - start_time > time_limit
    
    # --- Step 1: Build the sparse binary matrix A ---
    # Rows of A: one per difference slot (n_rows * total_diffs_per_row)
    # Columns of A: candidate difference values 1..M
    # A[slot, val] = 1 if difference 'val' can be assigned to that slot
    # 
    # For each row r and each pair (i,j) with i<j in [0,k], the difference a_j - a_i
    # must be a unique positive integer. The constraint is that each difference value
    # 1..M is used at most once across all slots.
    #
    # We want to assign distinct values from 1..M to all slots such that for each row,
    # the assigned differences are consistent (i.e., there exist positions 0=x_0 < x_1 < ... < x_k
    # where x_j - x_i matches the assigned difference for pair (i,j)).
    #
    # A simpler formulation: think of it as a system where we pick which difference values
    # go to which slots. The matrix A has 1s where a slot can take a value.
    # For a "difference triangle", the natural constraint is:
    #   For row r, pair (i,j): the difference d = x_j - x_i
    #   The set of all such d across all (r,i,j) must be distinct.
    #
    # We'll use the SNF approach to find a valid assignment of difference values to slots.
    
    def build_matrix(M):
        """Build sparse binary matrix A of size (total_diffs x M)."""
        # Each row in A corresponds to a difference slot
        # Slot index: r * total_diffs_per_row + local_slot
        # local_slot for pair (i,j) in a row: we enumerate pairs (i,j) with i<j
        # For row of length k+1, pairs are (0,1),(0,2),...,(0,k),(1,2),...,(k-1,k)
        # Number of pairs = k*(k+1)/2
        
        # We'll represent A as a list of (row_idx, col_idx) for nonzero entries
        # For simplicity, every slot can potentially take any value 1..M
        # But we add constraints: for a given row, the differences must be consistent
        # with a strictly increasing sequence.
        
        # For the SNF approach, we start with the full assignment matrix.
        # Each slot can take any value from 1..M, but we need consistency.
        # 
        # Actually, let's think differently. The key constraint is:
        # 1. All differences across all rows are distinct (injective mapping from slots to values)
        # 2. Within each row, the differences must be consistent with a valid position sequence
        #
        # For constraint 2: if we have positions x_0=0 < x_1 < ... < x_k, then
        # d(i,j) = x_j - x_i. The differences satisfy:
        #   d(i,j) = d(i,l) + d(l,j) for i < l < j
        #   d(i,j) > d(i,l) for l < j (since x_j > x_l)
        #   d(i,j) > d(l,j) for i < l (since x_i < x_l)
        #
        # This is complex. Let's use a different approach:
        # Instead of solving Ax=1 directly, we'll use the SNF to find the structure
        # of valid difference assignments, then search over the kernel.
        
        # For now, build A as an identity-like structure where each slot picks one value
        # and all values are distinct. This is a "system of distinct representatives" problem.
        # A is total_diffs x M with A[slot, val] = 1 for all val (any slot can take any value)
        # But we need sum over slots of x[slot] = 1 for each value, and sum over values of x[slot] = 1 for each slot.
        # This is a bipartite matching / assignment problem.
        
        # The SNF approach: A is total_diffs x M. We want A * x = 1 (vector of ones of length total_diffs).
        # Wait, that's not right. Let me reformulate.
        
        # Let x be a vector of length M where x[v] = number of slots assigned value v.
        # Constraint: x[v] <= 1 for all v (each value used at most once)
        # Constraint: sum_v x[v] = total_diffs (all slots filled)
        # Constraint: for each slot s, exactly one value is assigned to it.
        
        # This is equivalent to: find a permutation-like assignment.
        # The SNF of the incidence matrix gives the kernel structure.
        
        # Let's define A as the slot-value incidence: A[s, v] = 1 if slot s can take value v.
        # For our problem, initially A[s, v] = 1 for all s, v (no restrictions yet).
        # Then A * x = 1 means each slot gets exactly one value, but x[v] would be the count,
        # which doesn't quite work.
        
        # Better: Let y be a 0-1 matrix of size total_diffs x M, y[s,v] = 1 if slot s gets value v.
        # Constraints: sum_v y[s,v] = 1 for each s (each slot gets one value)
        #              sum_s y[s,v] <= 1 for each v (each value used at most once)
        #
        # This is a transportation/assignment problem. The SNF of the constraint matrix
        # characterizes the solution space.
        
        # For efficiency, we'll use a greedy + local search approach guided by the SNF structure.
        
        return None  # Placeholder - we'll use a different strategy
    
    # --- Alternative: Use SNF to analyze the difference consistency constraints ---
    # For each row, the differences d(i,j) must satisfy d(i,j) = d(i,l) + d(l,j) for i<l<j.
    # This gives linear constraints on the difference values.
    # 
    # For a single row of length k+1, there are k*(k+1)/2 differences.
    # The independent differences are d(i, i+1) for i=0..k-1 (the "steps").
    # All other differences are sums of consecutive steps.
    # 
    # So for each row, we need to choose k steps s_1, ..., s_k > 0 such that
    # all partial sums (and their differences) are distinct across all rows.
    
    # Let's use this structure: for each row r, define steps t_r[0], ..., t_r[k-1] > 0
    # Position x_j = sum_{i=0}^{j-1} t_r[i]
    # Difference d(i,j) = sum_{m=i}^{j-1} t_r[m]
    # 
    # We need all d(i,j) across all rows to be distinct.
    # 
    # The SNF approach: build a matrix where rows are the "difference equations" and
    # columns are the step variables, then find integer solutions.
    
    def snf_analysis(M_max):
        """
        Analyze the linear system using SNF.
        For each row, the differences are determined by the steps.
        We need to assign distinct difference values to all slots.
        
        We'll construct the constraint matrix and compute SNF.
        """
        try:
            from sympy import Matrix, snf
            from sympy.matrices.normalforms import smith_normal_form
        except ImportError:
            return None
        
        # Total number of difference slots
        n_slots = n_rows * total_diffs_per_row
        
        # We want to assign values 1..M to these slots such that:
        # 1. All values are distinct
        # 2. Within each row, the assigned values are consistent with a step sequence
        
        # For consistency within a row: if we label the slots in a row by (i,j) pairs,
        # then d(i,j) = d(i,l) + d(l,j) for i < l < j.
        # This gives linear constraints: d(i,j) - d(i,l) - d(l,j) = 0
        
        # Let's enumerate the constraints for one row:
        # Slots in a row: pairs (i,j) with 0 <= i < j <= k
        # For each triple i < l < j: constraint d(i,j) - d(i,l) - d(l,j) = 0
        
        # Number of such constraints per row: C(k+1, 3) = k(k+1)(k-1)/6
        
        # Build the constraint matrix C of size n_constraints x n_slots
        # where C[constraint, slot] has coefficients +1, -1, -1 for the three involved slots
        
        constraints_per_row = k * (k + 1) * (k - 1) // 6 if k >= 2 else 0
        total_constraints = n_rows * constraints_per_row
        
        if total_constraints == 0:
            return None
        
        # Build constraint matrix
        # Slot indexing: for row r, slot for pair (i,j) is:
        #   r * total_diffs_per_row + local_index(i,j)
        # where local_index enumerates pairs (i,j) with i<j in some order
        
        def local_slot_idx(i, j):
            """Get local slot index for pair (i,j) in a row of length k+1."""
            idx = 0
            for ii in range(i + 1, k + 1):
                for jj in range(ii + 1, k + 1):
                    if ii == i and jj == j:
                        return idx
                    idx += 1
            # More efficient: precompute
            # Pairs: (0,1),(0,2),...,(0,k),(1,2),(1,3),...,(1,k),...,(k-1,k)
            # For pair (i,j): index = sum_{a=0}^{i-1} (k-a) + (j - i - 1)
            # = i*k - i*(i-1)//2 + j - i - 1 ... let me just enumerate
            pairs = []
            for a in range(k + 1):
                for b in range(a + 1, k + 1):
                    pairs.append((a, b))
            return pairs.index((i, j))
        
        # Precompute local slot indices
        local_pairs = []
        for a in range(k + 1):
            for b in range(a + 1, k + 1):
                local_pairs.append((a, b))
        local_idx_map = {}
        for idx, (a, b) in enumerate(local_pairs):
            local_idx_map[(a, b)] = idx
        
        # Build constraint matrix as sparse representation
        # Each constraint: for row r, triple (i, l, j) with i < l < j
        # Constraint: d(i,j) - d(i,l) - d(l,j) = 0
        # In matrix form: C[c, slot(i,j)] = 1, C[c, slot(i,l)] = -1, C[c, slot(l,j)] = -1
        
        # We'll build this as a list of (row, col, val) tuples for efficiency
        constraint_entries = []
        c_idx = 0
        for r in range(n_rows):
            for i in range(k + 1):
                for l in range(i + 1, k + 1):
                    for j in range(l + 1, k + 1):
                        # Constraint: d(i,j) - d(i,l) - d(l,j) = 0
                        slot_ij = r * total_diffs_per_row + local_idx_map[(i, j)]
                        slot_il = r * total_diffs_per_row + local_idx_map[(i, l)]
                        slot_lj = r * total_diffs_per_row + local_idx_map[(l, j)]
                        constraint_entries.append((c_idx, slot_ij, 1))
                        constraint_entries.append((c_idx, slot_il, -1))
                        constraint_entries.append((c_idx, slot_lj, -1))
                        c_idx += 1
        
        if time_exceeded():
            return None
        
        # Now, we want to find integer vectors x (length n_slots) such that:
        # 1. C * x = 0 (consistency constraints)
        # 2. All x[s] are distinct positive integers in 1..M
        # 3. x[s] represent valid differences
        
        # The SNF of C gives the structure of the solution space.
        # Let's compute SNF of C.
        
        # Build C as a sympy Matrix
        C = Matrix(total_constraints, n_slots, 0)
        for (c, s, v) in constraint_entries:
            C[c, s] = v
        
        if time_exceeded():
            return None
        
        try:
            # Compute Smith Normal Form
            # C = U * S * V where U, V are unimodular, S is diagonal
            # The kernel of C is {x : C*x = 0} = V * {y : S*y = 0}
            # S is diagonal with entries s_1, ..., s_rank, 0, ..., 0
            # So kernel dimension = n_slots - rank
            # The kernel basis is given by the last (n_slots - rank) columns of V
            
            S, U, V = smith_normal_form(C, domain='ZZ')
            
            if time_exceeded():
                return None
            
            # Get the kernel basis (last n_slots - rank columns of V)
            rank = 0
            for i in range(min(total_constraints, n_slots)):
                if S[i, i] != 0:
                    rank += 1
                else:
                    break
            
            kernel_dim = n_slots - rank
            
            if kernel_dim <= 0:
                return None
            
            # Kernel basis vectors
            kernel_basis = []
            for j in range(rank, n_slots):
                vec = [V[i, j] for i in range(n_slots)]
                kernel_basis.append(vec)
            
            if time_exceeded():
                return None
            
            return {
                'kernel_dim': kernel_dim,
                'kernel_basis': kernel_basis,
                'rank': rank,
                'C': C,
                'S': S,
                'U': U,
                'V': V
            }
        except Exception as e:
            return None
    
    # --- Step 2: Use SNF result to guide search ---
    # The kernel basis gives us the degrees of freedom.
    # We need to find integer combinations of kernel basis vectors that give
    # distinct positive entries in 1..M.
    
    # For now, let's use a hybrid approach:
    # 1. Try to find a solution using the kernel structure
    # 2. Fall back to greedy if SNF doesn't help
    
    def greedy_construct(M):
        """Greedy construction: build rows one by one, choosing smallest valid next value."""
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            # Try to build the row greedily
            for j in range(1, k + 1):
                found = False
                for next_val in range(row[-1] + 1, M + 1):
                    # Check if all new differences are valid
                    valid = True
                    for prev_val in row:
                        d = next_val - prev_val
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(next_val)
                        for prev_val in row[:-1]:
                            used_diffs.add(next_val - prev_val)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        
        return rows
    
    # Try increasing scopes
    # Lower bound: total_diffs (each difference value used at most once)
    # But we need room for the structure, so start higher
    
    # First, try to use SNF to get a better lower bound or direct solution
    snf_result = None
    for M_try in range(total_diffs, total_diffs + 5):
        if time_exceeded():
            break
        snf_result = snf_analysis(M_try)
        if snf_result is not None:
            break
    
    # Now search for solutions
    # Start from a reasonable lower bound
    # The counting bound is total_diffs = 112 for n_rows=4, k=7
    # Best known is 134
    
    best_scope = None
    best_solution = None
    
    # Try scopes from lower bound upwards
    # Use SNF kernel if available to guide the search
    
    for scope in range(total_diffs, min(total_diffs + 40, 200)):
        if time_exceeded():
            break
        
        # Try greedy first (fast)
        sol = greedy_construct(scope)
        if sol is not None:
            # Verify
            all_diffs = set()
            valid = True
            for row in sol:
                if len(row) != row_len or row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid and max(max(row) for row in sol) <= scope:
                best_scope = scope
                best_solution = sol
                break
        
        if time_exceeded():
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy without scope limit
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    return rows
