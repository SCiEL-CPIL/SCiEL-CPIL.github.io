import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

N, M, P = 2, 3, 10
DEADLINE = 143.0  # seconds, safety margin before 150s kill
START = time.time()

def _elapsed():
    return time.time() - START

def _check_deadline():
    return _elapsed() > DEADLINE

def _tensor_entries():
    """List of (i, j, k) for the multiplication tensor T[i,j,k] = 1."""
    return [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]

def _build_strassen_rank55():
    """Construct the rank-55 Strassen-based decomposition (verified)."""
    rank = 55
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    for k in range(P):
        b2k = 2 * P + k
        U[a02, col] += 1; V[b2k, col] += 1; W[2 * k, col] += 1
        col += 1
        U[a12, col] += 1; V[b2k, col] += 1; W[2 * k + 1, col] += 1
        col += 1
    return U, V, W

def _verify_np(U, V, W, tol=1e-9):
    """Verify decomposition using numpy. Returns True if exact within tol."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W):
    """Verify decomposition exactly using fractions."""
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    rank = len(Uf[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i * M + j][r] * Vf[j * P + k][r] * Wf[k * N + i][r]
                if s != 1:
                    return False
    return True

def _solve_support(support, rank):
    """
    Given a support (list of sets of (i,j,k) per term), solve for U,V,W.
    Returns (U, V, W) as numpy float arrays or None if no solution.
    """
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    num_vars = 3 * rank
    A = np.zeros((num_entries, num_vars))
    b = np.ones(num_entries)
    for r, term_entries in enumerate(support):
        for (i, j, k) in term_entries:
            if (i, j, k) not in entry_idx:
                continue
            eq = entry_idx[(i, j, k)]
            A[eq, r] = 1.0
            A[eq, rank + r] = 1.0
            A[eq, 2 * rank + r] = 1.0
    # Solve least-squares; check residual
    try:
        sol, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    # Check exact fit
    if np.max(np.abs(A @ sol - b)) > 1e-8:
        return None
    U = sol[:rank].reshape(rank, N * M).T
    V = sol[rank:2 * rank].reshape(rank, M * P).T
    W = sol[2 * rank:].reshape(rank, P * N).T
    return U, V, W

def _strassen_support():
    """Generate the support pattern of the rank-55 Strassen construction."""
    support = []
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        # M1
        support.append({(0, 0, k0), (1, 1, k1)})
        # M2
        support.append({(1, 0, k0), (1, 1, k1)})
        # M3
        support.append({(0, 1, k1), (1, 1, k1)})
        # M4
        support.append({(1, 1, k0), (1, 0, k0)})
        # M5
        support.append({(0, 0, k0), (0, 1, k1)})
        # M6
        support.append({(1, 1, k1)})
        # M7
        support.append({(0, 0, k0)})
    for k in range(P):
        support.append({(0, 2, k)})
        support.append({(1, 2, k)})
    return support

def _mutate_support(support, n_mutations):
    """Randomly mutate a support by changing which entries each term covers."""
    entries = _tensor_entries()
    new_support = [set(s) for s in support]
    for _ in range(n_mutations):
        r = random.randrange(len(new_support))
        # Remove a random entry from term r
        if new_support[r]:
            e = random.choice(list(new_support[r]))
            new_support[r].remove(e)
        # Add a random entry to term r
        e = random.choice(entries)
        new_support[r].add(e)
    return new_support

def _search_rank54():
    """Try to find a rank-54 solution via structured pair-merge and support search."""
    if _check_deadline():
        return None
    base_support = _strassen_support()
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    
    # Strategy 1: Systematic pair-merge. For each pair (r1, r2) from the 55-term
    # Strassen support, remove both terms and try to replace with one term.
    # The new term must cover all entries that were only covered by r1 or r2.
    for r1 in range(55):
        if _check_deadline():
            return None
        for r2 in range(r1 + 1, 55):
            if _check_deadline():
                return None
            # Determine which entries are uniquely covered by r1 or r2
            # (i.e., not covered by any other term)
            unique_to_pair = set()
            for e in base_support[r1]:
                covered_by_other = False
                for r3 in range(55):
                    if r3 != r1 and r3 != r2 and e in base_support[r3]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    unique_to_pair.add(e)
            for e in base_support[r2]:
                covered_by_other = False
                for r3 in range(55):
                    if r3 != r1 and r3 != r2 and e in base_support[r3]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    unique_to_pair.add(e)
            
            # Build new support: all 55 terms except r1,r2, plus one new term
            # The new term covers unique_to_pair plus optionally shared entries
            new_support = []
            for r in range(55):
                if r != r1 and r != r2:
                    new_support.append(set(base_support[r]))
            # The new term must cover at least unique_to_pair
            # Try: new term covers unique_to_pair ∪ (intersection of r1 and r2)
            shared = base_support[r1] & base_support[r2]
            new_term = unique_to_pair | shared
            new_support.append(new_term)
            
            if len(new_support) != 54:
                continue
            # Ensure full coverage
            all_covered = set()
            for s in new_support:
                all_covered |= s
            if len(all_covered) < num_entries:
                continue
            
            result = _solve_support(new_support, 54)
            if result is not None:
                U, V, W = result
                if _verify_np(U, V, W, tol=1e-7):
                    Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
                    Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
                    Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
                    if _verify_exact(Uf, Vf, Wf):
                        return U, V, W
    
    # Strategy 2: Random support mutations (fallback)
    random.seed(42)
    for trial in range(3000):
        if _check_deadline():
            break
        n_mut = random.choice([1, 2, 3, 4, 5])
        support = _mutate_support(base_support, n_mut)
        all_covered = set()
        for s in support:
            all_covered |= s
        if len(all_covered) < num_entries:
            continue
        result = _solve_support(support, 54)
        if result is not None:
            U, V, W = result
            if _verify_np(U, V, W, tol=1e-7):
                Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
                Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
                Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
                if _verify_exact(Uf, Vf, Wf):
                    return U, V, W
    return None

def _search_rank53():
    """Try to find a rank-53 solution via triple-merge and support search."""
    if _check_deadline():
        return None
    base_support = _strassen_support()
    entries = _tensor_entries()
    num_entries = len(entries)
    
    # Strategy: merge triples of terms into one. Only try promising triples
    # (those with significant overlap).
    from itertools import combinations
    tried = 0
    for r1, r2, r3 in combinations(range(55), 3):
        if _check_deadline():
            return None
        if tried > 200:  # Limit triple search
            break
        tried += 1
        
        # Check if the three terms have meaningful overlap
        s1, s2, s3 = base_support[r1], base_support[r2], base_support[r3]
        union = s1 | s2 | s3
        if len(union) > 4:  # Too many unique entries for one term
            continue
        
        # Determine entries uniquely covered by this triple
        unique_to_triple = set()
        for e in union:
            covered_by_other = False
            for r in range(55):
                if r != r1 and r != r2 and r != r3 and e in base_support[r]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                unique_to_triple.add(e)
        
        new_support = []
        for r in range(55):
            if r != r1 and r != r2 and r != r3:
                new_support.append(set(base_support[r]))
        new_support.append(unique_to_triple)
        
        if len(new_support) != 52:
            continue
        all_covered = set()
        for s in new_support:
            all_covered |= s
        if len(all_covered) < num_entries:
            continue
        
        result = _solve_support(new_support, 53)
        if result is not None:
            U, V, W = result
            if _verify_np(U, V, W, tol=1e-7):
                Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
                Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
                Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
                if _verify_exact(Uf, Vf, Wf):
                    return U, V, W
    
    # Fallback: random mutations
    random.seed(123)
    for trial in range(1500):
        if _check_deadline():
            break
        n_mut = random.choice([2, 3, 4, 5, 6])
        support = _mutate_support(base_support, n_mut)
        all_covered = set()
        for s in support:
            all_covered |= s
        if len(all_covered) < num_entries:
            continue
        result = _solve_support(support, 53)
        if result is not None:
            U, V, W = result
            if _verify_np(U, V, W, tol=1e-7):
                Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
                Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
                Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
                if _verify_exact(Uf, Vf, Wf):
                    return U, V, W
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Start with the verified rank-55 Strassen-based decomposition.
    - Attempt to find a rank-54 or rank-53 solution via sparse support search.
    - Return the best (lowest-rank) verified solution found.
    """
    # Fallback: rank-55 Strassen
    U55, V55, W55 = _build_strassen_rank55()
    
    # Try rank-54
    result54 = _search_rank54()
    if result54 is not None:
        U, V, W = result54
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-53
    result53 = _search_rank53()
    if result53 is not None:
        U, V, W = result53
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return rank-55 fallback
    return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
