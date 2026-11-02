import numpy as np
import time
import itertools
import random
from collections import defaultdict

DIMENSIONS = [2, 4, 7]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[a,b,c] = 1 iff c = k*n+i where a=i*m+j, b=j*p+k."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = _build_target_tensor(n, m, p)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _verify_fast(U, V, W, Texp, tol=1e-9):
    """Faster verification using precomputed tensor."""
    rank = U.shape[1]
    T = np.zeros(Texp.shape, dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(T - Texp)) < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block(n, m, p, i0, j0, k0):
    """Return list of (u_vec, v_vec, w_vec) for a 2x2x2 Strassen block."""
    terms = []
    a_idx = i0 * m + j0
    b_idx = i0 * m + (j0 + 1)
    c_idx = (i0 + 1) * m + j0
    d_idx = (i0 + 1) * m + (j0 + 1)
    e_idx = j0 * p + k0
    f_idx = j0 * p + (k0 + 1)
    g_idx = (j0 + 1) * p + k0
    h_idx = (j0 + 1) * p + (k0 + 1)
    c00 = k0 * n + i0
    c01 = (k0 + 1) * n + i0
    c10 = k0 * n + (i0 + 1)
    c11 = (k0 + 1) * n + (i0 + 1)
    
    def mk_u(*idxs):
        v = {}
        for idx in idxs:
            v[idx] = v.get(idx, 0) + 1
        return v
    def mk_v(*idxs):
        v = {}
        for idx in idxs:
            v[idx] = v.get(idx, 0) + 1
        return v
    def mk_w(*idxs):
        v = {}
        for idx in idxs:
            v[idx] = v.get(idx, 0) + 1
        return v
    
    # M1 = (a+d)(e+h) -> c00 + c11
    terms.append((mk_u(a_idx, d_idx), mk_v(e_idx, h_idx), mk_w(c00, c11)))
    # M2 = (c+d)e -> c10 - c11
    terms.append((mk_u(c_idx, d_idx), mk_v(e_idx), {c10: 1, c11: -1}))
    # M3 = a(f-h) -> c01 + c11
    terms.append((mk_u(a_idx), {f_idx: 1, h_idx: -1}, mk_w(c01, c11)))
    # M4 = d(g-e) -> c00 + c10
    terms.append((mk_u(d_idx), {g_idx: 1, e_idx: -1}, mk_w(c00, c10)))
    # M5 = (a+b)h -> -c00 + c01
    terms.append((mk_u(a_idx, b_idx), mk_v(h_idx), {c00: -1, c01: 1}))
    # M6 = (c-a)(e+f) -> c11
    terms.append(({c_idx: 1, a_idx: -1}, mk_v(e_idx, f_idx), mk_w(c11)))
    # M7 = (b-d)(g+h) -> c00
    terms.append(({b_idx: 1, d_idx: -1}, mk_v(g_idx, h_idx), mk_w(c00)))
    
    return terms

def _schoolbook_221_block(n, m, p, i0, j0, k0):
    """Return list of (u_vec, v_vec, w_vec) for a 2x2x1 schoolbook block."""
    terms = []
    a_idx = i0 * m + j0
    b_idx = i0 * m + (j0 + 1)
    c_idx = (i0 + 1) * m + j0
    d_idx = (i0 + 1) * m + (j0 + 1)
    e_idx = j0 * p + k0
    g_idx = (j0 + 1) * p + k0
    c00 = k0 * n + i0
    c10 = k0 * n + (i0 + 1)
    
    terms.append(({a_idx: 1}, {e_idx: 1}, {c00: 1}))
    terms.append(({b_idx: 1}, {g_idx: 1}, {c00: 1}))
    terms.append(({c_idx: 1}, {e_idx: 1}, {c10: 1}))
    terms.append(({d_idx: 1}, {g_idx: 1}, {c10: 1}))
    
    return terms

def _strassen_221_block(n, m, p, i0, j0, k0):
    """Return list of (u_vec, v_vec, w_vec) for a 2x2x1 using Strassen-like 4-term decomposition."""
    # For 2x2x1: C = A*B where A is 2x2, B is 2x1
    # C[0] = a*e + b*g, C[1] = c*e + d*g
    # Use: P1=(a+d)*e, P2=(c+d)*e... no, let's use 4 terms (schoolbook is optimal for 2x2x1)
    # Actually 2x2x1 has rank 4 (schoolbook is optimal)
    return _schoolbook_221_block(n, m, p, i0, j0, k0)

def _build_initial_decomposition(n, m, p):
    """Build the initial Strassen-based decomposition with rank 50."""
    all_terms = []
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5), and singleton (6)
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            all_terms.extend(_strassen_block(n, m, p, 0, j_start, k_start))
        all_terms.extend(_schoolbook_221_block(n, m, p, 0, j_start, 6))
    return all_terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of (u_dict, v_dict, w_dict) to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx, r] = val
        for idx, val in v.items():
            V[idx, r] = val
        for idx, val in w.items():
            W[idx, r] = val
    return U, V, W

def _term_contribution(terms, n, m, p):
    """Compute the sum tensor from a list of terms."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for u, v, w in terms:
        for a, ua in u.items():
            for b, vb in v.items():
                for c, wc in w.items():
                    T[a, b, c] += ua * vb * wc
    return T

def _find_merge_candidates(terms, Texp, n, m, p, max_candidates=2000):
    """
    Find pairs of terms that can be merged (replaced by a single term) while preserving the tensor.
    Two terms (u1,v1,w1) and (u2,v2,w2) can be merged into (u,v,w) if:
    u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w
    
    We search for this by trying all possible rank-1 tensors that equal the sum.
    """
    candidates = []
    Texp_sparse = {}
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if Texp[a,b,c] != 0:
                    Texp_sparse[(a,b,c)] = Texp[a,b,c]
    
    for i in range(len(terms)):
        u1, v1, w1 = terms[i]
        for j in range(i+1, len(terms)):
            u2, v2, w2 = terms[j]
            
            # The sum tensor S = u1⊗v1⊗w1 + u2⊗v2⊗w2
            # S must be rank-1 for a merge to be possible
            # Build S as a sparse dict
            S = defaultdict(int)
            for a, ua in u1.items():
                for b, vb in v1.items():
                    for c, wc in w1.items():
                        S[(a,b,c)] += ua * vb * wc
            for a, ua in u2.items():
                for b, vb in v2.items():
                    for c, wc in w2.items():
                        S[(a,b,c)] += ua * vb * wc
            
            # Remove zeros
            S = {k: v for k, v in S.items() if v != 0}
            if not S:
                continue
            
            # Check if S is rank-1: S(a,b,c) = u(a)*v(b)*w(c) for some u,v,w
            # Pick a reference entry
            (a0, b0, c0), val0 = next(iter(S.items()))
            if val0 == 0:
                continue
            
            # Try to determine u, v, w
            # u(a0)*v(b0)*w(c0) = val0
            # For all (a,b,c) in S: u(a)*v(b)*w(c) = S(a,b,c)
            
            # Strategy: fix u(a0)=1, then v(b0)*w(c0) = val0
            # Try different factorizations of val0
            # Since entries are small integers, try small values
            
            # Get all unique a, b, c indices
            all_a = set(a for (a,b,c) in S.keys())
            all_b = set(b for (a,b,c) in S.keys())
            all_c = set(c for (a,b,c) in S.keys())
            
            # For rank-1: S(a,b,c) / S(a0,b0,c0) = [u(a)/u(a0)] * [v(b)/v(b0)] * [w(c)/w(c0)]
            # Let alpha(a) = u(a)/u(a0), beta(b) = v(b)/v(b0), gamma(c) = w(c)/w(c0)
            # Then S(a,b,c) = val0 * alpha(a) * beta(b) * gamma(c)
            
            # We need to find alpha, beta, gamma such that for all (a,b,c):
            # S(a,b,c) = val0 * alpha(a) * beta(b) * gamma(c)
            
            # Fix gamma(c0) = 1, then alpha(a)*beta(b)*gamma(c) = S(a,b,c)/val0
            # For c=c0: alpha(a)*beta(b) = S(a,b,c0)/val0 (for all a,b where this entry exists)
            # For a=a0: beta(b)*gamma(c) = S(a0,b,c)/val0 (for all b,c)
            # For b=b0: alpha(a)*gamma(c) = S(a,b0,c)/val0 (for all a,c)
            
            # This is a system of equations. Let's try to solve it.
            # From S(a,b,c0)/val0 = alpha(a)*beta(b) for all valid (a,b)
            # From S(a0,b,c)/val0 = beta(b)*gamma(c) for all valid (b,c)
            # From S(a,b0,c)/val0 = alpha(a)*gamma(c) for all valid (a,c)
            
            # Let's try: set alpha(a0)=1 (WLOG, since we can scale)
            # Then beta(b0) = S(a0,b0,c0)/(val0*alpha(a0)*gamma(c0)) = val0/(val0*1*gamma(c0)) = 1/gamma(c0)
            # And gamma(c0) = 1 (by our choice), so beta(b0) = 1
            # Then alpha(a) = S(a,b0,c0)/(val0*beta(b0)*gamma(c0)) = S(a,b0,c0)/val0
            # And gamma(c) = S(a0,b0,c)/(val0*alpha(a0)*beta(b0)) = S(a0,b0,c)/val0
            # And beta(b) = S(a0,b,c0)/(val0*alpha(a0)*gamma(c0)) = S(a0,b,c0)/val0
            
            # Check consistency: for all (a,b,c), S(a,b,c) should equal val0 * alpha(a) * beta(b) * gamma(c)
            
            alpha = {}
            beta = {}
            gamma = {}
            alpha[a0] = 1
            gamma[c0] = 1
            beta[b0] = 1
            
            # Determine alpha(a) for all a
            for (a, b, c), val in S.items():
                if b == b0 and c == c0:
                    alpha[a] = val / val0
                elif a == a0 and c == c0:
                    beta[b] = val / val0
                elif a == a0 and b == b0:
                    gamma[c] = val / val0
            
            # Now verify
            valid = True
            for (a, b, c), val in S.items():
                expected = val0 * alpha.get(a, 0) * beta.get(b, 0) * gamma.get(c, 0)
                if abs(val - expected) > 1e-9:
                    valid = False
                    break
            
            if valid:
                # Convert to integer/half-integer check
                u_new = {a: alpha.get(a, 0) for a in all_a if alpha.get(a, 0) != 0}
                v_new = {b: beta.get(b, 0) for b in all_b if beta.get(b, 0) != 0}
                w_new = {c: gamma.get(c, 0) for c in all_c if gamma.get(c, 0) != 0}
                
                # Check if entries are integer or half-integer
                is_valid = True
                for d in [u_new, v_new, w_new]:
                    for idx, val in d.items():
                        if abs(val - round(val*2)/2) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    candidates.append((i, j, u_new, v_new, w_new))
            
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    
    return candidates

def _greedy_merge(terms, Texp, n, m, p, deadline):
    """
    Greedily merge terms to reduce rank.
    """
    terms = list(terms)
    T = _term_contribution(terms, n, m, p)
    
    while time.time() < deadline:
        # Find all merge candidates
        candidates = _find_merge_candidates(terms, Texp, n, m, p, max_candidates=500)
        
        if not candidates:
            break
        
        # Try each candidate and pick the best one (that actually reduces rank)
        best_candidate = None
        best_gain = 0
        
        for (i, j, u_new, v_new, w_new) in candidates:
            # Create new terms list with i and j replaced by the merged term
            new_terms = []
            for k, t in enumerate(terms):
                if k == i or k == j:
                    continue
                new_terms.append(t)
            new_terms.append((u_new, v_new, w_new))
            
            # Verify
            T_new = _term_contribution(new_terms, n, m, p)
            if np.array_equal(T_new, Texp):
                gain = 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = new_terms
        
        if best_candidate is None:
            break
        
        terms = best_candidate
    
    return terms

def _local_search(terms, Texp, n, m, p, deadline, max_iters=100):
    """
    Local search: try replacing one term with a different rank-1 tensor to enable further merges.
    """
    terms = list(terms)
    T = _term_contribution(terms, n, m, p)
    best_rank = len(terms)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(len(terms)):
            if time.time() > deadline:
                break
            
            u, v, w = terms[idx]
            
            # Try perturbing this term slightly
            # Get all indices involved
            all_a = set(u.keys())
            all_b = set(v.keys())
            all_c = set(w.keys())
            
            # Try changing one coefficient
            for a in list(all_a):
                for b in list(all_b):
                    for c in list(all_c):
                        # Try setting the contribution at (a,b,c) to a different value
                        # This is complex; skip for now
                        pass
        
        if not improved:
            break
    
    return terms

def _try_different_blockings(n, m, p, deadline):
    """
    Try different ways to partition the j and k dimensions into blocks.
    """
    results = []
    
    # Current: j=(0,1),(2,3); k=(0,1),(2,3),(4,5),(6)
    # Try: j=(0,1),(2,3); k=(0,1),(2,3),(4,5,6) - 3x2x3 Strassen? No, Strassen is 2x2x2
    
    # Alternative: group k as (0,1,2,3,4,5,6) differently
    # Or group j differently: (0,1,2,3) as one block? Need 4x4x? 
    
    # For now, just return the standard blocking
    terms = _build_initial_decomposition(n, m, p)
    results.append(terms)
    
    return results

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin from 150s limit
    
    # Build initial decomposition
    terms = _build_initial_decomposition(n, m, p)
    Texp = _build_target_tensor(n, m, p)
    
    # Verify initial
    U, V, W = _terms_to_matrices(terms, n, m, p)
    if not _verify_fast(U, V, W, Texp):
        return _schoolbook(n, m, p)
    
    best_terms = terms
    best_rank = len(terms)
    
    # Phase 1: Greedy merge
    if time.time() < deadline - 30:
        merged = _greedy_merge(terms, Texp, n, m, p, deadline - 30)
        if len(merged) < best_rank:
            U, V, W = _terms_to_matrices(merged, n, m, p)
            if _verify_fast(U, V, W, Texp):
                best_terms = merged
                best_rank = len(merged)
    
    # Phase 2: Try to find more merges with different strategies
    if time.time() < deadline - 15:
        # Try removing one term at a time and seeing if we can compensate
        for idx in range(len(best_terms)):
            if time.time() > deadline - 15:
                break
            
            reduced = best_terms[:idx] + best_terms[idx+1:]
            T_reduced = _term_contribution(reduced, n, m, p)
            deficit = Texp - T_reduced
            
            # Check if deficit is rank-1
            if np.any(deficit != 0):
                # Find non-zero entries
                nz = np.argwhere(deficit != 0)
                if len(nz) > 0:
                    # Try to find a rank-1 tensor matching the deficit
                    a0, b0, c0 = nz[0]
                    val0 = deficit[a0, b0, c0]
                    
                    # Check if deficit is rank-1
                    all_a = set(a for (a,b,c) in nz)
                    all_b = set(b for (a,b,c) in nz)
                    all_c = set(c for (a,b,c) in nz)
                    
                    alpha = {a0: 1}
                    beta = {b0: 1}
                    gamma = {c0: 1}
                    
                    for (a, b, c) in nz:
                        val = deficit[a, b, c]
                        if b == b0 and c == c0:
                            alpha[a] = val / val0
                        elif a == a0 and c == c0:
                            beta[b] = val / val0
                        elif a == a0 and b == b0:
                            gamma[c] = val / val0
                    
                    # Verify
                    valid = True
                    for (a, b, c) in nz:
                        expected = val0 * alpha.get(a, 0) * beta.get(b, 0) * gamma.get(c, 0)
                        if abs(deficit[a, b, c] - expected) > 1e-9:
                            valid = False
                            break
                    
                    # Also check that entries not in nz are zero
                    if valid:
                        for a in range(n*m):
                            for b in range(m*p):
                                for c in range(p*n):
                                    if (a,b,c) not in set(map(tuple, nz)):
                                        expected = val0 * alpha.get(a, 0) * beta.get(b, 0) * gamma.get(c, 0)
                                        if abs(expected) > 1e-9:
                                            valid = False
                                            break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Check half-integer
                        is_half = True
                        for d in [alpha, beta, gamma]:
                            for idx2, val in d.items():
                                if abs(val - round(val*2)/2) > 1e-9:
                                    is_half = False
                                    break
                            if not is_half:
                                break
                        
                        if is_half:
                            u_new = {a: alpha.get(a, 0) for a in all_a if alpha.get(a, 0) != 0}
                            v_new = {b: beta.get(b, 0) for b in all_b if beta.get(b, 0) != 0}
                            w_new = {c: gamma.get(c, 0) for c in all_c if gamma.get(c, 0) != 0}
                            
                            new_terms = reduced + [(u_new, v_new, w_new)]
                            U, V, W = _terms_to_matrices(new_terms, n, m, p)
                            if _verify_fast(U, V, W, Texp):
                                if len(new_terms) < best_rank:
                                    best_terms = new_terms
                                    best_rank = len(new_terms)
    
    # Phase 3: Another round of greedy merge on the improved terms
    if time.time() < deadline - 10:
        merged2 = _greedy_merge(best_terms, Texp, n, m, p, deadline - 10)
        if len(merged2) < best_rank:
            U, V, W = _terms_to_matrices(merged2, n, m, p)
            if _verify_fast(U, V, W, Texp):
                best_terms = merged2
                best_rank = len(merged2)
    
    # Convert to final format
    U, V, W = _terms_to_matrices(best_terms, n, m, p)
    
    # Final verification
    if _verify_fast(U, V, W, Texp):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return _schoolbook(n, m, p)
