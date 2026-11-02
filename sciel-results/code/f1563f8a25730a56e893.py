import time
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _compute_ratio_fast(S):
    """Compute |S-S|/|S+S| for a small set S using set operations."""
    sumset = set()
    diffset = set()
    for a in S:
        for b in S:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(sumset) == 0:
        return 0.0
    return len(diffset) / len(sumset)

def _generate_modular_atoms(max_size=6, max_span=40):
    """Generate atoms using modular lattice principles.
    
    For a modulus m, select residues that form a dense sublattice.
    The key is to maximize spread of residues (for diffset) while
    causing sumset to wrap around and collapse.
    """
    atoms = []
    
    # Try different moduli and densities
    for m in range(5, 30):
        # Try different subset sizes
        for k in range(2, min(max_size, m + 1)):
            # Generate combinations of residues
            # We want residues that are "spread out" in Z_m
            # but whose sums concentrate
            for comb in combinations(range(m), k):
                if 0 not in comb:
                    # Add 0 to ensure 0 is in the set
                    S = sorted(list(comb) + [0])
                else:
                    S = sorted(list(comb))
                
                span = max(S)
                if span > max_span:
                    continue
                
                r = _compute_ratio_fast(S)
                if r > 1.0:
                    log_r = np.log(r)
                    log_span = np.log(max(span, 1))
                    efficiency = log_r / log_span if log_span > 0 else 0
                    
                    atoms.append({
                        'set': S,
                        'r': r,
                        'span': span,
                        'log_r': log_r,
                        'log_span': log_span,
                        'efficiency': efficiency,
                        'modulus': m
                    })
    
    # Also generate atoms from geometric-like structures
    # These are known to have good properties
    for base in range(2, 6):
        for length in range(2, max_size + 1):
            S = [base**i for i in range(length)]
            if max(S) <= max_span:
                r = _compute_ratio_fast(S)
                if r > 1.0:
                    span = max(S)
                    log_r = np.log(r)
                    log_span = np.log(max(span, 1))
                    efficiency = log_r / log_span if log_span > 0 else 0
                    atoms.append({
                        'set': S,
                        'r': r,
                        'span': span,
                        'log_r': log_r,
                        'log_span': log_span,
                        'efficiency': efficiency,
                        'modulus': None
                    })
    
    # Generate atoms from "2^k - 1" type structures (Mersenne-like)
    for k in range(1, 8):
        S = [2**i - 1 for i in range(1, k + 1)]
        if 0 not in S:
            S.append(0)
        S = sorted(S)
        if max(S) <= max_span:
            r = _compute_ratio_fast(S)
            if r > 1.0:
                span = max(S)
                log_r = np.log(r)
                log_span = np.log(max(span, 1))
                efficiency = log_r / log_span if log_span > 0 else 0
                atoms.append({
                    'set': S,
                    'r': r,
                    'span': span,
                    'log_r': log_r,
                    'log_span': log_span,
                    'efficiency': efficiency,
                    'modulus': None
                })
    
    # Remove duplicates and sort by efficiency
    unique_atoms = []
    seen = set()
    for atom in atoms:
        key = tuple(sorted(atom['set']))
        if key not in seen:
            seen.add(key)
            unique_atoms.append(atom)
    
    unique_atoms.sort(key=lambda x: x['efficiency'], reverse=True)
    
    return unique_atoms

def _calc_analytic_score(atom_indices, atoms):
    """Calculate the analytic score for a sequence of atoms."""
    if not atom_indices:
        return 0.0
    
    log_num = 0.0
    max_u = 0
    m = 1
    
    for idx in atom_indices:
        S = atoms[idx]['set']
        log_num += atoms[idx]['log_r']
        
        span = atoms[idx]['span']
        
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        m = 2 * max_u + 1
        
        if max_u > 10**18:
            return -np.inf
            
    if max_u == 0:
        return 0.0
        
    log_denom = np.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
        
    return 1.0 + log_num / log_denom

def _refine_gaps(U, max_iters=500, deadline=None):
    """Refine the set U using gap-based local search.
    
    This is a refinement pass that can find improvements
    not captured by the mixed-radix structure.
    """
    U = sorted(list(set(U)))
    if len(U) < 2:
        return U
    
    # Convert to gaps
    gaps = [U[i+1] - U[i] for i in range(len(U) - 1)]
    
    def score_from_gaps(g):
        if not g:
            return 0.0
        U_local = [0]
        for gi in g:
            U_local.append(U_local[-1] + gi)
        if U_local[-1] == 0:
            return 0.0
        return _score_set(U_local)
    
    current_gaps = gaps[:]
    current_score = score_from_gaps(current_gaps)
    best_gaps = current_gaps[:]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 100
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        move_type = np.random.randint(0, 4)
        new_gaps = current_gaps[:]
        
        if move_type == 0:  # Modify gap
            if len(new_gaps) > 0:
                idx = np.random.randint(0, len(new_gaps))
                delta = np.random.choice([-2, -1, 1, 2, 3])
                new_val = new_gaps[idx] + delta
                if new_val >= 1:
                    new_gaps[idx] = new_val
                    
        elif move_type == 1:  # Insert gap
            if len(new_gaps) < 30:
                pos = np.random.randint(0, len(new_gaps) + 1)
                new_gaps.insert(pos, np.random.randint(1, 10))
                
        elif move_type == 2:  # Remove gap
            if len(new_gaps) > 1:
                pos = np.random.randint(0, len(new_gaps))
                new_gaps.pop(pos)
                
        elif move_type == 3:  # Swap gaps
            if len(new_gaps) > 1:
                i, j = np.random.choice(len(new_gaps), 2, replace=False)
                new_gaps[i], new_gaps[j] = new_gaps[j], new_gaps[i]
                
        if new_gaps != current_gaps:
            new_score = score_from_gaps(new_gaps)
            if new_score > current_score:
                current_gaps = new_gaps
                current_score = new_score
                improved = True
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_gaps = new_gaps[:]
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    # Reconstruct U from best gaps
    U_best = [0]
    for g in best_gaps:
        U_best.append(U_best[-1] + g)
    
    return U_best

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Step 1: Generate atoms using modular lattice principles
    atoms = _generate_modular_atoms(max_size=6, max_span=40)
    
    if not atoms:
        # Fallback: simple atoms
        atoms = [{'set': [0, 1], 'r': 1.0, 'span': 1, 'log_r': 0.0, 'log_span': 0.0, 'efficiency': 0.0, 'modulus': None}]
    
    # Limit to top atoms for search efficiency
    search_pool = atoms[:min(150, len(atoms))]
    search_pool_indices = list(range(len(search_pool)))
    
    # Step 2: Initialize with best single atom
    best_score = 0.0
    best_seq = []
    
    for idx in search_pool_indices:
        s = _calc_analytic_score([idx], search_pool)
        if s > best_score:
            best_score = s
            best_seq = [idx]
    
    # Step 3: Local search on atom sequences
    current_seq = best_seq[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 150
    step = 0
    
    while time.time() < deadline - 20.0:  # Reserve 20s for refinement
        step += 1
        improved = False
        
        # Choose move type with bias
        if len(current_seq) < 5:
            move_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.3, 0.1])
        else:
            move_type = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.4, 0.1])
        
        new_seq = current_seq[:]
        
        if move_type == 0:  # Add atom
            if len(new_seq) < 25:
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                pos = np.random.randint(0, len(new_seq) + 1)
                new_seq.insert(pos, idx)
                
        elif move_type == 1:  # Remove atom
            if len(new_seq) > 1:
                pos = np.random.randint(0, len(new_seq))
                new_seq.pop(pos)
                
        elif move_type == 2:  # Replace atom
            if len(new_seq) > 0:
                pos = np.random.randint(0, len(new_seq))
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                new_seq[pos] = idx
                
        elif move_type == 3:  # Swap
            if len(new_seq) > 1:
                i, j = np.random.choice(len(new_seq), 2, replace=False)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                
        if new_seq != current_seq:
            new_score = _calc_analytic_score(new_seq, search_pool)
            
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
                improved = True
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
            else:
                no_improve_count += 1
        
        # Random restart if stuck
        if no_improve_count > max_no_improve:
            # Greedy build from scratch
            g_seq = []
            g_score = 0
            for _ in range(15):
                best_add_idx = -1
                best_add_score = g_score
                for idx in search_pool_indices[:30]:
                    test_seq = g_seq + [idx]
                    s = _calc_analytic_score(test_seq, search_pool)
                    if s > best_add_score:
                        best_add_score = s
                        best_add_idx = idx
                if best_add_idx != -1:
                    g_seq.append(best_add_idx)
                    g_score = best_add_score
                else:
                    break
                    
            if g_score > current_score:
                current_seq = g_seq
                current_score = g_score
                if g_score > best_score:
                    best_score = g_score
                    best_seq = g_seq[:]
            else:
                # Random restart
                size = np.random.randint(1, 8)
                new_seq = []
                for _ in range(size):
                    idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                    new_seq.append(idx)
                current_seq = new_seq
                current_score = _calc_analytic_score(current_seq, search_pool)
                
            no_improve_count = 0
    
    # Step 4: Construct the final set U from best_seq
    if not best_seq:
        return [0, 1]
        
    U = set([0])
    max_u = 0
    m = 1
    
    for idx in best_seq:
        S = search_pool[idx]['set']
        span = search_pool[idx]['span']
        
        new_U = set()
        for u in U:
            for s in S:
                new_U.add(u + m * s)
        U = new_U
        
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        m = 2 * max_u + 1
        
        if max_u > 10**18:
            break
            
    U_list = sorted(list(U))
    
    # Step 5: Refinement pass using gap-based local search
    if time.time() < deadline - 5.0 and len(U_list) <= 200:
        U_list = _refine_gaps(U_list, max_iters=300, deadline=deadline - 5.0)
    
    # Ensure 0 is in the set
    if 0 not in U_list:
        U_list.insert(0, 0)
        
    return U_list

if __name__ == "__main__":
    s = construct()
    print(f"Result Size: {len(s)}")
    if len(s) < 20:
        print(f"Result: {s}")
    else:
        print(f"Result: {s[:10]}... {s[-5:]}")
    print(f"Score: {_score_set(s)}")
