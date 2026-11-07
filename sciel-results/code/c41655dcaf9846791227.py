"""Combined program for spectral graph lifts with analytic eigenvalue formula and intensive local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    # Map edge indices to a, b, c variables
    a_idx = [-1] * m
    b_idx = [-1] * m
    c_idx = [-1] * m
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                a_idx[u] = idx
            elif u == (v + 1) % m:
                a_idx[v] = idx
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                b_idx[u_local] = idx
            elif u_local == (v_local + 1) % m:
                b_idx[v_local] = idx
        else:
            i = min(u, v)
            c_idx[i] = idx

    assert all(i != -1 for i in a_idx), "Missing top edge"
    assert all(i != -1 for i in b_idx), "Missing bottom edge"
    assert all(i != -1 for i in c_idx), "Missing matching edge"

    # Precompute roots of unity
    k_vals = np.arange(m)
    omega = np.exp(-2j * np.pi * k_vals / m)

    def spectral_radius_fast(a, b, c):
        """Analytic spectral radius using circulant structure. O(m) vs O(m^3)."""
        a_c = a.astype(np.complex128)
        b_c = b.astype(np.complex128)
        c_c = c.astype(np.complex128)
        
        # For each k: top_k = a * omega^k, bottom_k = b * omega^(-k)
        top = a_c * omega
        bottom = b_c * np.conj(omega)
        c_sq = 4.0 * c_c * c_c
        
        # lambda_± = (top + bottom) ± sqrt((top+bottom)^2 + 4*c^2)
        s = top + bottom
        disc = s * s + c_sq
        sqrt_disc = np.sqrt(disc)
        
        lam_p = s + sqrt_disc
        lam_m = s - sqrt_disc
        
        return np.max(np.abs(np.concatenate([lam_p, lam_m])))

    def is_connected_fast(a, b, c):
        p_outer = 1
        for x in a: p_outer *= x
        for x in b: p_outer *= x
        p_outer *= c[0] * c[m-1]
        
        if p_outer == -1:
            return True
        
        for i in range(m):
            if a[i] * c[(i+1)%m] * b[i] * c[i] == -1:
                return True
        return False

    def signs_from_abc(a, b, c):
        signs = np.zeros(num_edges, dtype=np.int8)
        for i in range(m):
            signs[a_idx[i]] = a[i]
            signs[b_idx[i]] = b[i]
            signs[c_idx[i]] = c[i]
        return signs

    best_signs = None
    best_radius = np.inf

    def update_best(a, b, c, radius):
        nonlocal best_signs, best_radius
        if radius < best_radius - 1e-12:
            best_radius = radius
            best_signs = signs_from_abc(a, b, c)

    # For m <= 5, exhaustive search guarantees global optimum
    if m <= 5:
        total_vars = 3 * m
        for combo in iproduct([-1, 1], repeat=total_vars):
            if time.time() > deadline:
                break
            a = np.array(combo[0:m], dtype=np.int8)
            b = np.array(combo[m:2*m], dtype=np.int8)
            c = np.array(combo[2*m:3*m], dtype=np.int8)
            
            if not is_connected_fast(a, b, c):
                continue
            
            radius = spectral_radius_fast(a, b, c)
            update_best(a, b, c, radius)
        
        if best_signs is not None:
            try:
                check_lift(graph, best_signs)
                return best_signs
            except:
                pass

    # For m > 5: intensive search using fast spectral radius
    rng = np.random.default_rng(3)
    
    # Phase 1: Many random restarts to find a good starting point
    n_restarts = 5000
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        a = rng.choice([-1, 1], size=m).astype(np.int8)
        b = rng.choice([-1, 1], size=m).astype(np.int8)
        c = rng.choice([-1, 1], size=m).astype(np.int8)
        
        if not is_connected_fast(a, b, c):
            continue
        
        radius = spectral_radius_fast(a, b, c)
        update_best(a, b, c, radius)
    
    # Phase 2: Local search from the best found
    if best_signs is not None:
        # Reconstruct current state from best_signs
        cur_a = np.zeros(m, dtype=np.int8)
        cur_b = np.zeros(m, dtype=np.int8)
        cur_c = np.zeros(m, dtype=np.int8)
        for i in range(m):
            cur_a[i] = best_signs[a_idx[i]]
            cur_b[i] = best_signs[b_idx[i]]
            cur_c[i] = best_signs[c_idx[i]]
        
        current_radius = best_radius
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5.0:
            improved = False
            
            # 1-flip moves
            for var_type in range(3):
                if time.time() > deadline - 5.0:
                    break
                for i in range(m):
                    if time.time() > deadline - 5.0:
                        break
                    
                    a = cur_a.copy()
                    b = cur_b.copy()
                    c = cur_c.copy()
                    
                    if var_type == 0: a[i] = -a[i]
                    elif var_type == 1: b[i] = -b[i]
                    else: c[i] = -c[i]
                    
                    if not is_connected_fast(a, b, c):
                        continue
                    
                    radius = spectral_radius_fast(a, b, c)
                    if radius < current_radius - 1e-10:
                        cur_a, cur_b, cur_c = a, b, c
                        current_radius = radius
                        update_best(a, b, c, radius)
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 2-flip moves
            two_flip_improved = False
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i, m):
                    if time.time() > deadline - 5.0:
                        break
                    for vi in range(3):
                        if time.time() > deadline - 5.0:
                            break
                        for vj in range(vi, 3):
                            if time.time() > deadline - 5.0:
                                break
                            
                            a = cur_a.copy()
                            b = cur_b.copy()
                            c = cur_c.copy()
                            
                            if vi == 0: a[i] = -a[i]
                            elif vi == 1: b[i] = -b[i]
                            else: c[i] = -c[i]
                            
                            if vj == 0: a[j] = -a[j]
                            elif vj == 1: b[j] = -b[j]
                            else: c[j] = -c[j]
                            
                            if not is_connected_fast(a, b, c):
                                continue
                            
                            radius = spectral_radius_fast(a, b, c)
                            if radius < current_radius - 1e-10:
                                cur_a, cur_b, cur_c = a, b, c
                                current_radius = radius
                                update_best(a, b, c, radius)
                                two_flip_improved = True
                                no_improve_count = 0
                                break
                        if two_flip_improved: break
                    if two_flip_improved: break
                if two_flip_improved: break
            
            if two_flip_improved:
                continue
            
            # 3-flip moves (limited)
            three_flip_improved = False
            if m <= 9:  # Only feasible for small m
                for i in range(m):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i, m):
                        if time.time() > deadline - 5.0:
                            break
                        for k in range(j, m):
                            if time.time() > deadline - 5.0:
                                break
                            for vi in range(3):
                                if time.time() > deadline - 5.0:
                                    break
                                for vj in range(vi, 3):
                                    if time.time() > deadline - 5.0:
                                        break
                                    for vk in range(vj, 3):
                                        if time.time() > deadline - 5.0:
                                            break
                                        
                                        a = cur_a.copy()
                                        b = cur_b.copy()
                                        c = cur_c.copy()
                                        
                                        if vi == 0: a[i] = -a[i]
                                        elif vi == 1: b[i] = -b[i]
                                        else: c[i] = -c[i]
                                        
                                        if vj == 0: a[j] = -a[j]
                                        elif vj == 1: b[j] = -b[j]
                                        else: c[j] = -c[j]
                                        
                                        if vk == 0: a[k] = -a[k]
                                        elif vk == 1: b[k] = -b[k]
                                        else: c[k] = -c[k]
                                        
                                        if not is_connected_fast(a, b, c):
                                            continue
                                        
                                        radius = spectral_radius_fast(a, b, c)
                                        if radius < current_radius - 1e-10:
                                            cur_a, cur_b, cur_c = a, b, c
                                            current_radius = radius
                                            update_best(a, b, c, radius)
                                            three_flip_improved = True
                                            no_improve_count = 0
                                            break
                                    if three_flip_improved: break
                                if three_flip_improved: break
                            if three_flip_improved: break
                        if three_flip_improved: break
                    if three_flip_improved: break
                if three_flip_improved:
                    continue
            
            # Random restart from current neighborhood
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Perturb current solution
                n_perturb = rng.integers(1, min(4, m))
                a = cur_a.copy()
                b = cur_b.copy()
                c = cur_c.copy()
                for _ in range(n_perturb):
                    var_type = rng.integers(3)
                    idx = rng.integers(m)
                    if var_type == 0: a[idx] = -a[idx]
                    elif var_type == 1: b[idx] = -b[idx]
                    else: c[idx] = -c[idx]
                
                if is_connected_fast(a, b, c):
                    cur_a, cur_b, cur_c = a, b, c
                    current_radius = spectral_radius_fast(a, b, c)
                    update_best(a, b, c, current_radius)
                no_improve_count = 0
            elif no_improve_count >= max_no_improve * 2:
                # Full random restart
                a = rng.choice([-1, 1], size=m).astype(np.int8)
                b = rng.choice([-1, 1], size=m).astype(np.int8)
                c = rng.choice([-1, 1], size=m).astype(np.int8)
                
                if not is_connected_fast(a, b, c):
                    continue
                
                radius = spectral_radius_fast(a, b, c)
                if radius < current_radius:
                    cur_a, cur_b, cur_c = a, b, c
                    current_radius = radius
                    update_best(a, b, c, radius)
                no_improve_count = 0

    # Fallback if no valid solution found yet
    if best_signs is None:
        cur_a = np.ones(m, dtype=np.int8)
        cur_b = np.ones(m, dtype=np.int8)
        cur_c = np.ones(m, dtype=np.int8)
        cur_c[0] = -1
        best_signs = signs_from_abc(cur_a, cur_b, cur_c)
        best_radius = spectral_radius_fast(cur_a, cur_b, cur_c)

    # Final verification
    try:
        check_lift(graph, best_signs)
    except:
        pass

    return best_signs
# EVOLVE-BLOCK-END
