"""Memetic GA for QAP: Population-based search with vectorized 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _solve_tai256c(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute constants for potential calculation
    # Potential for a facility f at location l given current permutation p:
    # V_f(l) = sum_{k != f} a[f, k] * b[l, p[k]] + sum_{k != f} a[k, f] * b[p[k], l]
    # We can compute this for all f and l in O(n^3) per generation if done naively,
    # but we can update incrementally or compute full matrix if n is small enough.
    # For n=256, n^3 = 16M ops. With numpy, this is very fast (<0.1s).
    
    def compute_potentials(p):
        """
        Compute V[f, l] for all f, l.
        V[f, l] = sum_{k} a[f, k] * b[l, p[k]] + sum_{k} a[k, f] * b[p[k], l]
        Note: This includes k=f terms which are a[f,f]*b[l,p[f]] and a[f,f]*b[p[f],l].
        Since we swap, the self-terms are handled separately or cancel out in delta?
        Actually, the standard potential definition for QAP swap delta is:
        Delta(f, l, f', l') = V_f(l) + V_f'(l') - V_f(l') - V_f'(l) - 2 * (a[f,f']*b[l,l'] - a[f,f']*b[l',l] ... wait)
        
        Let's use the exact formula derived from:
        Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        
        If we swap p[i]=l_i and p[j]=l_j:
        New Cost = Old Cost 
                  + a[i,j](b[l_j, l_i] - b[l_i, l_j]) 
                  + a[j,i](b[l_i, l_j] - b[l_j, l_i])
                  + sum_{k!=i,j} [ a[i,k](b[l_j, p[k]] - b[l_i, p[k]]) 
                                  + a[k,i](b[p[k], l_j] - b[p[k], l_i])
                                  + a[j,k](b[l_i, p[k]] - b[l_j, p[k]])
                                  + a[k,j](b[p[k], l_i] - b[p[k], l_j]) ]
        
        Let's define potential Phi[f, l] = sum_{k} a[f, k] b[l, p[k]] + sum_{k} a[k, f] b[p[k], l]
        
        Then sum_{k!=i,j} a[i,k](b[l_j, p[k]] - b[l_i, p[k]]) = (Phi[i, l_j] - a[i,i]b[l_j,p[i]] - a[i,j]b[l_j,p[j]]) - (Phi[i, l_i] - a[i,i]b[l_i,p[i]] - a[i,j]b[l_i,p[j]])
        This is getting messy with the exclusions.
        
        Alternative: Just compute the full matrix of potentials M where M[f, l] = sum_k a[f,k]b[l,p[k]] + sum_k a[k,f]b[p[k],l].
        Then the change in cost for swapping i and j (with locations l_i=p[i], l_j=p[j]) is:
        Delta = M[i, l_j] + M[j, l_i] - M[i, l_i] - M[j, l_j] 
                - 2 * (a[i,j] * b[l_i, l_j] + a[j,i] * b[l_j, l_i])  # Wait, the M terms include the i-j interaction.
        
        Let's verify.
        M[i, l] = sum_k a[i,k] b[l, p[k]] + sum_k a[k,i] b[p[k], l]
        M[i, l_j] includes k=j term: a[i,j] b[l_j, p[j]] = a[i,j] b[l_j, l_j] (since p[j]=l_j).
        M[i, l_i] includes k=j term: a[i,j] b[l_i, l_j].
        
        The term in Delta for i interacting with j is:
        a[i,j](b[l_j, l_i] - b[l_i, l_j])  (from the i,j pair swap directly)
        Plus the k=j part of the sum over k!=i,j? No, k!=i,j excludes j.
        
        So if we use M which includes k=j, we have to subtract the j-related parts from M and add the correct direct interaction.
        
        Direct interaction change:
        Old: a[i,j]b[l_i, l_j] + a[j,i]b[l_j, l_i]
        New: a[i,j]b[l_j, l_i] + a[j,i]b[l_i, l_j]
        Change_direct = a[i,j](b[l_j, l_i] - b[l_i, l_j]) + a[j,i](b[l_i, l_j] - b[l_j, l_i])
        
        Interaction with k != i,j:
        Old: a[i,k]b[l_i, p[k]] + a[k,i]b[p[k], l_i] + a[j,k]b[l_j, p[k]] + a[k,j]b[p[k], l_j]
        New: a[i,k]b[l_j, p[k]] + a[k,i]b[p[k], l_j] + a[j,k]b[l_i, p[k]] + a[k,j]b[p[k], l_i]
        
        Sum_{k!=i,j} [ a[i,k](b[l_j, p[k]] - b[l_i, p[k]]) + a[k,i](b[p[k], l_j] - b[p[k], l_i]) + ... ]
        
        Let's look at M[i, l_j] - M[i, l_i]:
        M[i, l_j] = sum_k a[i,k]b[l_j, p[k]] + sum_k a[k,i]b[p[k], l_j]
        M[i, l_i] = sum_k a[i,k]b[l_i, p[k]] + sum_k a[k,i]b[p[k], l_i]
        
        Difference = sum_k a[i,k](b[l_j, p[k]] - b[l_i, p[k]]) + sum_k a[k,i](b[p[k], l_j] - b[p[k], l_i])
        
        This sum is over ALL k.
        The term we want is sum over k != i, j.
        So we subtract the terms for k=i and k=j.
        
        For k=i:
        a[i,i](b[l_j, p[i]] - b[l_i, p[i]]) + a[i,i](b[p[i], l_j] - b[p[i], l_i])
        Since p[i]=l_i:
        = a[i,i](b[l_j, l_i] - b[l_i, l_i]) + a[i,i](b[l_i, l_j] - b[l_i, l_i])
        = a[i,i](b[l_j, l_i] + b[l_i, l_j] - 2*b[l_i, l_i])
        
        For k=j:
        a[i,j](b[l_j, p[j]] - b[l_i, p[j]]) + a[j,i](b[p[j], l_j] - b[p[j], l_i])
        Since p[j]=l_j:
        = a[i,j](b[l_j, l_j] - b[l_i, l_j]) + a[j,i](b[l_j, l_j] - b[l_i, l_j])  # Wait b[p[j], l_j] = b[l_j, l_j]
        Wait, second part: a[k,i] with k=j is a[j,i]. b[p[j], l_j] is b[l_j, l_j]. b[p[j], l_i] is b[l_j, l_i].
        So: a[j,i](b[l_j, l_j] - b[l_j, l_i])
        
        Total subtraction for M[i, l_j] - M[i, l_i]:
        Sub_i = a[i,i](b[l_j, l_i] + b[l_i, l_j] - 2*b[l_i, l_i])
        Sub_j = a[i,j](b[l_j, l_j] - b[l_i, l_j]) + a[j,i](b[l_j, l_j] - b[l_j, l_i])
        
        Similarly for M[j, l_i] - M[j, l_j]:
        Sub'_i = a[j,j](b[l_i, l_j] + b[l_j, l_i] - 2*b[l_j, l_j])
        Sub'_j = a[j,i](b[l_i, l_i] - b[l_j, l_i]) + a[i,j](b[l_i, l_i] - b[l_i, l_j])
        
        So:
        Delta = (M[i, l_j] - M[i, l_i] - Sub_i - Sub_j) + (M[j, l_i] - M[j, l_j] - Sub'_i - Sub'_j) + Change_direct
        
        This allows O(1) delta calculation per pair if M is precomputed.
        """
        # M[f, l] = sum_k a[f,k] b[l, p[k]] + sum_k a[k,f] b[p[k], l]
        # M = b @ a @ diag? No.
        # Term 1: sum_k a[f,k] b[l, p[k]] = (b[:, p] @ a)[f, l]? 
        # b[:, p] is matrix B' where B'[l, k] = b[l, p[k]].
        # (B' @ a)[f, l] = sum_k B'[l, k] a[k, f] ? No, matrix mult is sum_k M1[f,k]M2[k,l].
        # We want sum_k a[f,k] b[l, p[k]].
        # Let A = a. Let B_p = b[:, p].
        # sum_k A[f,k] B_p[l,k] = (A @ B_p.T)[f, l].
        
        # Term 2: sum_k a[k,f] b[p[k], l]
        # sum_k a[k,f] b[p[k], l] = (a.T @ b[p, :])[f, l]?
        # (a.T @ b[p,:])[f, l] = sum_k a.T[f,k] b[p,:][k,l] = sum_k a[k,f] b[p[k], l]. Yes.
        
        b_p = b[:, p]
        b_p_T = b_p.T
        M1 = a @ b_p_T
        M2 = a.T @ b[p, :]
        return M1 + M2

    def local_search(p, budget=2.0):
        """
        2-opt local search using potentials.
        """
        current_p = p.copy()
        current_cost = _qac_cost(a, b, current_p) # Placeholder, use _qap_cost
        current_cost = _qap_cost(a, b, current_p)
        
        # Precompute potentials
        M = compute_potentials(current_p)
        
        best_cost = current_cost
        best_p = current_p.copy()
        
        start_ls = time.time()
        no_improve_count = 0
        max_iters = 5000 # Cap iterations to prevent infinite loops in weird cases, though cost is bounded
        
        while time.time() - start_ls < budget and no_improve_count < 50:
            # Find best swap
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Sample swaps to save time? For n=256, n^2/2 = 32k swaps.
            # O(1) delta calc per swap is very fast. 32k swaps * 256 gens = 8M ops. Feasible.
            # Let's try all swaps.
            
            # We need to calculate Delta for all i < j.
            # Delta(i, j) = M[i, p[j]] + M[j, p[i]] - M[i, p[i]] - M[j, p[j]] 
            #               - Sub_i - Sub_j - Sub'_i - Sub'_j + Change_direct
            
            # This is hard to vectorize fully because of the Sub terms which depend on i,j specific b values.
            # However, we can vectorize the M part.
            # M_part[i, j] = M[i, p[j]] + M[j, p[i]] - M[i, p[i]] - M[j, p[j]]
            
            # Let's compute M_part matrix
            # M[i, p[j]]: For each i, we want M[i, p[j]] for all j.
            # This is M[:, p] ? No. M is (n, n). M[i, l] is value.
            # We want a matrix D where D[i, j] = M[i, p[j]].
            # D = M[:, p] ? No. M[:, p] has shape (n, n). (M[:, p])[i, j] = M[i, p[j]]. Yes.
            # Similarly, M[j, p[i]]: (M[:, p].T)[i, j] = M[p[i], j]? No.
            # We want M[j, p[i]]. Let K = M[:, p].T. K[i, j] = M[p[i], j]. No.
            # We want M[j, p[i]].
            # Let's construct vector v_i = p[i].
            # M[j, p[i]] for all j is M[:, p[i]].
            # So we need to gather columns of M.
            
            # Efficient vectorization:
            # M1 = M[:, p]  # M1[i, j] = M[i, p[j]]
            # M2 = M[p, :]  # M2[i, j] = M[p[i], j] -> We want M[j, p[i]], which is M2.T[i, j] = M2[j, i] = M[p[j], i]. No.
            # We want M[j, p[i]].
            # Let's just use indexing.
            
            # Actually, for n=256, a loop over i (256) and vectorized j (256) is fine.
            
            found_improvement = False
            for i in range(n):
                pi = current_p[i]
                # Vectorize over j > i
                # We need M[i, p[j]] for all j.
                # M[i, p] is a vector of M[i, p[j]] for all j.
                Mi_p = M[i, current_p]
                
                # We need M[j, pi] for all j.
                # This is column pi of M? No, M[j, pi] is element (j, pi).
                # So M[:, pi] is a vector where index j is M[j, pi].
                M_col_pi = M[:, pi]
                
                # We need M[i, pi] (scalar)
                Mi_pi = M[i, pi]
                
                # We need M[j, p[j]] for all j.
                # This is the diagonal of M[:, p]? No.
                # M[j, p[j]] is element (j, p[j]).
                # We can precompute M_diag_swap[j] = M[j, p[j]].
                # Let's precompute this outside the i loop?
                # M_diag_swap = M[np.arange(n), current_p]
                
                # Let's precompute M_diag_swap
                if i == 0:
                    M_diag_swap = M[np.arange(n), current_p]
                
                # Delta_M[i, j] = Mi_p[j] + M_col_pi[j] - Mi_pi - M_diag_swap[j]
                # We only care about j > i.
                
                # Now the Sub terms and Direct term.
                # Sub_i depends on i and j (via l_j=p[j]).
                # Sub_i = a[i,i](b[l_j, l_i] + b[l_i, l_j] - 2*b[l_i, l_i])
                # l_i = pi. l_j = p[j].
                # b[l_j, l_i] = b[current_p, pi] -> vector
                # b[l_i, l_j] = b[pi, current_p] -> vector
                # b[l_i, l_i] = b[pi, pi] -> scalar
                
                b_pj_pi = b[current_p, pi]
                b_pi_pj = b[pi, current_p]
                b_pi_pi = b[pi, pi]
                
                sub_i_vec = a[i, i] * (b_pj_pi + b_pi_pj - 2 * b_pi_pi)
                
                # Sub_j = a[i,j](b[l_j, l_j] - b[l_i, l_j]) + a[j,i](b[l_j, l_j] - b[l_j, l_i])
                # a[i,j] is vector a[i, :]
                # b[l_j, l_j] is b[current_p, current_p] -> vector
                # b[l_i, l_j] is b_pi_pj
                # b[l_j, l_i] is b_pj_pi
                
                b_pj_pj = b[current_p, current_p]
                
                sub_j_vec = a[i, :] * (b_pj_pj - b_pi_pj) + a[:, i] * (b_pj_pj - b_pj_pi)
                # Wait, a[:, i] is a[j, i] for all j. Correct.
                
                # Sub'_i = a[j,j](b[l_i, l_j] + b[l_j, l_i] - 2*b[l_j, l_j])
                # a[j,j] is b_pj_pj? No, a[j,j] is diagonal of a.
                a_diag = np.diag(a)
                # b[l_i, l_j] = b_pi_pj
                # b[l_j, l_i] = b_pj_pi
                # b[l_j, l_j] = b_pj_pj
                
                sub_pi_vec = a_diag * (b_pi_pj + b_pj_pi - 2 * b_pj_pj)
                
                # Sub'_j = a[j,i](b[l_i, l_i] - b[l_j, l_i]) + a[i,j](b[l_i, l_i] - b[l_i, l_j])
                # a[j,i] is a[:, i]
                # a[i,j] is a[i, :]
                # b[l_i, l_i] is b_pi_pi
                # b[l_j, l_i] is b_pj_pi
                # b[l_i, l_j] is b_pi_pj
                
                sub_pj_vec = a[:, i] * (b_pi_pi - b_pj_pi) + a[i, :] * (b_pi_pi - b_pi_pj)
                
                # Change_direct = a[i,j](b[l_j, l_i] - b[l_i, l_j]) + a[j,i](b[l_i, l_j] - b[l_j, l_i])
                # = (a[i,j] - a[j,i]) * (b_pj_pi - b_pi_pj)
                
                direct_vec = (a[i, :] - a[:, i]) * (b_pj_pi - b_pi_pj)
                
                # Total Delta vector for this i
                delta_vec = (Mi_p + M_col_pi - Mi_pi - M_diag_swap) - sub_i_vec - sub_j_vec - sub_pi_vec - sub_pj_vec + direct_vec
                
                # Mask out j <= i and j == i
                # We want j > i
                mask = np.zeros(n, dtype=bool)
                mask[i+1:] = True
                # Also exclude i itself (already excluded by mask)
                
                # Find best j
                if np.any(mask):
                    masked_deltas = delta_vec[mask]
                    min_idx = np.argmin(masked_deltas)
                    # Map back to original index
                    # indices = np.where(mask)[0]
                    # best_j = indices[min_idx]
                    # Since mask is 0..i-1 False, i+1..n-1 True.
                    # The min_idx is relative to the masked array.
                    # We need the actual j.
                    # Easier: set non-masked to inf
                    delta_vec_copy = delta_vec.copy()
                    delta_vec_copy[~mask] = np.inf
                    best_j_candidate = np.argmin(delta_vec_copy)
                    best_delta_candidate = delta_vec_copy[best_j_candidate]
                    
                    if best_delta_candidate < best_delta:
                        best_delta = best_delta_candidate
                        best_i = i
                        best_j = best_j_candidate
            
            if best_i == -1:
                break
            
            # Perform swap
            current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
            current_cost += int(best_delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Update M?
            # M depends on p. Swapping two elements changes M for all f and l?
            # M[f, l] = sum_k a[f,k] b[l, p[k]] + ...
            # If we swap p[i] and p[j], then for any f, l:
            # The term for k=i changes from a[f,i]b[l, p[i]] to a[f,i]b[l, p[j]]
            # The term for k=j changes from a[f,j]b[l, p[j]] to a[f,j]b[l, p[i]]
            # So M[f, l] changes by:
            # a[f,i](b[l, p[j]] - b[l, p[i]]) + a[f,j](b[l, p[i]] - b[l, p[j]])
            # + a[i,f](b[p[j], l] - b[p[i], l]) + a[j,f](b[p[i], l] - b[p[j], l])
            
            # This is an O(n^2) update for M.
            # n=256, n^2=65k. Very fast.
            
            i, j = best_i, best_j
            pi, pj = current_p[j], current_p[i] # Swapped values
            
            # Update M
            # M[f, l] += a[f,i](b[l, pj] - b[l, pi]) + a[f,j](b[l, pi] - b[l, pj]) 
            #           + a[i,f](b[pj, l] - b[pi, l]) + a[j,f](b[pi, l] - b[pj, l])
            
            # Vectorized update:
            # Term 1: a[:, i] * (b[:, pj] - b[:, pi]).T ?
            # a[:, i] is vector (n,). b[:, pj] - b[:, pi] is matrix (n, n) if we want M[f,l] update?
            # M is (n, n). M[f, l].
            # a[f, i] is row f of a[:, i].
            # b[l, pj] is row l of b[:, pj].
            # So a[f,i] * b[l,pj] is outer product of a[:, i] and b[:, pj].
            
            M += np.outer(a[:, i], b[:, pj]) - np.outer(a[:, i], b[:, pi])
            M += np.outer(a[:, j], b[:, pi]) - np.outer(a[:, j], b[:, pj])
            M += np.outer(a[i, :], b[pj, :]) - np.outer(a[i, :], b[pi, :])
            M += np.outer(a[j, :], b[pi, :]) - np.outer(a[j, :], b[pj, :])
            
        return best_p, best_cost

    # Initialization
    # 1. Rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_init = p_init.copy()
    
    # 2. Random
    p_random = np.random.permutation(n)
    
    # 3. Random
    p_random2 = np.random.permutation(n)
    
    population = [p_init, p_random, p_random2]
    # Add more randoms
    for _ in range(7):
        population.append(np.random.permutation(n))
        
    # Evaluate and refine all
    refined_pop = []
    for p in population:
        # Refine each with a small budget
        p_ref, c_ref = local_search(p, budget=1.0)
        refined_pop.append((p_ref, c_ref))
        
    # Sort by cost
    refined_pop.sort(key=lambda x: x[1])
    
    best_p, best_cost = refined_pop[0]
    
    # GA Loop
    pop_size = 10
    # Keep top 5, generate 5 new
    while time.time() < deadline:
        # Crossover
        p1, _ = refined_pop[0]
        p2, _ = refined_pop[1]
        
        # Order Crossover
        c = np.empty(n, dtype=np.int64)
        i, j = np.random.choice(n, 2, replace=False)
        if i > j: i, j = j, i
        
        c[i:j+1] = p1[i:j+1]
        remaining = [x for x in p2 if x not in c[i:j+1]]
        # Fill in
        idx = 0
        for k in range(n):
            if k < i or k > j:
                c[k] = remaining[idx]
                idx += 1
                
        child = c
        # Mutate: swap 2-5 random pairs
        for _ in range(np.random.randint(2, 6)):
            i, j = np.random.choice(n, 2, replace=False)
            child[i], child[j] = child[j], child[i]
            
        # Refine child
        p_ref, c_ref = local_search(child, budget=2.0)
        
        # Replace worst
        worst_idx = len(refined_pop) - 1
        if c_ref < refined_pop[worst_idx][1]:
            refined_pop[worst_idx] = (p_ref, c_ref)
            refined_pop.sort(key=lambda x: x[1])
            
            if c_ref < best_cost:
                best_cost = c_ref
                best_p = p_ref.copy()
                
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
