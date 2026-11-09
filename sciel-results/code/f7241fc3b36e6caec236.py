def _search_decomposition(n, m, p, target_rank, deadline):
    import random
    import time
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute index mappings for (i, j, k) -> (u_idx, v_idx, w_idx)
    indices = []
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
                
    # Map u_idx to list of (v_idx, w_idx)
    u_to_vw = {}
    # Map v_idx to list of (u_idx, w_idx)
    v_to_uw = {}
    # Map w_idx to list of (u_idx, v_idx)
    w_to_uv = {}
    
    for u_idx, v_idx, w_idx in indices:
        if u_idx not in u_to_vw:
            u_to_vw[u_idx] = []
        u_to_vw[u_idx].append((v_idx, w_idx))
        
        if v_idx not in v_to_uw:
            v_to_uw[v_idx] = []
        v_to_uw[v_idx].append((u_idx, w_idx))
        
        if w_idx not in w_to_uv:
            w_to_uv[w_idx] = []
        w_to_uv[w_idx].append((u_idx, v_idx))

    def _init_sums(U, V, W):
        sums = {}
        for u_idx, v_idx, w_idx in indices:
            total = 0
            for r in range(target_rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            sums[(u_idx, v_idx, w_idx)] = total
        return sums

    def _calc_error(sums):
        errors = 0
        for s in sums.values():
            if s % 3 != 1:
                errors += 1
        return errors

    def _check_decomposition(sums):
        for s in sums.values():
            if s != 1:
                return False
        return True

    random.seed(42 + target_rank)
    
    num_restarts = 100
    steps_per_restart = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        def _rand_val():
            r = random.random()
            if r < 0.5:
                return 0
            elif r < 0.75:
                return random.choice([-1, 1])
            else:
                return random.choice([-2, 2])

        U = [[_rand_val() for _ in range(target_rank)] for _ in range(dim_u)]
        V = [[_rand_val() for _ in range(target_rank)] for _ in range(dim_v)]
        W = [[_rand_val() for _ in range(target_rank)] for _ in range(dim_w)]
        
        sums = _init_sums(U, V, W)
        best_err = _calc_error(sums)
        current_err = best_err
        temperature = 1.0
        cooling = 0.995
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
            if best_err == 0:
                break
                
            mat_choice = random.choice([0, 1, 2])
            if mat_choice == 0:
                mat = U
                dim = dim_u
                # affected entries are in u_to_vw
            elif mat_choice == 1:
                mat = V
                dim = dim_v
            else:
                mat = W
                dim = dim_w
                
            r = random.randint(0, dim - 1)
            c = random.randint(0, target_rank - 1)
            
            old_val = mat[r][c]
            candidates = [v for v in [-2, -1, 0, 1, 2] if v != old_val]
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                # Calculate delta error
                # Change mat[r][c] from old_val to new_val
                # delta = (new_val - old_val) * other1 * other2
                delta_err = 0
                # We need to update sums and check error change
                # Instead of updating sums immediately, let's compute the change in error count
                # This is tricky because error is count of (sum % 3 != 1).
                # Let's just update sums for the affected entries and recompute error for those entries.
                
                # Identify affected entries
                if mat_choice == 0:
                    affected = u_to_vw[r]
                    # (v_idx, w_idx)
                elif mat_choice == 1:
                    affected = v_to_uw[r]
                else:
                    affected = w_to_uv[r]
                
                # For each affected entry, compute old sum mod 3 and new sum mod 3
                # old_sum = sums[key]
                # new_sum = old_sum + delta
                # delta = (new_val - old_val) * V[v][c] * W[w][c] (if U changed)
                
                old_err_count = 0
                new_err_count = 0
                delta_val = new_val - old_val
                
                for v_idx, w_idx in affected:
                    key = (r, v_idx, w_idx) if mat_choice == 0 else (u_idx, r, w_idx) if mat_choice == 1 else (u_idx, v_idx, r)
                    # Wait, the key structure needs to be consistent.
                    # Let's store sums in a dict with key (u, v, w)
                    
                    # We need the other two values
                    if mat_choice == 0:
                        # U[r][c] changes. r is u_idx.
                        # v_idx, w_idx from affected
                        other1 = V[v_idx][c]
                        other2 = W[w_idx][c]
                        key = (r, v_idx, w_idx)
                    elif mat_choice == 1:
                        # V[r][c] changes. r is v_idx.
                        # u_idx, w_idx from affected
                        other1 = U[u_idx][c]
                        other2 = W[w_idx][c]
                        key = (u_idx, r, w_idx)
                    else:
                        # W[r][c] changes. r is w_idx.
                        # u_idx, v_idx from affected
                        other1 = U[u_idx][c]
                        other2 = V[v_idx][c]
                        key = (u_idx, v_idx, r)
                    
                    old_sum = sums[key]
                    new_sum = old_sum + delta_val * other1 * other2
                    
                    if old_sum % 3 != 1:
                        old_err_count += 1
                    if new_sum % 3 != 1:
                        new_err_count += 1
                
                delta_err = new_err_count - old_err_count
                
                if delta_err < 0 or random.random() < pow(2.718, -delta_err / temperature):
                    # Accept
                    # Update sums
                    for v_idx, w_idx in affected:
                        if mat_choice == 0:
                            key = (r, v_idx, w_idx)
                            other1 = V[v_idx][c]
                            other2 = W[w_idx][c]
                        elif mat_choice == 1:
                            # Need u_idx, w_idx
                            # affected is list of (u_idx, w_idx)
                            u_idx, w_idx = v_idx, w_idx # Variable names in loop are confusing
                            # Let's rename loop vars
                            pass 
                        # This logic is getting messy. Let's simplify by just updating sums and recalculating error for affected keys.
                        pass
                    
                    # Let's restart the loop logic for acceptance cleanly
                    pass

                # This incremental approach is complex to code correctly in a short time.
                # Let's stick to the full recalculation but with the speedup from sparse init and more steps?
                # No, the full recalculation is O(N*R) = 720 ops.
                # 2000 steps * 720 ops = 1.4M ops.
                # 100 restarts * 1.4M = 140M ops.
                # Python can do ~10-50M simple ops/sec.
                # So 140M ops might take 3-15 seconds.
                # This is feasible!
                
                # So the current simple approach might be fast enough if we just run it.
                # The previous code had 20 restarts * 200 steps = 4000 steps total.
                # 4000 * 720 = 2.8M ops. Very fast.
                # So why did it fail? Probably bad luck or local minima.
                # Increasing restarts to 100 and steps to 2000 gives 200,000 steps.
                # 200,000 * 720 = 144M ops.
                # This should take ~5-10 seconds.
                # So we have time to do this.
                
                break # Stop this draft, implement the simple version with more steps.

            temperature *= cooling

        if best_err == 0:
            if _check_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
                
    return None
