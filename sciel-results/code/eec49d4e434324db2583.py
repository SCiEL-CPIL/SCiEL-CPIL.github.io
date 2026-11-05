        def solve_voltage(m, time_limit):
            v = np.random.randint(1, m, size=21).astype(np.int64)
            
            sums = np.mod(C @ v, m)
            cost = np.sum(sums == 0)
            
            if cost == 0:
                return v
            
            best_v = v.copy()
            best_cost = cost
            best_sums = sums.copy()
            
            T = 1.0
            cooling = 0.999
            start_t = time.time()
            
            while time.time() - start_t < time_limit:
                if remaining() < 5:
                    break
                
                e = random.randint(0, 20)
                old_val = v[e]
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue
                v[e] = new_val
                
                delta_val = new_val - old_val
                new_cost = cost
                affected_indices = []
                old_sums = []
                
                for w_idx, coeff in edge_updates[e]:
                    affected_indices.append(w_idx)
                    old_sum = sums[w_idx]
                    old_sums.append(old_sum)
                    new_sum = (old_sum + delta_val * coeff) % m
                    sums[w_idx] = new_sum
                    
                    if old_sum == 0:
                        if new_sum != 0:
                            new_cost -= 1
                    else:
                        if new_sum == 0:
                            new_cost += 1
                
                if new_cost < cost or (T > 0.01 and random.random() < np.exp(-(new_cost - cost) / T)):
                    cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        best_sums = sums.copy()
                        if best_cost == 0:
                            return best_v
                else:
                    v[e] = old_val
                    for i, w_idx in enumerate(affected_indices):
                        sums[w_idx] = old_sums[i]
                
                T *= cooling
            
            return best_v if best_cost == 0 else None
