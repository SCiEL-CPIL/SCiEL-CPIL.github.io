    while time.time() < deadline:
        added = False
        # Try to Add
        for _ in range(20):
            if time.time() > deadline:
                break
            v = random.randint(1, N)
            if is_valid_add(v):
                try_add(v)
                added = True
                break # If we added, re-evaluate state? Or continue?
                # Breaking allows us to try more adds in the next outer loop iteration
                # or just continue the inner loop.
                # Actually, if we add, the state changes.
                # We can just break the inner add loop and go to swaps or next outer iteration.
        
        if not added:
            # Try Swaps
            if len(chosen) > 0:
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    u = random.choice(chosen)
                    v = random.randint(1, N)
                    if v in chosen_set:
                        continue
                    
                    # Remove u
                    sum_counts[2 * u] -= 1
                    for y in chosen:
                        if y != u:
                            sum_counts[u + y] -= 1
                    chosen.remove(u)
                    chosen_set.remove(u)
                    
                    # Check v
                    ok = True
                    if sum_counts[2 * v] > 1:
                        ok = False
                    else:
                        for y in chosen:
                            if sum_counts[v + y] > 1:
                                ok = False
                                break
                    
                    if ok:
                        # Add v
                        if sum_counts[2 * v] == 1:
                            sum_counts[2 * v] += 1
                        else:
                            sum_counts[2 * v] = 1
                        for y in chosen:
                            if sum_counts[v + y] == 1:
                                sum_counts[v + y] += 1
                            else:
                                sum_counts[v + y] = 1
                        chosen.append(v)
                        chosen_set.add(v)
                    else:
                        # Revert u
                        sum_counts[2 * u] += 1
                        for y in chosen:
                            if y != u:
                                sum_counts[u + y] += 1
                        chosen.append(u)
                        chosen_set.add(u)
