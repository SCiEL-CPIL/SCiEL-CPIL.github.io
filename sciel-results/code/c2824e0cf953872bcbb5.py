    # Phase 7: Iterated Local Search
    if best_path and time.time() - start_time < time_limit - 10:
        current_best = list(best_path)
        rng_ils = random.Random(999)
        
        while time.time() - start_time < time_limit - 2:
            # Perturb current_best
            # Strategy: Reverse a random segment
            p_len = len(current_best)
            if p_len > 2:
                i = rng_ils.randint(0, p_len - 1)
                j = rng_ils.randint(0, p_len - 1)
                if i > j: i, j = j, i
                # Ensure segment length > 1
                if j - i < 1:
                    j = min(p_len - 1, i + 1)
                
                perturbed = current_best[:i] + current_best[i:j+1][::-1] + current_best[j+1:]
                
                # Check if perturbed is valid (it should be if edges exist)
                # Reversing a segment in a path is valid if the new boundary edges exist.
                # The internal edges are reversed but still exist (undirected).
                # We need to check:
                # 1. If i > 0: (current_best[i-1], current_best[j]) must be edge? 
                #    Wait, the new path is ... A, B_rev, C ...
                #    Original: ... A, B_start ... B_end, C ...
                #    New: ... A, B_end ... B_start, C ...
                #    Edges to check: (A, B_end) and (B_start, C).
                
                if i > 0 and j < p_len - 1:
                    if current_best[j] not in adj[current_best[i-1]] or current_best[i] not in adj[current_best[j+1]]:
                        # Invalid perturbation, try another or skip
                        continue
                elif i == 0 and j < p_len - 1:
                    if current_best[i] not in adj[current_best[j+1]]:
                        continue
                elif i > 0 and j == p_len - 1:
                    if current_best[j] not in adj[current_best[i-1]]:
                        continue
                # If i==0 and j==end, it's just full reversal, always valid.
                
                # Refine perturbed path
                refined = full_refine(perturbed)
                
                if len(refined) > len(current_best):
                    current_best = refined
                    best_path = refined
                # Else, we keep current_best as the base for next perturbation?
                # Or do we perturb current_best again?
                # Standard ILS: x = local_search(perturb(x))
                # If we don't improve, x becomes the worse path.
                # This allows escaping local minima.
                # But if we go too far, we lose progress.
                # Let's use a "Reheat" strategy:
                # If no improvement for 10 iterations, restart from best_path.
                # For now, let's just perturb current_best.
                current_best = refined
            else:
                break
    