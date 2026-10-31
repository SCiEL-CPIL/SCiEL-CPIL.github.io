<<<<<<< SEARCH
    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded
            
            # Second expansion pass
            if best_result is not None and time.time() < deadline - 10:
                expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
                if expanded2 is not None:
                    valid, val = check_validity(expanded2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded2

        # 3. NEW: Contact Graph Optimization (Solving for exact tangencies)
        if best_result is not None and time.time() < deadline - 12:
            contact_res = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 6)
            if contact_res is not None:
                valid, val = check_validity(contact_res)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_res

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final quick polish
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2
=======
    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Contact Graph Optimization (closing gaps between near-tangent circles)
        if best_result is not None and time.time() < deadline - 15:
            contact_res = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 8)
            if contact_res is not None:
                valid, val = check_validity(contact_res)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_res

        # 4. High-precision SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final expansion polish
        if best_result is not None and time.time() < deadline - 6:
            expanded_final = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 2)
            if expanded_final is not None:
                valid, val = check_validity(expanded_final)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded_final

        # 6. Final quick polish
        if best_result is not None and time.time() < deadline - 3:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2
>>>>>>> REPLACE
