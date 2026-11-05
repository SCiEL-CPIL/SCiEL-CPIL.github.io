    # Final polish
    if best_triples:
        used = get_used_pairs(best_triples)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            valid_adds = find_valid_adds(used)
            if valid_adds:
                # Try all valid adds? Or just one?
                # Try all to find the one that minimizes alpha
                best_add_idx = -1
                best_add_alpha = _calc_alpha_fast(best_triples, n)
                for idx in valid_adds:
                    t_add = all_triples[idx]
                    new_triples = best_triples + [t_add]
                    new_alpha = _calc_alpha_fast(new_triples, n)
                    if new_alpha < best_add_alpha:
                        best_add_alpha = new_alpha
                        best_add_idx = idx
                if best_add_idx != -1:
                    t_add = all_triples[best_add_idx]
                    best_triples.append(t_add)
                    for p in triple_pairs[best_add_idx]:
                        used.add(p)
                    best_alpha = best_add_alpha
                    improved = True
