    def gen_all_nonbacktracking_closed_walks(length):
        """Generate coefficient vectors for all non-backtracking closed walks of given length in K_6."""
        # We will generate walks for each start vertex 0..5
        # and deduplicate the resulting (edge, sign) lists
        seen_constraints = set()
        
        for start in range(base_n):
            def walk_gen(pos, cur, prv, path):
                if pos == length:
                    # Close the walk: need cur -> start
                    # and start != prv (no backtrack at closure)
                    # and cur != start (edge exists, distinct in K6)
                    if cur != start and prv != start:
                        # Compute constraint
                        terms = []
                        # path is [v0, ..., v_{L-1}] where v0=start
                        # We need to add the last edge (cur, start)
                        full_path = path + [start]
                        for i in range(length):
                            u, v = full_path[i], full_path[i+1]
                            eu, ev = min(u, v), max(u, v)
                            eidx = edge_index[(eu, ev)]
                            sign = 1 if u < v else -1
                            terms.append((eidx, sign))
                        
                        # Normalize the constraint to deduplicate
                        # A constraint is a set of (edge, coeff) pairs.
                        # Coefficients can be -1, 0, 1, 2...
                        # Actually, in K6, an edge might appear multiple times?
                        # No, non-backtracking walks in a simple graph don't traverse the same edge twice in opposite directions immediately.
                        # But they can traverse the same edge multiple times in the same direction?
                        # No, if you traverse u->v, you can't traverse u->v again immediately because v->u is backtrack.
                        # Can you traverse u->v, then v->w, then w->...->u, then u->v?
                        # Yes. So an edge can appear multiple times.
                        # So the coefficient can be > 1.
                        
                        # To deduplicate, we can use a canonical form.
                        # Sort the terms.
                        # But (e, 1) and (e, 1) are two separate terms?
                        # No, the sum is what matters.
                        # So we should aggregate coefficients.
                        coeff_dict = {}
                        for eidx, sign in terms:
                            coeff_dict[eidx] = coeff_dict.get(eidx, 0) + sign
                        
                        # Convert to sorted tuple of (eidx, coeff)
                        canonical = tuple(sorted(coeff_dict.items()))
                        
                        if canonical not in seen_constraints:
                            seen_constraints.add(canonical)
                            yield canonical
                    return

                if pos == length - 1:
                    # Next must be start
                    if cur != start and prv != start:
                        # We can yield here or let the pos==length block handle it?
                        # The pos==length block expects path to have length L-1 elements?
                        # My walk_gen appends to path.
                        # Let's stick to the logic:
                        # If pos == length - 1, we are at the last vertex before closing.
                        # We check if we can close.
                        # If yes, we construct the full path and yield.
                        # But my previous logic yielded in pos==length block.
                        # Let's make it consistent.
                        pass 

                for nv in range(base_n):
                    if nv == cur or nv == prv:
                        continue
                    # Also, if this is the last step, nv must be start?
                    # No, we handle closure at pos == length - 1 or length.
                    # Let's say pos is the index of the current vertex in the path.
                    # path = [v0, ..., v_pos]
                    # We want to choose v_{pos+1}.
                    
                    if pos + 1 == length:
                        # This is the last step. It must go to start.
                        if nv == start:
                            path.append(nv)
                            yield from walk_gen(pos + 1, nv, cur, path)
                            path.pop()
                    else:
                        path.append(nv)
                        yield from walk_gen(pos + 1, nv, cur, path)
                        path.pop()

            # Start the walk
            # v0 = start
            # v1 can be any neighbor (all others in K6)
            for v1 in range(base_n):
                if v1 == start:
                    continue
                yield from walk_gen(1, v1, start, [start, v1])
        
        return list(seen_constraints)
