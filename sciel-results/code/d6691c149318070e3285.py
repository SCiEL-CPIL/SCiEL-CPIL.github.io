<<<<<<< SEARCH
        # Run hill climbing with time budget
        remaining = 325.0 - (time.time() - _start_time)
        if remaining > 10.0:
            _hill_climb_ordering(n, order, size)
=======
        # Run hill climbing with time budget
        remaining = 325.0 - (time.time() - _start_time)
        if remaining > 10.0:
            _hill_climb_ordering(n, order, size)
        
        # After hill climbing, do a final pass: try local swaps around each position
        # to catch any remaining improvements
        if not _is_deadline_exceeded():
            final_order = _orderings[n]['order']
            final_size = _orderings[n]['size']
            improved = True
            while improved and not _is_deadline_exceeded():
                improved = False
                for pos in range(len(final_order) - 1):
                    if _is_deadline_exceeded():
                        break
                    # Try swapping pos with pos+1
                    new_s = _eval_swap(final_order, n, pos, pos + 1)
                    if new_s > final_size:
                        final_order[pos], final_order[pos + 1] = final_order[pos + 1], final_order[pos]
                        final_size = new_s
                        _orderings[n]['size'] = final_size
                        improved = True
            _orderings[n]['order'] = final_order
            _orderings[n]['rank_map'] = {v: i for i, v in enumerate(final_order)}
>>>>>>> REPLACE
