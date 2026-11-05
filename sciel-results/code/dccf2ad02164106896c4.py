        if swaps_done == 0:
            no_improvement_count += 1
        else:
            # Swaps don't directly improve score, but might enable future expansion
            # We reset no_improvement_count only on expansion
            pass
