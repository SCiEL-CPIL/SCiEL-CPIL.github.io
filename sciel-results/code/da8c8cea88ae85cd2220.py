<<<<<<< SEARCH
    # Final check: ensure we return a valid list of ints
    return [int(v) for v in best_chosen]
=======
    # Final verification and cleanup
    # Verify the solution is valid
    final_set = set(int(v) for v in best_chosen)
    assert len(final_set) == len(best_chosen), "Duplicate elements found"
    assert all(1 <= v <= N for v in final_set), "Out of range elements found"
    
    # Verify B2[2] constraint
    count_arr = np.zeros(MAX_SUM + 1, dtype=np.int64)
    sorted_chosen = sorted(final_set)
    for i, a in enumerate(sorted_chosen):
        for j in range(i, len(sorted_chosen)):
            b = sorted_chosen[j]
            s = a + b
            count_arr[s] += 1
    assert np.all(count_arr <= K), "B2[2] constraint violated"
    
    return [int(v) for v in best_chosen]
>>>>>>> REPLACE
