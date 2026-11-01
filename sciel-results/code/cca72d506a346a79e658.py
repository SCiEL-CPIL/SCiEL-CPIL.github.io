<<<<<<< SEARCH
    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
=======
    # Final Polish on the global best before returning
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1:
        final_polish_result, final_polish_size = swap_polish(
            best_in_set, max_time=remaining_time
        )
        if final_polish_size > best_size:
            best_size = final_polish_size
            best_in_set = final_polish_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
>>>>>>> REPLACE
