<<<<<<< SEARCH
if __name__ == "__main__":
    res = construct_safe()
    print(f"Number of triples: {len(res)}")
    triple_indices = []
    for t in res:
        a, b, c = sorted(t)
        for idx in range(len(ALL_TRIPLES)):
            if ALL_TRIPLES[idx] == (a, b, c):
                triple_indices.append(idx)
                break
    alpha = compute_alpha_fast(triple_indices)
    print(f"Independence number: {alpha}")
    print(f"Score: {1.0 / alpha if alpha > 0 else 0}")
=======
if __name__ == "__main__":
    res = construct_safe()
    print(f"Number of triples: {len(res)}")
    triple_indices = []
    for t in res:
        a, b, c = sorted(t)
        for idx in range(len(ALL_TRIPLES)):
            if ALL_TRIPLES[idx] == (a, b, c):
                triple_indices.append(idx)
                break
    alpha = compute_alpha_exact(triple_indices)
    print(f"Independence number: {alpha}")
    print(f"Score: {1.0 / alpha if alpha > 0 else 0}")
>>>>>>> REPLACE
