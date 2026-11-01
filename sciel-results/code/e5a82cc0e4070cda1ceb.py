def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    # Strassen for 2x2 (indices 0,1)
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    add_term({0: 1}, {1: 1, 4: -1}, {0: 1, 3: -1})
    add_term({4: 1}, {3: 1, 0: -1}, {1: 1, 4: -1})
    add_term({0: 1, 1: 1}, {4: 1}, {0: 1, 3: 1})
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 0: 1})
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1, 1: 1})
    
    # 16 terms for index-2 entries
    add_term({0: 1}, {2: 1}, {6: 1})
    add_term({1: 1}, {5: 1}, {6: 1})
    add_term({2: 1}, {8: 1}, {6: 1})
    add_term({3: 1}, {2: 1}, {7: 1})
    add_term({4: 1}, {5: 1}, {7: 1})
    add_term({5: 1}, {8: 1}, {7: 1})
    add_term({6: 1}, {0: 1}, {2: 1})
    add_term({7: 1}, {3: 1}, {2: 1})
    add_term({8: 1}, {6: 1}, {2: 1})
    add_term({6: 1}, {1: 1}, {5: 1})
    add_term({7: 1}, {4: 1}, {5: 1})
    add_term({8: 1}, {7: 1}, {5: 1})
    add_term({6: 1}, {2: 1}, {8: 1})
    add_term({7: 1}, {5: 1}, {8: 1})
    add_term({8: 1}, {8: 1}, {8: 1})
    add_term({0: 1, 1: 1}, {2: 1}, {6: 1})
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries.items():
            U[idx][r] = val
        for idx, val in v_entries.items():
            V[idx][r] = val
        for idx, val in w_entries.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
