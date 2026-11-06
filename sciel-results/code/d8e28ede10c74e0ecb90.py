<<<<<<< SEARCH
    # 1. All odds
    odds_mask = 0
    for i in range(1, n_max + 1, 2):
        odds_mask |= (1 << i)
    initial_solutions.append(odd_mask if 'odd_mask' in dir() else odds_mask)
=======
    # 1. All odds
    odds_mask = 0
    for i in range(1, n_max + 1, 2):
        odds_mask |= (1 << i)
    initial_solutions.append(odds_mask)
>>>>>>> REPLACE
