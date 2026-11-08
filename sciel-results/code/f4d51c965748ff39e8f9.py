# A_32 B_23:
strassen_2x2(2, 3, 0, 1, 2, 3)  # 7
for i in [2, 3]:
    for j in [0, 1]:
        add_term(...)  # 4
for j in [0, 1]:
    for k in [2, 3]:
        add_term(...)  # 4
for j in [0, 1]:
    add_term(...)  # 2

# A_33 B_33:
strassen_2x2(2, 3, 2, 3, 2, 3)  # 7
for i in [2, 3]:
    for k in [2, 3]:
        add_term(...)  # 4
for i in [2, 3]:
    for j in [2, 3]:
        add_term(...)  # 4
for i in [2, 3]:
    add_term(...)  # 2
for j in [2, 3]:
    for k in [2, 3]:
        add_term(...)  # 4
for k in [2, 3]:
    add_term(...)  # 2
for j in [2, 3]:
    add_term(...)  # 2
add_term(...)  # 1
