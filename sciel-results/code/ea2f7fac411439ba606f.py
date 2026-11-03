# 2x2x4 rank 12 decomposition
# A: 2x2, B: 2x4
# C: 2x4
# Indices for A: 00, 01, 10, 11
# Indices for B: 00, 01, 02, 03, 10, 11, 12, 13
# Indices for C: 00, 01, 02, 03, 10, 11, 12, 13

# Terms (u, v, w) where u is A index, v is B index, w is C index
# Coefficients are 1, -1, 0.5 etc.

# This is risky. I will use a fallback:
# If I cannot guarantee 24, I will keep 28.
# But the directive says "design a new program... from scratch".
# The current one is 28.
# I will try to implement the 24-rank one.
