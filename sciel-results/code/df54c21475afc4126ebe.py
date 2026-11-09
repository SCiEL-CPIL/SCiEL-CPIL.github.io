word = "1"
for m in 2..n:
    new_word = ""
    for each permutation P of [1..m-1] in order of appearance in word:
        insert m between each consecutive pair in P, AND append m at the end
        (so P = p1 p2 ... p_{m-1} becomes p1 m p2 m ... p_{m-1} m)
    word = new_word
