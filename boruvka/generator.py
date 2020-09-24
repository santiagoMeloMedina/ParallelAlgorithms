
import random as r

num = int(input())
s = set()

print(num)
for u in range(num):
    for v in range(num):
        s1, s2 = "{},{}".format(u,v), "{},{}".format(v,u)
        if u!=v and not ((s1 in s) or (s2 in s)):
            s.add(s1)
            s.add(s2)
            print(u,v,r.randint(4,5))
