class dsu:
    def __init__(self):
        self.point3d = {}
        self.parent = {}
        self.dsu = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.dsu[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.make_set(x)
        self.make_set(y)

        rx = self.find(x)
        ry = self.find(y)

        if rx == ry:
            return

        if self.dsu[rx] < self.dsu[ry]:
            self.parent[rx] = ry
        elif self.dsu[rx] > self.dsu[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.dsu[rx] += 1

    def groups(self):
        res = {}
        for x in self.parent:
            root = self.find(x)
            if root not in res:
                res[root] = []
            res[root].append(x)
        return list(res.values())