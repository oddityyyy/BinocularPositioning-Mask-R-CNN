def lp4pos(self):
    lp = []  # 一张图片的定位精度
    l = len(self.uD)
    lpe = 0
    for i in range(l):
        v = [self.uD[i], self.dD[i], self.lD[i], self.rD[i]]
        avg = list_average(v)
        max = list_max(v)
        lpe = 1-(max-avg)/avg  # lpe每个苹果的定位精度
        print(v)
        lp.append(lpe)
    return list_average(lp)