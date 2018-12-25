from itertools import combinations


# 提取全部项集
def comb(lst):
    ret = []
    for i in range(1, len(lst) + 1):
        # combinations排列，C31+C32+C33 = 3+3+1 = 7
        ret += list(combinations(lst, i))
    return ret


# 是几项集
class AprLayer(object):
    d = dict()

    def __init__(self):
        self.d = dict()


# 项集
class AprNode(object):
    def __init__(self, node):
        self.s = set(node)
        self.size = len(self.s)
        self.lnk_nodes = dict()
        self.num = 0

    # 重写hash
    # 一种映射方式，对项集排序，再用"__"连接起来
    def __hash__(self):
        return hash("__".join(sorted([str(itm) for itm in list(self.s)])))

    #重写eq
    # 对排序的项集进行比较
    def __eq__(self, other):
        if "__".join(sorted([str(itm) for itm in list(self.s)])) == "__".join(
                sorted([str(itm) for itm in list(other.s)])):
            return True
        return False

    def isSubnode(self, node):
        return self.s.issubset(node.s)

    # num+1
    def incNum(self, num=1):
        self.num += num

    # 添加node
    def addLnk(self, node):
        self.lnk_nodes[node] = node.s


# 由AprNode和APrLayer构成
class AprBlk():
    def __init__(self, data):
        cnt = 0
        self.apr_layers = dict()
        # data的个数
        self.data_num = len(data)
        for datum in data:
            cnt += 1
            datum = comb(datum)
            nodes = [AprNode(da) for da in datum]
            for node in nodes:
                if not node.size in self.apr_layers:
                    self.apr_layers[node.size] = AprLayer()
                if not node in self.apr_layers[node.size].d:
                    self.apr_layers[node.size].d[node] = node
                self.apr_layers[node.size].d[node].incNum()
            for node in nodes:

                if node.size == 1:
                    continue
                for sn in node.s:
                    # 高阶项集-低阶项集
                    sub_n = AprNode(node.s - set([sn]))
                    # 建立高阶项集和低阶项集的联系
                    self.apr_layers[node.size - 1].d[sub_n].addLnk(node)

    # 获取阈值
    def getFreqItems(self, thd=1, hd=1):
        freq_items = []
        for layer in self.apr_layers:
            for node in self.apr_layers[layer].d:
                if self.apr_layers[layer].d[node].num < thd:
                    continue
                # 项集，项集出现次数
                freq_items.append((self.apr_layers[layer].d[node].s, self.apr_layers[layer].d[node].num))
        freq_items.sort(key=lambda x: x[1], reverse=True)
        # 截取前hd个元素
        return freq_items[:hd]

    # 获取置信度
    def getConf(self, low=True, h_thd=10, l_thd=1, hd=1):
        confidence = []
        for layer in self.apr_layers:
            for node in self.apr_layers[layer].d:
                if self.apr_layers[layer].d[node].num < h_thd:
                    continue
                for lnk_node in node.lnk_nodes:
                    if lnk_node.num < l_thd:
                        continue
                    # 置信度=低阶频繁项集连接高阶频繁项集的数量/低阶频繁项集的数量
                    conf = float(lnk_node.num) / float(node.num)
                    confidence.append([node.s, node.num, lnk_node.s, lnk_node.num, conf])

        confidence.sort(key=lambda x: x[4])
        if low:
            # 提取前hd个
            return confidence[:hd]
        else:
            # 截取从倒数hd个到第一个元素
            return confidence[-hd::-1]


# 关联规则类
class AssctAnaClass():
    def fit(self, data):
        self.apr_blk = AprBlk(data)
        return self

    # 获取阈值
    def get_freq(self, thd=1, hd=1):
        return self.apr_blk.getFreqItems(thd=thd, hd=hd)

    # 获取高置信度项集组合
    def get_conf_high(self, l_thd, h_thd=10):
        return self.apr_blk.getConf(low=False, h_thd=h_thd, l_thd=l_thd)

    # 获取低置信度项集组合
    def get_conf_low(self, h_thd, hd, l_thd=1):
        return self.apr_blk.getConf(h_thd=h_thd, l_thd=l_thd, hd=hd)


def main():
    data = [
        ["牛奶", "啤酒", "尿布"],
        ["牛奶", "啤酒", "咖啡", "尿布"],
        ["香肠", "牛奶", "饼干"],
        ["尿布", "果汁", "啤酒"],
        ["钉子", "啤酒"],
        ["尿布", "毛巾", "香肠"],
        ["啤酒", "毛巾", "尿布", "饼干"]
    ]
    # 输出项集出现的次数
    # thd是最小出现次数(最小阈值)，hd是提取次数
    print("Freq", AssctAnaClass().fit(data).get_freq(thd=3, hd=4))
    # 输出阈值和置信度
    # l_thd是最小出现次数（最小阈值）,h_thd是最大出现次数（最大阈值）
    print("Conf", AssctAnaClass().fit(data).get_conf_high(l_thd=1, h_thd=5))
    # hd是提取次数，l_thd是最小出现次数（最小阈值）,h_thd是最大出现次数（最大阈值）
    print("Conf", AssctAnaClass().fit(data).get_conf_low(h_thd=5, hd=10, l_thd=3))


if __name__ == "__main__":
    main()
