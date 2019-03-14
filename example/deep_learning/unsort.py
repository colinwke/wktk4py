# @date: 2018/11/12 22:13
# @author: wangke
# @concat: wangkebb@163.com
# =========================
from wktk.deep_learning.unsort import LossKeeper


def main():
    loss_keeper = LossKeeper()

    for i in range(100):
        loss_keeper.update(i, i + 1, i + 2, i + 3, i + 4)
        print(i, i + 1, i + 2, i + 3, i + 4)

        if i % 10 == 0:
            print('%s, %s, %s, %s, %s, %s' % (i, *tuple(loss_keeper.get_avg())))


if __name__ == '__main__':
    main()
