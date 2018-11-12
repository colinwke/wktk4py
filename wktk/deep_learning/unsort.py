# @date: 2018/11/8 14:30
# @author: wangke
# @concat: wangkebb@163.com
# =========================

"""deep learning unsort utils."""


class LossKeeper(object):

    def __init__(self, n_loss):
        # init values
        self.n_loss = n_loss
        self.losses = [0.0 for _ in range(self.n_loss)]

        # setting iter2step value
        self.i_iter = 0

    def update(self, *args):
        """update losses"""
        if len(args) != self.n_loss:
            raise ValueError("parameter error!")

        for i in range(self.n_loss):
            self.losses[i] += args[i].item()

        self.i_iter += 1

    def get_avg(self):
        """return average loss and reset."""
        losses = [i / self.i_iter for i in self.losses]

        self.__init__(self.n_loss)
        return losses
