# @date: 2018/11/8 14:30
# @author: wangke
# @concat: wangkebb@163.com
# =========================

"""deep learning unsort utils."""


class LossKeeper(object):

    def __init__(self):
        # init values
        self.n_loss = 0
        # setting iter2step value
        self.i_iter = 0

    def update(self, *args):
        """update losses"""
        len_args = len(args)

        if self.n_loss == 0:
            self.n_loss = len_args
            self.losses = [0 for _ in range(self.n_loss)]
        elif self.n_loss != len_args:
            raise ValueError("number of loss not match!")

        self.losses = [self.losses[i] + args[i] for i in range(self.n_loss)]
        self.i_iter += 1

    def get_avg(self):
        """return average loss and reset."""
        losses = [i / self.i_iter for i in self.losses]

        self.__init__()
        return losses
