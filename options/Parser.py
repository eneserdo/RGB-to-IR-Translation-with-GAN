import argparse

class Parser():
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):
        self.arg.add_argument('--dir', default="./", help="Directory")
        self.arg.add_argument('--number',default=10, help="Give a number")
        self.arg.add_argument('--cont_train',default=False, help="Continue the previous training")
        self.arg.add_argument('--transposed',default=False, help="Use transposed convolution")
        self.arg.add_argument('--loss',default='lsgan', help="Enter the loss type: lsgan or gan")

        return self.arg.parse_args()



