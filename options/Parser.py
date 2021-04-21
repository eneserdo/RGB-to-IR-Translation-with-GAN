import argparse

class Parser():
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):
        self.arg.add_argument('--dir', default="./", help="Directory")
        self.arg.add_argument('--number',default=10, help="Give a number")

        return self.arg.parse_args()



