from NiaPy.algorithms.basic import GreyWolfOptimizer

# our custom benchmark classs
class MyBenchmark(object):
    def __init__(self):
        # define lower bound of benchmark function
        self.Lower = -11
        # define upper bound of benchmark function
        self.Upper = 11

    # function which returns evaluate function
    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate

for i in range(10):

    algorithm = GreyWolfOptimizer(D=10, NP=100, nFES=10000, benchmark=MyBenchmark())
    best = algorithm.run()

    print(best[1])