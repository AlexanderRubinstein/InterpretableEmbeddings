from docopt import docopt
from scipy.stats.stats import spearmanr

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        ws_eval.py [options] <representation> <representation_path> <task_path>
    
    Options:
        --neg NUM    Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c        Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
        --normalize  Use row-normalized word vectors
    """)

    representation = create_representation(args)
    data = read_test_set(representation, args['<task_path>'])
    correlation = evaluate(representation, data)
    print args['<representation>'], args['<representation_path>'], '\t%0.3f' % correlation


def read_test_set(representation, path):
    test = []
    unks = 0
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            if x not in representation.wi or y not in representation.wi:
                unks += 1
                continue
            test.append(((x, y), float(sim)))
    print 'skipped ' + str(unks) + ' unk sets'
    return test


def evaluate(representation, data):
    results = []
    for (x, y), sim in data:
        results.append((representation.similarity(x, y), sim))
    actual, expected = zip(*results)

    return spearmanr(actual, expected)[0]

if __name__ == '__main__':
    main()
