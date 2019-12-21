from sparsesvd import sparsesvd

from docopt import docopt
import numpy as np

from representations.explicit import PositiveExplicit
from representations.matrix_serializer import save_vocabulary
# from scipy.sparse.linalg import svds


def main():
    args = docopt("""
    Usage:
        pmi2svd.py [options] <pmi_path> <output_path>
    
    Options:
        --dim NUM    Dimensionality of eigenvectors [default: 500]
        --neg NUM    Number of negative samples; subtracts its log from PMI [default: 1]
    """)
    
    pmi_path = args['<pmi_path>']
    output_path = args['<output_path>']
    dim = int(args['--dim'])
    neg = int(args['--neg'])
    
    explicit = PositiveExplicit(pmi_path, normalize=False, neg=neg)

    ut, s, vt = sparsesvd(explicit.m.tocsc(), dim)
    # sparse_m = explicit.m.tocsc()
    # k = min(dim, max(1, min(sparse_m.shape[0], sparse_m.shape[1])))
    # ut, s, vt = svds(sparse_m, k=k)

    np.save(output_path + '.ut.npy', ut)
    np.save(output_path + '.s.npy', s)
    np.save(output_path + '.vt.npy', vt)
    save_vocabulary(output_path + '.words.vocab', explicit.iw)
    save_vocabulary(output_path + '.contexts.vocab', explicit.ic)


if __name__ == '__main__':
    main()
