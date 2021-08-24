# EPEM: Efficient Parameter Estimation for Multiple Class Monotone Missing Data

This repo contains scripts to reproduce the results in the paper:

"EPEM: Efficient Parameter Estimation for Multiple Class Monotone Missing Data",

which is published in <a href='https://www.sciencedirect.com/science/article/abs/pii/S0020025521002346'>
Information Sciences</a>.

<a href='https://youtu.be/I8oKiTyR0QU'> A video on motivation and data partition.</a> 

# Usage
The notebooks are created by Google's Colaboratory.

[The "Parameter estimation error" notebook](https://github.com/thunguyen177/EPEM/blob/master/Parameter%20estimation%20error.ipynb) produces the results in *Table 2: Parameters estimation errors with different missing rates*, except for MNIST, fashion MNIST. Meanwhile, the notebook *Parameter estimation error shuffled* produces the results on *Table B.5: Parameters estimation errors with different missing rates on shuffled data* in the Appendix except for MNIST, fashion MNIST.

[The "application in linear discriminant analysis" notebook](https://github.com/thunguyen177/EPEM/blob/master/application%20in%20linear%20discriminant%20analysis.ipynb)  produces the results in *Table 3: The cross-validation errors on datasets with different missing rates in LDA application*, except for MNIST, fashion MNIST.

The folder "parameter estimation MNIST _ Fashion MNIST" contains the codes that produce the results in *Table 2: Parameters estimation errors with different missing rates* for MNIST, fashion MNIST.

The folder "LDA on MNIST_ Fashion MNISTT" contains the codes that produce the results in *Table 3: The cross-validation errors on datasets with different missing rates in LDA application* for MNIST, fashion MNIST.

## References
We recommend you to cite our following paper when using these codes for further investigation:
```bash
@inproceedings{hieu2020,
  title={EPEM: Efficient Parameter Estimation for Multiple Class Monotone Missing Data},
  author={Thu Nguyen, Duy H. M. Nguyen, Huy Nguyen, Binh T. Nguyen, Bruce A. Wade },
  booktitle={Under Submission},
  year={2020}
}
```
Further requests can directly be sent to the corresponding authors: Thu Nguyen (thu.nguyen@louisiana.edu) and Binh T. Nguyen (ngtbinh@hcmus.edu.vn) for an appropriate permission.
