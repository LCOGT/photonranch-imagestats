# handler.py

try:
    import unzip_requirements
except ImportError:
    pass

import numpy as np


def main(event,context):
    a = np.arange(15).reshape(3,5)

    print("Your numpy array:")
    print(a)


if __name__=="__main__":
    main('','')
