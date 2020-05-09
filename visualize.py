import argparse
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='file w/ episode values to visualize')
    args = parser.parse_args()

    f = open(args.file, 'rb')
    vals = pickle.load(f)
    f.close()
 
    plt.plot(vals[1:])
    plt.ylabel("Account value at end of episode")
    plt.xlabel("Episode")
    plt.axhline(y=vals[0], color='g', linestyle='-')
    plt.show()
