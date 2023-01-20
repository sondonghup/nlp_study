import argparse
import os
import shutil
import random
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="data directory", default="../data/")
parser.add_argument("--max_len", help="max sequence length", default=10)
args = parser.parse_args()
def generate_dataset(data_path, name, size):
    # path = os.path.join(root, name)

    if not os.path.isdir(data_path):
        print("fa")
        os.mkdir(data_path)
    # generate data file
    print(data_path)
    with open(data_path+name+".txt", "w") as fout:
        for _ in range(size):
            length = random.randint(1, args.max_len)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            fout.write("\t".join([" ".join(seq), " ".join(reversed(seq))]))
            fout.write("\n")
    # generate vocabulary
    src_vocab = os.path.join(data_path, "vocab.source")
    with open(src_vocab, "w") as fout:
        fout.write("\n".join([str(i) for i in range(10)]))
    tgt_vocab = os.path.join(data_path, "vocab.target")
    shutil.copy(src_vocab, tgt_vocab)
if __name__ == "__main__":
    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    toy_dir = os.path.join(data_dir, "toy_reverse")
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)
    print(toy_dir)
    generate_dataset("/Users/sondo/study/data/train/", "train", 10000)
    generate_dataset("/Users/sondo/study/data/dev/", "dev", 1000)
    generate_dataset("/Users/sondo/study/data/test/", "test", 1000)