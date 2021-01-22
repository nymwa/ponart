import pickle
from ponart.util import make_tokenizer

def load_corpus():
    with open('soweli/corpus/tatoeba/tatoeba.txt') as f:
        lst = f.readlines()
    return lst

def main():
    tokenizer = make_tokenizer()
    sent_list = [tokenizer(sent) for sent in load_corpus()]
    with open('data/tatoeba.pickle', 'wb') as f:
        pickle.dump(sent_list, f)

if __name__ == '__main__':
    main()

