from soweli.tunimi import IloTunimi

def make_tokenizer():
    return IloTunimi('pad:cls:msk:sep:unk:num:prp')

