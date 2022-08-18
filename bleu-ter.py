import sacrebleu

hypo_path = 'xxxxx'  # hypotheses path
ref_path = 'yyyyyy'  # references path


def read_text_file(filename):
    with open(filename, 'r') as f:
        output = [line.strip() for line in f]
    return output


def eval_metric(metric, hypos, ref):
    if metric == "bleu":

        scores = sacrebleu.corpus_bleu(hypos, [ref]).score
    else:
        scores = sacrebleu.corpus_ter(hypos, [ref]).score
    return scores


hypo = read_text_file(hypo_path)
ref = read_text_file(ref_path)

Bleu = eval_metric('bleu', hypo, ref)
Ter = eval_metric('ter', hypo, ref)

print("Bleu: {}".format(Bleu))
print("Meteor: {}".format(Ter))

