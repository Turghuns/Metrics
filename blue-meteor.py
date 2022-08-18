'''
This Program is developed to run pretrained model on a test dataset for our VAG-NMT
'''

from bleu import compute_bleu
import os
from nmtpytorch.cocoeval import Meteor

# Define the Directory of the Test Data Path
src_path = 'results/src.en'
tgt_path = 'results/ref2016.de'
hyp_path = 'results/hyp2016.de'

target_language = 'de'


def read_text_file(filename):
    with open(filename, 'r') as f:
        output = [line.strip() for line in f]
    return output


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()
    return data


# Initilalize a Meteor Scorer
Meteor_Scorer = Meteor(target_language)

# Load the original test dataset
test_source = load_data(os.path.join(src_path))
test_target = load_data(os.path.join(tgt_path))
hyp = read_text_file(os.path.join(hyp_path))
assert len(test_target) == len(hyp), 'The source and hypotheses  must be the same size'
print('The size of Test Source is: {}'.format(len(test_source)))
print('The size of Test Target is: {}'.format(len(test_target)))

# Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]] for original data
test_data = [[x.strip(), y.strip()] for x, y in zip(test_source, test_target)]

hyp = [d.split() for d in hyp]
test_translations = hyp

test_y_ref = [[d[1].split()] for d in test_data]
# Define the test data
test_y_ref_meteor = dict((key, [value[1]]) for key, value in enumerate(test_data))

# Compute the test bleu score and test meteor score
test_bleu = compute_bleu(test_y_ref, test_translations)
# Compute the METEOR Score
test_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(test_translations))
test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, test_translations_meteor)

print("Bleu: {}".format(test_bleu[0]))
print("Meteor: {}".format(test_meteor[0]))
