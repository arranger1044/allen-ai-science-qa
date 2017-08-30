import os

import pandas

from dataset import collect_document_paths

FEATURE_PATH = '/media/valerio/formalità/tot-scores/tano_feature/noneg/'
COLS = ['qid', 'aid', 'f3']

features_paths = collect_document_paths(FEATURE_PATH, '\.feature')

OUT_DIR = '/media/valerio/formalità/tot-scores/tano_f3/noneg/'
os.makedirs(OUT_DIR, exist_ok=True)

new_ext = '_'.join(str(col) for col in COLS)

for path in features_paths:
    print('Processing path {}'.format(path))
    base_dir = os.path.dirname(path)
    file_name = os.path.basename(path)
    feature_frame = pandas.read_csv(path)
    feature_frame_selected = feature_frame[COLS]

    out_path = os.path.join(OUT_DIR, file_name + '.' + new_ext)
    feature_frame_selected.to_csv(out_path, index=False)
