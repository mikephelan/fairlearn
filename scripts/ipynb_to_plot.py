import argparse
import jupytext
import os
import re

src_fname_divider_bad = ' '
tgt_fname_divider = '_'
src_fname_ext = '.ipynb'
tgt_fname_ext = '.py'
tgt_fname_prefix = 'plot'
tgt_nbook_fmt = 'py:percent'

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="the root directory from which to locate files for transformation")
args = parser.parse_args()

for folder, subFolders, files in os.walk(args.directory):
    for filename in files:
        tgt_fname_base = os.path.splitext(filename)[:1][0]
        # retreive the last item of the filename tuple, then extract the zeroeth (and only) item for comparison
        if(os.path.splitext(filename)[1:][0] == src_fname_ext):
            # replace all filename spaces with underscores
            if(src_fname_divider_bad in tgt_fname_base):
                tgt_fname_base = tgt_fname_base.replace(src_fname_divider_bad, tgt_fname_divider)
            # define the target filename using the conventions of fairlearn/autodoc
            tgt_fname = tgt_fname_prefix + tgt_fname_divider + tgt_fname_base + tgt_fname_ext
            src_nbook = jupytext.read(folder + os.sep + filename)
            jupytext.write(src_nbook, tgt_fname, fmt=tgt_nbook_fmt)


