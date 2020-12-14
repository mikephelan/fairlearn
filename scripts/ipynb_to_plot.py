# Licensed under the MIT License.

"""Convert Jupytext Markdown (.ipynb) notebooks to percent format (.py) notebooks, recursively by folder

Usage: python3 ipynb_to_plot.py path-to-top-level-folder
"""

import argparse
import jupytext
import os
import re
import sys

def _build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="the root directory from which to locate files for transformation")
    return parser

def main(argv):
    src_fname_divider_bad = ' '
    tgt_fname_divider = '_'
    src_fname_ext = '.ipynb'
    tgt_fname_ext = '.py'
    tgt_fname_prefix = 'plot'
    tgt_nbook_fmt = 'py:percent'
    tgt_nbook_docstr_div = '"""'
    tgt_nbook_docstr_div_int = '==========================='

    parser = _build_argument_parser()
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
                tgt_nbook = jupytext.writes(src_nbook, fmt=tgt_nbook_fmt)
                with open(os.path.join(folder, tgt_fname), 'w') as f:
                    newlines = []
                    newlines.append(tgt_nbook_docstr_div)
                    newlines.append('\n')
                    newlines.append(tgt_nbook_docstr_div_int)
                    newlines.append('\n')
                    # break up the base filename into capitalized words and append them to the docstring
                    newlines.append('OpenDP ' + " ".join(re.split(tgt_fname_divider, tgt_fname_base.title())))
                    newlines.append('\n')
                    newlines.append(tgt_nbook_docstr_div_int)
                    newlines.append('\n')
                    newlines.append(tgt_nbook_docstr_div)
                    newlines.append('\n')
                    newlines.append('\n')
                    for line in newlines:
                        f.write(line)
                    f.write(tgt_nbook)

if __name__ == "__main__":
    main(sys.argv[1:])
