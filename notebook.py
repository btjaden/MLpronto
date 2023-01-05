######################################
#####   MLpronto version 1.0.1   #####
######################################

import sys, pathlib
import nbformat as nbf


#########################
#####   FUNCTIONS   #####
#########################

def outputNotebook(python_filename):
        gold = '<font color="goldenrod">'

        # Read in Python file
        nb = nbf.v4.new_notebook()
        nb['cells'] = [nbf.v4.new_markdown_cell('<H1><center><font color="indigo">Notebook created by</font> <font color="goldenrod">MLpronto</font></center></H1><img src="https://cs.wellesley.edu/~btjaden/MLpronto/img/H3.png" width="75"/>')]
        with open(python_filename, 'r') as in_file: lines = in_file.readlines()
        i = 0
        blocks = []
        while (i < len(lines)):
                # Get block of text (either comments or code) as separated by blank lines
                while (i < len(lines)) and (lines[i].strip() == ''): i += 1
                block = []
                while (i < len(lines)) and (lines[i].strip() != ''):
                        block.append(lines[i])
                        i += 1
                if (len(block) > 0): blocks.append(block)

        # Convert block to notebook cell
        for block in blocks:
                # Block is a one line comment
                if (len(block) == 1) and (block[0].startswith('#')):
                        comment = '## ' + gold + block[0][1:].strip() + '</font>'
                        nb['cells'].append(nbf.v4.new_markdown_cell(comment))

                # Block is one line of code
                elif (len(block) == 1) and (not block[0].startswith('#')):
                        nb['cells'].append(nbf.v4.new_code_cell(block[0]))

                # Block is multiple lines of comments
                elif (block[0].startswith('#')) and (block[1].startswith('#')):
                        comment = ''.join(block).replace('#', '').strip()
                        if (comment.lower() == 'header'):
                                nb['cells'].append(nbf.v4.new_markdown_cell('## ' + gold + 'HEADER' + '</font>'))

                # First line is a comment and remaining lines are code
                elif (block[0].startswith('#')) and (not block[1].startswith('#')):
                        comment = '## ' + gold + block[0][1:].strip() + '</font>'
                        nb['cells'].append(nbf.v4.new_markdown_cell(comment))
                        nb['cells'].append(nbf.v4.new_code_cell(''.join(block[1:])))

                # All lines are code
                elif (not block[0].startswith('#')) and (not block[1].startswith('#')):
                        nb['cells'].append(nbf.v4.new_code_cell(''.join(block)))

                else: None  # Not sure about this case

        # Output notebook
        notebook_filename = str(pathlib.PurePath(python_filename).with_suffix('.ipynb'))
        nbf.write(nb, notebook_filename)



##############################
##########   MAIN   ##########
##############################

if __name__ == '__main__':
        if len(sys.argv) < 2:
                sys.stderr.write("\nUSAGE: notebook.py <*.py>" + "\n\n")
                sys.stderr.write("notebook takes a Python file as input and outputs an equivalent IPython Notebook. The notebook has the same name as the Python file but with the file extension .ipynb.\n\n")
                sys.exit(1)

        outputNotebook(sys.argv[1])

