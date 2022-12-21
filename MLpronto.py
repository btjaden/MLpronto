import os, sys, json, subprocess
from inspect import getmembers, isfunction
import mapper


#####################
#####   USAGE   #####
#####################

if len(sys.argv) < 3:
	sys.stderr.write("\nUSAGE: MLpronto.py <datafile> <parameters.json>" + "\n\n")
	sys.stderr.write("MLpronto requires two command line arguments. MLpronto takes a data file (.csv, .tsv, .txt, .xls, .xlsx, .xlsm, .xlsb, .ods) and a JSON file (indicating ML parameter options) and it generates a file with ML code that can be executed to analyze the data file. Output is to a file with the same name as the input data file but with *x_* appended on the front and the file extension replaced with *.py*.\n\n")
	sys.exit(1)



#########################
#####   FUNCTIONS   #####
#########################

def verifyFile(filename):
        if (os.path.isfile(filename)): return filename
        else:
                sys.stderr.write('\nError - unable to locate the file ' + filename + '\n')
                sys.stderr.write('Program exiting.\n\n')
                sys.exit(1)



def generateCode(data_filename, blocks):
        # Name of code file
        head, tail = os.path.split(data_filename)
        code_filename = head + os.sep + 'x_' + os.path.splitext(tail)[0] + '.py'

        # Create dictionary mapping function name (string) to function object
        functions_list = getmembers(mapper, isfunction)
        functions = {}
        for f in functions_list: functions[f[0]] = f[1]

        with open(code_filename, 'w') as out_file:
                # Write header, e.g., libraries
                for b in blocks: out_file.write(functions[b]('header', params))

                # Write body, e.g., code
                for b in blocks: out_file.write(functions[b]('body', params))

        sys.stdout.write('\n' + 'A file of ML code has been generated:\t' + code_filename + '\n')
        return code_filename



def executeCode(code_filename, data_filename):
        sys.stdout.write('Results of executing the code file...' + '\n')
        currentDir = os.getcwd()
        os.chdir(os.path.split(data_filename)[0])
        p = subprocess.run(['python', os.path.split(code_filename)[1]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir(currentDir)

        # Check if there was any error when executing the code file
        if (p.returncode != 0) or (len(p.stderr.decode()) > 0):
                sys.stderr.write('Error - the code file that was generated could not be executed successfully on the data file.\n')
                if (len(p.stderr.decode()) > 0):
                        sys.stderr.write('Executing the code file caused the following error:\n\n')
                        sys.stderr.write(p.stderr.decode() + '\n\n')
        else:
                sys.stdout.write('\n' + p.stdout.decode() + '\n\n')



##############################
##########   MAIN   ##########
##############################

# CHECK THAT FILES EXIST
data_filename = verifyFile(sys.argv[1])
parameter_filename = verifyFile(sys.argv[2])

# LOAD PARAMETER OPTIONS FROM JSON FILE
with open(parameter_filename, 'r') as f: params = json.load(f)  # Load parameters
params['filename_temp'] = os.path.split(data_filename)[1]  # Temp data file name

# DEFINE SEQUENCE OF CODE BLOCKS TO EXECUTE
blocks = ['header',
          'libraries',
          'body',
          'preprocessing',
          'read_data',
          'remove_rows',
          'labels_contain_nan',
          'labels_to_y',
          'labels_non_numeric',
          'remove_cols',
          'encode_binary_features',
          'one_hot_encode_features',
          'features_to_X',
          'univariate_imputation',
          'multivariate_imputation',
          'split_data',
          'feature_scaling',
          'model',
          'train_and_predict',
          'evaluate',
          'output',
          'random_seed']

# GENERATE CODE
code_filename = generateCode(data_filename, blocks)

# EXECUTE CODE
executeCode(code_filename, data_filename)

