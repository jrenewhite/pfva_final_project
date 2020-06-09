import os
import argparse
from tqdm import tqdm, trange
from datetime import datetime


def CheckSuffix(file, suffix):
    if not suffix or len(suffix) == 0:
        return True
    if not suffix in file or len(suffix) >= len(file):
        return False
    filename, file_extension = os.path.splitext(file)
    return filename[(len(suffix)*-1):] == suffix


def GetFilesFromDirectory(directory):
    files = []
    if os.path.isdir(directory):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    else:
        raise Exception(f"Directory '{directory}' is not valid.")


def Execute(args):
    # Initialize the array of testing and training dataset files

    # region File lists initialization
    testing_files = []

    if args.input_files:
        testing_files = args.input_files

    if args.input_directory:
        testing_files = list(set().union(testing_files, list(filter(
            lambda file: CheckSuffix(file, args.test_file_suffix), GetFilesFromDirectory(args.input_directory)))))

    testing_files.sort()

    print(
        f"Testing files detected (*{args.test_file_suffix}.[arff|dat]): {len(testing_files)}")

   # endregion

    print("===============================================================================")
    tst = trange(len(testing_files), desc='Testing files...',
                 leave=True, unit="dataset")
    # tqdm(testing_files, desc="Testing patterns...", unit="dataset"):

    now = datetime.now()  # current date and time
    resultsId = now.strftime("%Y%m%d%H%M%S")

    for f in tst:
        tst.set_description(f"Classifying instances from {testing_files[f]}")
        tst.refresh()  # to show immediately the update
        Classify(testing_files[f], args.output_directory, resultsId,
                 args.delete_binary, args.test_file_suffix)
        tst.set_description(f"Results saved for {testing_files[f]}")
        tst.refresh()  # to show immediately the update


if __name__ == '__main__':

    defaultDataDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "data", "python")
    defaultOutputDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "output")

    defaultTrainingFiles = list()
    defaultTestingFiles = list()

    parser = argparse.ArgumentParser(
        description="Implementation through parallel programing the Predictive Factor Variancce Association technique (PFVA). .")

    parser.add_argument("-i", "--input-directory",
                        type=str,
                        metavar="'"+defaultDataDir+"'",
                        default=defaultDataDir,
                        help="the directory with files to be classified")

    parser.add_argument("-o", "--output-directory",
                        type=str,
                        metavar="'"+defaultOutputDir+"'",
                        default=defaultOutputDir,
                        # default="\\root\\parentDir\\dir",
                        help="the output directory for the patterns")

    parser.add_argument("-p", "--processors",
                        type=str,
                        required=True,
                        help="number of processors to use")

    parser.add_argument("-v", "--verbose",
                        type=bool,
                        metavar="<True/False>",
                        default=True,
                        help="states if text shall be printed")

    parser.add_argument("-n", "--no-iterations",
                        type=int,
                        metavar="n",
                        default=100,
                        help="")

    parser.add_argument("-r", "--learning-rate",
                        type=float,
                        metavar="d",
                        default=0.005,
                        help="")

args = parser.parse_args()

print("==========================================================")
print("               _____  ________      __                    ")
print("              |  __ \|  ____\ \    / /\                   ")
print("              | |__) | |__   \ \  / /  \                  ")
print("              |  ___/|  __|   \ \/ / /\ \                 ")
print("              | |    | |       \  / ____ \                ")
print("              |_|    |_|        \/_/    \_\               ")
print("                                                          ")
print("==========================================================")
 print()
  # Arguments received: Uncomment for debbuging purposes
  # print()
  # Uncomment for debbuging purposes
  print()

   if not args.training_files and not args.training_directory and not args.input_files and not args.input_directory:
        parser.print_help()
    else:
        Execute(args)
