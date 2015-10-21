#!/usr/bin/python

'''
script to compare decimal values. Run as ./compare.py [file1] [file2].
'''

import sys

MAX_DIFF = 1

def main():

   ## Only run if there are two file names as input.
   if len(sys.argv) == 3:

      ## Attempt to open the two input files.
      try:
         file_1 = open(sys.argv[1], "r")
         file_2 = open(sys.argv[2], "r")

         line_count = 0
         value_count = 0

         ## Loop over each line in both files
         for line1, line2 in zip(file_1, file_2):

            ## Quick check to ensure that both lines have the same number of elements
            if len(line1.split()) == len(line2.split()):

               ## Loop over each pair of elements in each set of lines
               for value1, value2 in zip(line1.split(), line2.split()):

                  ## Fail if the two values aren't within tolerance.
                  if abs(float(value1) - float(value2)) > MAX_DIFF:
                     print(value1 + " and " + value2 + " at line %d " %(line_count)\
                     + "index %d " %(value_count) + "are outside the "\
                     + "tolerance of %d." %(MAX_DIFF))
                     sys.exit()

                  value_count += 1

            ## If the two lines to be compared are of unequal element count, exit.
            else:
               print("Line %d of the two files have different lengths, exiting." %(line_count))
               sys.exit()

            line_count += 1
            value_count = 0

         file_1.close()
         file_2.close()

         print("All values within both files are within a tolerance of %d." %(MAX_DIFF))

      except IOError:
         print "Unable to open input files, exiting."

   ## If incorrect number of inputs, print error and exit.
   else:
         print "Run as compare.py [file1] [file2]."

if __name__ == "__main__":
   main()
