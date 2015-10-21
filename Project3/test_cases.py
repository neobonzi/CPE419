#!/usr/bin/python

from subprocess import call

# make single-precision
call(['make','clean'])
call(['make','single'])
print 

# Test 1
call(['./mm','input/1408.in', 'input/1408.in'])
print "result [1408x1408]x[1408x1408]:"
call(['./val_compare.py','result.out', 'output/1408.out'])
call(['rm','result.out'])
print 

# Test 2
call(['./mm','input/555x666.in', 'input/666x777.in'])
print "result [555x666]x[666x777]:"
call(['./val_compare.py','result.out', 'output/555x777.out'])
call(['rm','result.out'])
print

# Test 3
call(['./mm','input/1x3.in', 'input/3x1.in'])
print "result [1x3]x[3x1]:"
call(['./val_compare.py','result.out', 'output/1x1.out'])
call(['rm','result.out'])
print

# Test 4
call(['./mm','input/4x3.in', 'input/3x2.in'])
print "result [4x3]x[3x2]:"
call(['./val_compare.py','result.out', 'output/4x2.out'])
call(['rm','result.out'])
print

