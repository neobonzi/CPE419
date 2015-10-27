#!/usr/bin/python

from subprocess import call

# make single-precision
call(['make','clean'])
call(['make', 'vectoranalyze'])
print

# Test 1
print "Building vector analyze..."
call(['./vectoranalyze'])

print "Comparing output for results.out"
call(['diff','analyze.out', 'output/result1.out'])
#call(['rm','result.out'])


print



