#!/usr/bin/python

from subprocess import call

# make single-precision
call(['make','clean'])
call(['make','single'])
print 

# Test 1
call(['./vectorsum','/home/clupo/vectors/10000.a', '/home/clupo/vectors/10000.b'])
print "Running test for 10000.a + 10000.b:"
call(['diff','result.out', 'output/10000ab.out'])
call(['rm','result.out'])
print 



