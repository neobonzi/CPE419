#!/usr/bin/python

from subprocess import call

# make single-precision
call(['make','clean'])
call(['make'])
print 

# Test 1
print "Building files..."
call(['./vectorsum','/home/clupo/vectors/10000.a', '/home/clupo/vectors/10000.b'])

print "Comparing output for results.out"
call(['diff','result.out', 'output/10000ab.out'])
call(['rm','result.out'])

print "Comparing output for hist.a"
call(['diff','hist.a', 'output/hist.a'])
call(['rm','hist.a'])

print "Comparing output for hist.b"
call(['diff','hist.b', 'output/hist.b'])
call(['rm','hist.b'])

print "Comparing output for hist.c"
call(['diff','hist.c', 'output/hist.c'])
call(['rm','hist.c'])

print 



