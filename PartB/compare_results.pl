#!/usr/bin/perl

use strict;
use Test::Number::Delta within => 1.0, tests => 1;

my $goldfile = shift;
my <file = shift;
my @gold;
my @result;

# open file
open(GOLD, "$goldfile" ) or die("Unable to open $goldfile");
open(TEST, "<file" ) or die("Unable to open <file");

# read file into an array
while (<GOLD>) {
  chomp;
    @gold = split(/ /,$_); 
    }

    while (<TEST>) {
      chomp;
        @result = split(/ /,$_);
        }

        foreach my $g (@gold) {
        }

        delta_ok( \@gold, \@result, 'Values within tolerance.');

        # close file 
        close(GOLD);
        close(TEST);
