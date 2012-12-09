#!/usr/bin/python

import fileinput

i = 0
file_pattern = "./%(grade)i/%(num)i.txt"

def uni(input):
  r = ''
  try:
    r = str(input).decode('ascii', 'ignore')
  except Exception, e:   
    raise e
  return r

for line in fileinput.input():
  i = i + 1
  essay,grade = line.split("\t")
  f = file(file_pattern % { 'grade': int(grade), 'num': i }, 'w')
  f.write(uni(essay))
