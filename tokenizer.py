import collections
import math
import os
import re
import sys
import time
import six
from datetime import datetime
import tqdm


def read_nonempty(filename):
  with open(filename, 'r') as file:
    return [line.strip() for line in file.readlines()
            if line.strip() not in ['', '.']]


def check_mrs(content, i):
  is_mr = (i >= 2 and 
           content[i-2:i].lower() in ['mr', 'ms'] and
           (i < 3 or content[i-3] == ' '))
  is_mrs = (i >= 3 and 
            content[i-3:i].lower() == 'mrs' and 
            (i < 4 or content[i-4] == ' '))
  return is_mr or is_mrs


def check_ABB_mid(content, i):
  if i <= 0:
    return False
  if i >= len(content)-1:
    return False
  l, r = content[i-1], content[i+1]
  return l.isupper() and r.isupper()


def check_ABB_end(content, i):
  if i <= 0:
    return False
  l = content[i-1]
  return l.isupper()


def fix_file(filename):
  if not os.path.exists(filename + '.fixed'):
    with open(filename, 'r') as file:
      contents = file.read()
    contents = fix_contents(contents)

    with open(filename+'.fixed', 'w') as file:
      file.write(contents)

  return filename + '.fixed'


def fix_contents(contents):
  # first step: replace special characters 
  check_list = ['\uFE16', '\uFE15', '\u0027','\u2018', '\u2019',
                '“', '”', '\u3164', '\u1160', 
                '\u0022', '\u201c', '\u201d', '"',
                '[', '\ufe47', '(', '\u208d',
                ']', '\ufe48', ')' , '\u208e', 
                '—', '_', '–', '&']
  alter_chars = ['?', '!', '&apos;', '&apos;', '&apos;',
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&#91;', '&#91;', '&#91;', '&#91;',
                 '&#93;', '&#93;', '&#93;', '&#93;', 
                 '-', '-', '-', '&amp;']

  replace_dict = dict(zip(check_list, alter_chars))

  print('[1/4]')
  new_contents = ''
  for i, char in tqdm.tqdm(enumerate(contents), total=len(contents)):
    if char == '&' and (contents[i:i+5] == '&amp;' or
                        contents[i:i+6] == '&quot;' or
                        contents[i:i+6] == '&apos;' or
                        contents[i:i+5] == '&#93;' or
                        contents[i:i+5] == '&#91;'):
      new_contents += char
      continue
    new_contents += replace_dict.get(char, char)
  contents = new_contents

  # second: add spaces
  check_sp_list = [',', '?', '!', '&apos;', '&amp;', '&quot;', '&#91;', 
                   '&#93;', '-', '/', '%', ':', '$', '#', '&', '*', ';', '=', '+', '$', '#', '@', '~', '>', '<']
  
  print('[2/4]')
  new_contents = ''
  i = 0
  l100 = len(contents)//100
  while i < len(contents):
    if i // l100 > (i-1) // l100:
      sys.stdout.write(str(i // l100) + '%')
      sys.stdout.flush()
    char = contents[i]
    found = False
    for string in check_sp_list:
      if string == contents[i: i+len(string)]:
        new_contents += ' ' + string 
        if string != '&apos;':
          new_contents += ' '
        i += len(string)
        found = True
        break
    if not found:
      new_contents += char
      i += 1
  contents = new_contents

  print('[3/4]')
  new_contents = ''
  for i, char in tqdm.tqdm(enumerate(contents), total=len(contents)):
    if char != '.':
      new_contents += char
      continue
    elif check_mrs(contents, i):
      # case 1: Mr. Mrs. Ms.
      new_contents += '. '
    elif check_ABB_mid(contents, i):
      # case 2: U[.]S.A.
      new_contents += '.'
    elif check_ABB_end(contents, i):
      # case 3: U.S.A[.]
      new_contents += '. '
    else:
      new_contents += ' . '

  contents = new_contents
  
  # third: remove not necessary spaces.
  print('[4/4]')
  new_contents = ''
  for char in tqdm.tqdm(contents):
    if new_contents and new_contents[-1] == ' ' and char == ' ':
      continue
    new_contents += char
  contents = new_contents
  
  return contents.strip()


def is_number(char):
  chars = {'0':1, '1':1, '2':1, '3':1, '4':1,
           '5':1, '6':1, '7':1, '8':1, '9':1}
  if char in chars:
    return True
  return False


def test_isnumber():
  string = ' 39. 8 %, 15. 1% ; 28% ; 38. 7% ; and 26. 9% ' \
            + 'respectively. The prevalences of osteopenia at ' \
            + 'those ROI are 54. 8% ; 46. 3% ; 60. 2% ; 45. 2% ' \
            + 'and 62. 7% respectively .'
  for char in string:
    if is_number(char):
      print(char)

# test_isnumber()

def is_upper(char):
  if char == char.upper():
    return True
  return False


def is_char(char):
  chars = {'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1,
           'g':1, 'h':1, 'i':1, 'j':1, 'k':1, 'l':1,
           'm':1, 'n':1, 'o':1, 'p':1, 'q':1, 'r':1,
           's':1, 't':1, 'u':1, 'v':1, 'w':1, 'x':1,
           'y':1, 'z':1}
  if char in chars:
    return True
  return False


def is_end_of_quote(char):
  if not is_char(char) or not is_upper(char):
    return True
  return False


def unfix_contents(contents):
  check_list = ['&apos;', '&quot;', '&#91;',
                '&#93;', '&lt;', '&gt;',
                ' . ', ' , ',
                '&amp;'
                ]
  alter_chars = ['\u0027', '"', '(', 
                 ')', '<', '>',
                 '. ', ', ',
                 '&']

  replace_dict = dict(zip(check_list, alter_chars))
  new_contents = ''
  i = 0
  while i < len(contents):
    char = contents[i]
    found = False
    for string in replace_dict:
      if string == contents[i: i+len(string)]:
        new_contents += replace_dict[string]
        i += len(string)
        found = True
        break
    
    if not found:
      new_contents += char
      i += 1
  
  contents = new_contents
  
  #fix n 't 
  check_list1 = ["n 't ", " 've ", " 'd ", " 'm ", 
                " 're ", " 's ", " 'll ", "s ' ",
                ' ( ',' ) ', ' ; ', ' % ', ' / ', ' : ',
                ' + - ', ' - ', ' ? ' ]
  alter_chars1 = ["n't ", "'ve ", "'d ", "'m ", 
                "'re ", "'s ", "'ll ", "s' ",
                ' (',') ', '; ', '% ', '/', ': ',
                ' +- ', '-', '? ']
  new_contents = ''
  replace_dict1 = dict(zip(check_list1, alter_chars1))
  i = 0

  while i < len(contents):
    char = contents[i]
    found = False
    for string in replace_dict1:
      if string == contents[i: i+len(string)]:
        new_contents += replace_dict1[string]
        i += len(string)
        found = True
        break
    
    if not found:
      new_contents += char
      i += 1
  
  contents = new_contents

  # rm space in number eg: 4. 5 -> 4.5 or 4, 6 -> 4,6 //how abt 12. 2. 1. ... ?
  i = 0
  new_contents = ''
  while i < len(contents):
    char = contents[i]
    found = False
    if char == '.' or char == ',':
      if is_number(contents[i-1]) and \
          contents[i+1] == ' ' and \
          is_number(contents[i+2]):
        new_contents += (char + contents[i+2])
        i += 3
        found = True
  
    if not found:
      new_contents += char
      i += 1
  
  contents = new_contents

  #rm ' .', ' ?', ' !', ' ;'
  i = 0
  new_contents = ''
  while i < len(contents):
    char = contents[i]
    found = False
    fixchars = ['.', '!', '?', ';']
    if char == ' ' and contents[i+1] in fixchars:
      found = True
      new_contents += contents[i+1]
      i += 2
    
    if not found:
      new_contents += char
      i += 1
  
  contents = new_contents

  #rm E. coli
  i = 0
  new_contents = ''
  while i < len(contents):
    char = contents[i]
    found = False
    #end quote
    if i < len(contents) - 2 and char == '.' and \
        is_upper(contents[i-1]) and \
        contents[i+1] == ' ' and \
        not is_upper(contents[i+2]):
      found = True
      new_contents += char + contents[i+2]
      i += 3
    if not found:
      new_contents += char
      i += 1
  
  contents = new_contents

  return contents.strip()


def unfix_file(fname):
    org_place = os.getcwd() + '/'
    saving_place = os.getcwd() + '/'

    if not os.path.exists(saving_place + fname + '.unfix'):
        print('unfixing ', fname)
        with open(org_place + fname, 'r') as file:
            contents = file.read()
        contents = unfix_contents(contents)

        with open(saving_place + fname + '.unfix', 'w') as file:
            file.write(contents)

        return fname + '.unfix'
    else:
        print('already done')
