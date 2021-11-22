import sys
import os
import re
import json

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

result = []
for data_point in data:
    old_tokens = re.split(r'\s', data_point['text'])
    tokens = []
    for tok in old_tokens:
        if '(' in tok and ')' in tok:
            tokens.extend([tok[:tok.index('(') + 1], tok[tok.index('(') + 1:tok.index(')')], tok[tok.index(')'):]])
        elif '(' in tok:
            tokens.extend([tok[:tok.index('(') + 1], tok[tok.index('('):]])
        elif ')' in tok:
            tokens.extend([tok[:tok.index(')')], tok[tok.index(')'):]])
        elif '/' in tok:
            lst = []
            prev = -1
            for i, c in enumerate(tok):
                if prev == -1 and c == '/':
                    lst.extend([tok[:i], '/'])
                    prev = i + 1
                elif c == '/':
                    lst.extend([tok[prev:i], '/'])
                    prev = i + 1
            if prev < len(tok):
                lst.append(tok[prev:])
            tokens.extend(lst)
        else:
            tokens.append(tok)
    labels = ['O'] * len(tokens)
    for s_a in data_point['acronyms']:
        char_count = 0
        for index, token in enumerate(tokens):
            print(token, char_count, index)
            if char_count <= s_a[0] and char_count + len(token) >= s_a[1]:
                labels[index] = 'B-short'
                break
            char_count += len(token)
            if token != ')' and token != '(' and token != '/' and index < len(tokens) - 1 and tokens[index + 1] != '/':
                char_count += 1
    for l_a in data_point['long-forms']:
        char_count = 0
        start = 0
        for index, token in enumerate(tokens):
            if char_count <= l_a[0] and char_count + len(token) >= l_a[0] and start == 0:
                # print(char_count, index)
                labels[index] = 'B-long'
                start = 1
            elif char_count + len(token) >= l_a[1] and start == 1:
                labels[index] = 'I-long'
                start = 0
                break
            elif start == 1 and char_count + len(token) <= l_a[1]:
                labels[index] = 'I-long'
                if char_count + len(token) == l_a[1]:
                    start = 0
                    break
            
            char_count += len(token)
            if token != ')' and token != '(' and token != '/' and index < len(tokens) - 1 and tokens[index + 1] != '/':
                char_count += 1
    _res = {"id": data_point['ID'], "tokens": tokens, "labels": labels}
    result.append(_res)

file_list = sys.argv[1].split('/')
dir = "/".join(file_list[:-1])

with open(f'{dir}/scidr_{file_list[-1]}', 'w') as f:
    json.dump(result, f, indent=2)