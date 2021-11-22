import pandas as pd
import os
from tqdm import tqdm

d = {'add': 0, 'or': 0, 'callq': 0, 'push': 0, 'nopl': 0, 'sub': 0,
     'sete': 0, 'cmove': 0, 'cmovne': 0, 'mov': 0,
     'nopw': 0, 'xchg': 0,
     'movups': 0, 'imul': 0, 'setl': 0, 'cmovl': 0, 'movabs': 0, 'movl': 0,
     'pop': 0, 'jge': 0, 'cmp': 0, 'setne': 0, 'xor': 0, 'test': 0, 'jne': 0,
     'and': 0, 'je': 0, 'jmpq': 0}

colums = ['size','add', 'or', 'callq', 'push', 'nopl', 'sub', 'sete', 'cmove', 'cmovne', 'mov', 'nopw', 'xchg', 'movups', 'imul', 'setl', 'cmovl', 'movabs', 'movl', 'pop', 'jge', 'cmp', 'setne', 'xor', 'test', 'jne', 'and', 'je', 'jmpq']



def parsing_asm(path, code = d):
    opcode = dict(code)
    f = open(path,encoding='UTF8')
    line = f.readline()
    size = os.path.getsize(path)
    while line:
        s = line[0]
        if s == ' ':
            word = line.split("\t")
            if len(word) > 2:
                op = word[-1].split(" ")[0]
                if op in opcode.keys():
                    opcode[op] += 1

        line = f.readline()

    v = list(opcode.values())
    result = [size] + v

    return result

if __name__ == '__main__':
    a = [colums]

    for root, dirs, files in os.walk("./gcc/dumped/original"):
        for i, fname in enumerate(tqdm(files)):
            full_fname = os.path.join(root, fname)
            l = parsing_asm(full_fname,d)
            a.append(l)

    pd.DataFrame(a).to_csv("size.csv")




