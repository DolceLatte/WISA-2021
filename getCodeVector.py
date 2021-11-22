import os
from tqdm import tqdm
import pandas as pd

def makedict(path):
    opcode = dict()
    for root, dirs, files in os.walk(path):
        for i, fname in enumerate(tqdm(files)):
            full_fname = os.path.join(root, fname)
            f = open(full_fname,encoding='UTF8')
            line = f.readline()
            while line:
                s = line[0]
                if s == ' ':
                    word = line.split("\t")
                    if len(word) > 2:
                        op = word[-1].split(" ")[0]
                        if op in opcode.keys():
                            opcode[op] += 1
                        else:
                            opcode[op] = 0
                line = f.readline()
    return opcode

def parsing_asm(path,code):
    opcode = dict(code)
    f = open(path,encoding='UTF8')
    line = f.readline()
    while line:
        s = line[0]
        if s == ' ':
            word = line.split("\t")
            op = word[-1].split(" ")[0]
            if op in opcode.keys():
                opcode[op] += 1
        line = f.readline()
    return list(opcode.values())

#d = {'push': 0, 'mov': 0, 'movb': 0, 'pop': 0, 'retq': 0, 'xchg': 0, 'sub': 0, 'imul': 0, 'and': 0, 'cmp': 0, 'sete': 0, 'setl': 0, 'or': 0, 'test': 0, 'jne': 0, 'jmpq': 0, 'add': 0, 'movl': 0, 'callq': 0, 'xor': 0, 'movzbl': 0, 'cmove': 0, 'movw': 0, 'movzwl': 0, 'data16': 0, 'nopw': 0, 'nopl': 0, 'je': 0, 'movabs': 0, 'movsbl': 0, 'setne': 0, 'nop\n': 0, 'cmpl': 0, 'cmovne': 0, 'shl': 0, 'shr': 0, 'movsbq': 0, 'movsd': 0, 'addsd': 0, 'cvttsd2si': 0, 'cvtsi2sdl': 0, 'setge': 0, 'lea': 0, 'jle': 0, 'jge': 0, 'seta': 0, 'movslq': 0, 'cmpb': 0, 'movq': 0, 'cmpq': 0, 'cvtsd2ss': 0, 'movss': 0, 'setg': 0, 'sar': 0, 'movswl': 0, 'movups': 0, 'setle': 0, 'jg': 0, 'jae': 0, 'cltd': 0, 'idivl': 0, 'jl': 0, 'idiv': 0, 'ja': 0, 'movaps': 0, 'cqto': 0, 'xorps': 0, 'ucomisd': 0, 'setp': 0, 'cvtss2sd': 0, 'divsd': 0, 'cvtsi2ssl': 0, 'mulss': 0, 'cvttss2si': 0, 'cvtsi2ss': 0, 'subss': 0, 'setae': 0, 'addss': 0, 'jp': 0, 'inc': 0, 'fldt': 0, 'fstpt': 0, 'jb': 0, 'jbe': 0, 'ud2': 0, 'setb': 0, 'mulsd': 0, 'subsd': 0, 'flds': 0, 'frndint': 0, 'fstps': 0, 'pand': 0, 'movd': 0, 'cvtsi2sd': 0, 'movswq': 0, 'fdivrp': 0, 'fmulp': 0, 'fld': 0, 'fsub': 0, 'fnstcw': 0, 'fldcw': 0, 'fistpll': 0, 'fucomip': 0, 'fstp': 0, 'cmova': 0, 'ucomiss': 0, 'setbe': 0, 'fxch': 0, 'movlps': 0, 'jnp': 0, 'cvtsi2sdq': 0, 'fldl': 0, 'divss': 0, 'divq': 0, 'divl': 0, 'punpckldq': 0, 'movapd': 0, 'subpd': 0, 'pshufd': 0, 'addpd': 0, 'cwtl': 0, 'div': 0, 'fchs': 0, 'testb': 0, 'pxor': 0, 'fld1': 0, 'fldz': 0, 'fadd': 0, 'not': 0, 'setnp': 0, 'cmovb': 0, 'cpuid': 0, 'cmovl': 0, 'paddusb': 0, 'emms': 0, 'cmovg': 0, 'cmpw': 0, 'dec': 0, 'fildl': 0, 'faddp': 0, 'fstpl': 0, 'paddw': 0, 'addps': 0, 'rep': 0, 'cltq': 0, 'movdqa': 0, 'lock': 0, 'sets': 0, 'fcmovne': 0, 'prefetcht0': 0, 'sqrtsd': 0, 'jns': 0, 'idivq': 0, 'cmpltsd': 0, 'andpd': 0, 'andnpd': 0, 'orpd': 0, 'fildll': 0, 'fsubr': 0, 'fsubrp': 0, 'fmul': 0, 'fsubp': 0, 'fistpl': 0, 'cmplesd': 0, 'fst': 0, 'f2xm1': 0, 'fscale': 0, 'bts': 0, 'sbb': 0, 'fabs': 0, 'fpatan': 0, 'fsqrt': 0, 'punpcklbw': 0, 'punpcklwd': 0, 'shufps': 0, 'packuswb': 0, 'pinsrw': 0, 'pshufhw': 0, 'pshuflw': 0, 'cmovge': 0, 'por': 0, 'cvtsi2ssq': 0, 'js': 0, 'neg': 0, 'shld': 0, 'cld': 0, 'std': 0, 'fdivp': 0, 'adc': 0, 'xgetbv': 0, 'movupd': 0, 'cmpeqsd': 0, 'cmovp': 0, 'cmpunordsd': 0, 'cmpneqsd': 0, 'cmpnltsd': 0, 'cmpnlesd': 0, 'cmpordsd': 0, 'cmpeqss': 0, 'cmpltss': 0, 'cmpless': 0, 'cmpunordss': 0, 'cmpneqss': 0, 'cmpnltss': 0, 'cmpnless': 0, 'cmpordss': 0, 'movhlps': 0, 'pextrw': 0, 'movlhps': 0, 'unpcklpd': 0, 'unpcklps': 0, 'pcmpeqd': 0, 'pandn': 0, 'comisd': 0, 'comiss': 0, 'cvtdq2pd': 0, 'cvtdq2ps': 0, 'cvtpd2dq': 0, 'cvtpd2ps': 0, 'cvtps2dq': 0, 'cvtps2pd': 0, 'cvtsd2si': 0, 'cvtss2si': 0, 'cvttpd2dq': 0, 'cvttps2dq': 0, 'divpd': 0, 'divps': 0, 'movdqu': 0, 'maskmovdqu': 0, 'maxpd': 0, 'maxps': 0, 'maxsd': 0, 'maxss': 0, 'minpd': 0, 'minps': 0, 'minsd': 0, 'minss': 0, 'unpckhpd': 0, 'movmskpd': 0, 'movmskps': 0, 'movntdq': 0, 'movntpd': 0, 'movntps': 0, 'punpcklqdq': 0, 'mulpd': 0, 'mulps': 0, 'packssdw': 0, 'packsswb': 0, 'paddb': 0, 'paddd': 0, 'paddq': 0, 'paddsb': 0, 'paddsw': 0, 'paddusw': 0, 'pavgb': 0, 'pavgw': 0, 'pcmpeqb': 0, 'pcmpeqw': 0, 'pcmpgtb': 0, 'pcmpgtd': 0, 'pcmpgtw': 0, 'shufpd': 0, 'psllw': 0, 'pslld': 0, 'psllq': 0, 'pslldq': 0, 'pmaxsw': 0, 'pmaxub': 0, 'pminsw': 0, 'pminub': 0, 'pmovmskb': 0, 'pmulhuw': 0, 'pmulhw': 0, 'pmullw': 0, 'pmuludq': 0, 'psadbw': 0, 'psrldq': 0, 'psrad': 0, 'psraw': 0, 'psrld': 0, 'psrlq': 0, 'psrlw': 0, 'psubb': 0, 'psubd': 0, 'psubq': 0, 'psubsb': 0, 'psubsw': 0, 'psubw': 0, 'punpckhbw': 0, 'punpckhdq': 0, 'punpckhqdq': 0, 'punpckhwd': 0, 'rcpss': 0, 'rcpps': 0, 'rsqrtss': 0, 'rsqrtps': 0, 'sqrtpd': 0, 'sqrtss': 0, 'sqrtps': 0, 'subps': 0, 'unpckhps': 0, 'vmovaps': 0, 'movlpd': 0, 'cmovbe': 0, 'bsr': 0, 'bsf': 0, 'bswap': 0, 'rol': 0, 'seto': 0, 'mull': 0, 'mul': 0, 'mulq': 0, 'imulb': 0, 'mulb': 0, 'mulw': 0, 'setns': 0, 'cmovs': 0, 'cmovae': 0, 'prefetchnta': 0, 'prefetcht2': 0, 'prefetcht1': 0, 'fucomi': 0, 'andps': 0, 'andnps': 0, 'orps': 0, 'fdivr': 0, 'fdiv': 0, 'cmovle': 0, 'stmxcsr': 0, 'ldmxcsr': 0, 'mfence': 0, 'vmovss': 0, 'vmulss': 0, 'vaddss': 0, 'vmovsd': 0, 'vmulsd': 0, 'vaddsd': 0, 'vsubss': 0, 'vsubsd': 0, 'vmovd': 0, 'vmovq': 0, 'popcnt': 0, 'fxrstor': 0, 'fxrstor64': 0, 'fxsave': 0, 'shrd': 0, 'iretq': 0, 'movntdqa': 0, 'vandps': 0, 'vzeroupper': 0, 'vmovups': 0, 'cmpordps': 0, 'cmpordpd': 0, 'cvtpi2ps': 0, 'pmaddwd': 0, 'crc32q': 0, 'maskmovq': 0, 'fcmovu': 0, 'cmpeqpd': 0, 'cmpeqps': 0, 'cwtd': 0, 'rdtsc': 0, 'rdtscp': 0, 'rdpmc': 0, 'vfmadd213sd': 0, 'bt': 0, 'pushfq': 0, 'rdrand': 0, 'pushq': 0, 'vmovdqa': 0, 'kmovw': 0, 'kxnorw': 0, 'kshiftrw': 0, 'kxorw': 0, 'korw': 0, 'vcvtss2sd': 0, 'rex': 0, 'pause': 0, 'pshufw': 0, 'cmpunordps': 0, 'cmpunordpd': 0, 'psubusw': 0, 'cmpltps': 0, 'cmpleps': 0, 'cmpneqps': 0, 'cmpltpd': 0, 'cmplepd': 0, 'cmpneqpd': 0, 'popfq': 0, 'fxsave64': 0, '(bad)': 0, 'decl': 0, 'stos': 0, 'vxorpd': 0, 'vpcmpeqd': 0, 'vinsertf128': 0, 'vxorps': 0, 'vorps': 0, 'rorb': 0, 'stc': 0, 'hlt': 0, 'cbtw': 0}
'''
d = {
    'callq': 0, 'pop': 0, 'push': 0, 'jge': 0,
    'add': 0, 'movl': 0, 'cmp': 0,  'sete': 0,
    'test': 0, 'jmpq': 0, 'setl': 0, 'or': 0,
    'and': 0, 'imul': 0, 'cmovne': 0, 'jne': 0,
    'sub': 0,'je': 0, 'xor': 0
   }
'''
'''
d = {
    'callq': 0, 'pop': 1, 'push': 2, 'jge': 3,
    'add': 4, 'movl': 5, 'cmp': 6,  'sete': 7,
    'test': 8, 'jmpq': 9, 'setl': 10, 'or': 11,
    'and': 12, 'imul': 13, 'cmovne': 14, 'jne': 15,
    'sub': 16,'je': 17, 'xor': 18
   }
'''

# d = {
#     'pop': 1,'jge': 3,'cmp': 6,'setl': 10, 'or': 11,'and': 12, 'imul': 13, 'xor': 18
#    }


# d = {
# 'push': 0,'sub': 0,'mov':0,'or': 0,'jmpq': 0,'imul':0,
# 'add': 0,'xor': 0,'cmovne':0,'jne': 0,'je': 0,'movl':0,
# }

# d = {
#     'sub':0,'test':0,'je':0,'callq':0,
#     'add':0,'jmpq':0,'xor':0,'and':0,
#     'pop':0,'push':0,'cmp':0,'imul':0,
#     'or':0,'sete':0,'setl':0
#    }

def print_files_in_dir(root_dir,dir_name):
    files = os.listdir(root_dir)
    for file in tqdm(files):
        path = os.path.join(root_dir, file)
        if os.path.isdir(path):
            print_files_in_dir(path, file)
        else:
            l = parsing_asm(path, d)
            df = pd.DataFrame(l)
            file_name, file_ext = os.path.splitext(file)
            df.to_csv('../FilePreprocessing/gcc_top_20_nomov/{}'.format(file_name + "_" + dir_name+file_ext),
                       index=None, header=None)
            #df.to_csv('C:\\Users\\김정우\\PycharmProjects\\obf_torch\\FilePreprocessing\\gcc_top_20_nomov\\{}'.format(file),
             #        index=None, header=None)

# d = {
#      'push': 0, 'mov':0,'pop': 0, 'sub': 0, 'imul': 0, 'add': 0 ,'cmp': 0,  'callq': 0, 'sete': 0,
#      'setl': 0, 'or':0, 'test': 0,'jne': 0,'jmpq': 0,'and': 0,'movl': 0,'xor': 0, 'cmovne': 0,
#      'je': 0
#  }
'''
d = {
     'setl': 0,'sete': 0,
     'push': 0, 'pop': 0, 'movabs':0, 'movl': 0,'cmovne': 0,'cmove': 0,'cmovl': 0,
     'or': 0, 'test': 0,'and': 0,'xor': 0,
     'jne': 0, 'jmpq': 0,'jge': 0,'je':0,
     'sub': 0, 'add': 0 ,'cmp': 0,
 }
'''
'''
d = {'add': 0, 'or': 0, 'callq': 0, 'push': 0, 'nopl': 0, 'sub': 0,
     'sete': 0, 'cmove': 0, 'cmovne': 0, #'mov': 0,
     'nopw': 0, 'xchg': 0,
     'movups': 0, 'imul': 0, 'setl': 0, 'cmovl': 0, 'movabs': 0, 'movl': 0,
     'pop': 0, 'jge': 0, 'cmp': 0, 'setne': 0, 'xor': 0, 'test': 0, 'jne': 0,
     'and': 0, 'je': 0, 'jmpq': 0}
'''

d = { #'mov': 0,
    'jp': 0, 'add': 0, 'movsd': 0, 'cmove': 0, 'setne': 0, 'cvtsi2sdl': 0, 'jne': 0, 
    'or': 0, 'movabs': 0, 'jmpq': 0, 'xchg': 0, 'setp': 0, 'sub': 0, 'callq': 0, 
    'imul': 0, 'nopw': 0, 'test': 0, 'cvtsi2sd': 0, 'jae': 0, 'je': 0, 'and': 0, 
    'cmovne': 0, 'sete': 0, 'pop': 0, 'dec': 0, 'movups': 0, 'lea': 0, 'cmp': 0, 'xor': 0,
    'nopl': 0, 'setl': 0, 'cmovl': 0, 'push': 0, 'jge': 0, 'movaps': 0, 'movl': 0}

'''
d = {'cpuid': 0,
 'xchg': 0,
 'movslq': 0,
 'movabs': 0,
 'lea': 0,
 'retq': 0,
 'cmpl': 0,
 'pop': 0,
 'callq': 0,
 'cmovl': 0,
 'push': 0,
 'jge': 0,
 'imul': 0,
 'movl': 0,
 'cmp': 0,
 'add': 0,
 #'mov': 0,
 'jmpq': 0,
 'sete': 0,
 'test': 0,
 'setl': 0,
 'or': 0,
 'and': 0,
 'je': 0,
 'cmovne': 0,
 'sub': 0,
 'jne': 0,
 'xor': 0}
'''
'''
d = {
     'mov':0, 'sub': 0, 'imul': 0, 'add': 0 ,'cmp': 0, 'or':0,'jne': 0,'jmpq': 0,'and': 0,'movl': 0,
    'xor': 0, 'je': 0
}
'''

#d = {'push': 0, 'mov': 0, 'movb': 0, 'pop': 0, 'retq': 0, 'xchg': 0, 'sub': 0, 'imul': 0, 'and': 0, 'cmp': 0, 'sete': 0, 'setl': 0, 'or': 0, 'test': 0, 'jne': 0, 'jmpq': 0, 'add': 0, 'movl': 0, 'callq': 0, 'xor': 0, 'movzbl': 0, 'cmove': 0, 'movw': 0, 'movzwl': 0, 'nopw': 0, 'nopl': 0, 'je': 0, 'movabs': 0, 'movsbl': 0, 'setne': 0, 'nop\n': 0, 'movslq': 0, 'cmpl': 0, 'cmovne': 0, 'shl': 0, 'shr': 0, 'movsbq': 0, 'movsd': 0, 'addsd': 0, 'cvttsd2si': 0, 'cvtsi2sdl': 0, 'setge': 0, 'lea': 0, 'jle': 0, 'jge': 0, 'seta': 0, 'cmpb': 0, 'movq': 0, 'cmpq': 0, 'cvtsd2ss': 0, 'movss': 0, 'setg': 0, 'sar': 0, 'movswl': 0, 'movups': 0, 'setle': 0, 'jg': 0, 'jae': 0, 'cltd': 0, 'idivl': 0, 'jl': 0, 'idiv': 0, 'ja': 0, 'movaps': 0, 'cqto': 0, 'xorps': 0, 'ucomisd': 0, 'setp': 0, 'cvtss2sd': 0, 'divsd': 0, 'cvtsi2ssl': 0, 'mulss': 0, 'cvttss2si': 0, 'cvtsi2ss': 0, 'subss': 0, 'setae': 0, 'addss': 0, 'jp': 0, 'inc': 0, 'fldt': 0, 'fstpt': 0, 'jb': 0, 'jbe': 0, 'ud2': 0, 'setb': 0, 'mulsd': 0, 'subsd': 0, 'flds': 0, 'frndint': 0, 'fstps': 0, 'faddp': 0, 'pand': 0, 'movd': 0, 'cvtsi2sd': 0, 'movswq': 0, 'fdivrp': 0, 'fmulp': 0, 'fld': 0, 'fsub': 0, 'fnstcw': 0, 'fldcw': 0, 'fistpll': 0, 'fucomip': 0, 'fstp': 0, 'cmova': 0, 'ucomiss': 0, 'setbe': 0, 'fxch': 0, 'movlps': 0, 'jnp': 0, 'cvtsi2sdq': 0, 'fldl': 0, 'divss': 0, 'divq': 0, 'divl': 0, 'punpckldq': 0, 'movapd': 0, 'subpd': 0, 'pshufd': 0, 'addpd': 0, 'cwtl': 0, 'div': 0, 'fchs': 0, 'testb': 0, 'pxor': 0, 'fld1': 0, 'fldz': 0, 'fadd': 0, 'not': 0, 'setnp': 0, 'cmovb': 0, 'cpuid': 0, 'cmovl': 0, 'paddusb': 0, 'emms': 0, 'cmovg': 0, 'cmpw': 0, 'dec': 0, 'fildl': 0, 'fstpl': 0, 'prefetcht0': 0, 'paddw': 0, 'addps': 0, 'rep': 0, 'cltq': 0, 'movdqa': 0, 'unpcklps': 0, 'lock': 0, 'sets': 0, 'fcmovne': 0, 'sqrtsd': 0, 'jns': 0, 'idivq': 0, 'cmpltsd': 0, 'andpd': 0, 'andnpd': 0, 'orpd': 0, 'fildll': 0, 'fsubr': 0, 'fsubrp': 0, 'fmul': 0, 'fsubp': 0, 'fistpl': 0, 'cmplesd': 0, 'fst': 0, 'f2xm1': 0, 'fscale': 0, 'bts': 0, 'sbb': 0, 'fabs': 0, 'fpatan': 0, 'fsqrt': 0, 'punpcklbw': 0, 'punpcklwd': 0, 'shufps': 0, 'packuswb': 0, 'pinsrw': 0, 'pshufhw': 0, 'pshuflw': 0, 'cmovge': 0, 'pcmpeqd': 0, 'por': 0, 'cvtsi2ssq': 0, 'js': 0, 'neg': 0, 'shld': 0, 'cld': 0, 'std': 0, 'fdivp': 0, 'adc': 0, 'mfence': 0, 'vxorpd': 0, 'xgetbv': 0, 'movupd': 0, 'cmpeqsd': 0, 'cmovp': 0, 'cmpunordsd': 0, 'cmpneqsd': 0, 'cmpnltsd': 0, 'cmpnlesd': 0, 'cmpordsd': 0, 'cmpeqss': 0, 'cmpltss': 0, 'cmpless': 0, 'cmpunordss': 0, 'cmpneqss': 0, 'cmpnltss': 0, 'cmpnless': 0, 'cmpordss': 0, 'movhlps': 0, 'pextrw': 0, 'movlhps': 0, 'unpcklpd': 0, 'pandn': 0, 'comisd': 0, 'comiss': 0, 'cvtdq2pd': 0, 'cvtdq2ps': 0, 'cvtpd2dq': 0, 'cvtpd2ps': 0, 'cvtps2dq': 0, 'cvtps2pd': 0, 'cvtsd2si': 0, 'cvtss2si': 0, 'cvttpd2dq': 0, 'cvttps2dq': 0, 'divpd': 0, 'divps': 0, 'movdqu': 0, 'maskmovdqu': 0, 'maxpd': 0, 'maxps': 0, 'maxsd': 0, 'maxss': 0, 'minpd': 0, 'minps': 0, 'minsd': 0, 'minss': 0, 'unpckhpd': 0, 'movmskpd': 0, 'movmskps': 0, 'movntdq': 0, 'movntpd': 0, 'movntps': 0, 'punpcklqdq': 0, 'mulpd': 0, 'mulps': 0, 'packssdw': 0, 'packsswb': 0, 'paddb': 0, 'paddd': 0, 'paddq': 0, 'paddsb': 0, 'paddsw': 0, 'paddusw': 0, 'pavgb': 0, 'pavgw': 0, 'pcmpeqb': 0, 'pcmpeqw': 0, 'pcmpgtb': 0, 'pcmpgtd': 0, 'pcmpgtw': 0, 'shufpd': 0, 'psllw': 0, 'pslld': 0, 'psllq': 0, 'pslldq': 0, 'pmaxsw': 0, 'pmaxub': 0, 'pminsw': 0, 'pminub': 0, 'pmovmskb': 0, 'pmulhuw': 0, 'pmulhw': 0, 'pmullw': 0, 'pmuludq': 0, 'psadbw': 0, 'psrldq': 0, 'psrad': 0, 'psraw': 0, 'psrld': 0, 'psrlq': 0, 'psrlw': 0, 'psubb': 0, 'psubd': 0, 'psubq': 0, 'psubsb': 0, 'psubsw': 0, 'psubw': 0, 'punpckhbw': 0, 'punpckhdq': 0, 'punpckhqdq': 0, 'punpckhwd': 0, 'rcpss': 0, 'rcpps': 0, 'rsqrtss': 0, 'rsqrtps': 0, 'sqrtpd': 0, 'sqrtss': 0, 'sqrtps': 0, 'subps': 0, 'unpckhps': 0, 'vmovaps': 0, 'vcmpltss': 0, 'vaddss': 0, 'movlpd': 0, 'cmovbe': 0, 'bsr': 0, 'bsf': 0, 'bswap': 0, 'rol': 0, 'seto': 0, 'mull': 0, 'mul': 0, 'mulq': 0, 'imulb': 0, 'mulb': 0, 'mulw': 0, 'setns': 0, 'cmovs': 0, 'cmovae': 0, 'prefetchnta': 0, 'prefetcht2': 0, 'prefetcht1': 0, 'fucomi': 0, 'andps': 0, 'andnps': 0, 'orps': 0, 'fdivr': 0, 'fdiv': 0, 'cmovle': 0, 'stmxcsr': 0, 'ldmxcsr': 0, 'cmpeqps': 0, 'vmovss': 0, 'vmulss': 0, 'vmovsd': 0, 'vmulsd': 0, 'vaddsd': 0, 'vsubss': 0, 'vsubsd': 0, 'vmovd': 0, 'vmovq': 0, 'popcnt': 0, 'fxrstor': 0, 'fxrstor64': 0, 'fxsave': 0, 'fxsave64': 0, 'shrd': 0, 'iretq': 0, 'movntdqa': 0, 'vandps': 0, 'vzeroupper': 0, 'vmovups': 0, 'cmpordps': 0, 'cmpordpd': 0, 'pause': 0, 'cvtpi2ps': 0, 'pmaddwd': 0, 'crc32q': 0, 'maskmovq': 0, 'fcmovu': 0, 'cmpeqpd': 0, 'cwtd': 0, 'rdtsc': 0, 'rdtscp': 0, 'rdpmc': 0, 'vfmadd213sd': 0, 'fcmovb': 0, 'bt': 0, 'pushfq': 0, 'rdrand': 0, 'pushq': 0, 'vmovdqa': 0, 'kmovw': 0, 'kxnorw': 0, 'kshiftrw': 0, 'kxorw': 0, 'korw': 0, 'vcvtss2sd': 0, 'rex': 0, 'pshufw': 0, 'cmpunordps': 0, 'cmpunordpd': 0, 'psubusw': 0, 'cmpltps': 0, 'cmpleps': 0, 'cmpneqps': 0, 'cmpltpd': 0, 'cmplepd': 0, 'cmpneqpd': 0, 'popfq': 0, '(bad)': 0, 'decl': 0, 'stos': 0, 'vpcmpeqd': 0, 'vinsertf128': 0, 'vxorps': 0, 'vorps': 0, 'rorb': 0, 'stc': 0, 'hlt': 0, 'cbtw': 0}


if __name__ == '__main__':
    print(len(d))

    root_dir = "../FilePreprocessing/gcc_dumped"
    #root_dir = "../dumped_tigress/"
    print_files_in_dir(root_dir,root_dir)

    # for root, dirs, files in os.walk(root_dir):
    #     for i, fname in enumerate(tqdm(files)):
    #         full_fname = os.path.join(root, fname)
    #         l = parsing_asm(full_fname,d)
    #         df = pd.DataFrame(l)
    #         df.to_csv('../FilePreprocessing/tigress_top_15_nomov/{}'.format(fname),index=None,header=None)
