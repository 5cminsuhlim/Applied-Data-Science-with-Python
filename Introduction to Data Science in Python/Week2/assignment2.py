#Question 1
import pandas as pd
import numpy as np
import scipy.stats as stats

def proportion_of_education():
    df = pd.read_csv('assets/NISPUF17.csv', index_col = 0)

    EDUC = df['EDUC1']

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0

    for k, v in EDUC.iteritems():
        if v == 1:
            c1 += 1
        elif v == 2:
            c2 += 1
        elif v == 3:
            c3 += 1
        elif v == 4:
            c4 += 1

    prop = {"less than high school": c1 / len(EDUC),
        "high school": c2 / len(EDUC),
        "more than high school but not college": c3 / len(EDUC),
        "college": c4 / len(EDUC)}

    return prop

#Question 2
import pandas as pd
import numpy as np
import scipy.stats as stats

def average_influenza_doses():
    df = pd.read_csv('assets/NISPUF17.csv', index_col = 0)

    CBF = df['CBF_01']
    CBF

    FLU = df['P_NUMFLU']
    FLU

    df = df[['CBF_01', 'P_NUMFLU']]
    df.dropna(inplace = True)

    rm = df[df['CBF_01'] == 1]['P_NUMFLU'].sum() / len(df[df['CBF_01'] == 1])
    nm = df[df['CBF_01'] == 2]['P_NUMFLU'].sum() / len(df[df['CBF_01'] == 2])

    return(rm, nm)

#Question 3
import pandas as pd
import numpy as np
import scipy.stats as stats

def chickenpox_by_sex():
    df = pd.read_csv('assets/NISPUF17.csv', index_col = 0)

    VAX = df[df['P_NUMVRC'] >= 1]
    VAX

    POS = VAX[VAX['HAD_CPOX'] == 1]
    POS
    NEG = VAX[VAX['HAD_CPOX'] == 2]
    NEG

    M_POS = POS[POS['SEX'] == 1]
    M_POS
    M_NEG = NEG[NEG['SEX'] == 1]
    M_NEG

    F_POS = POS[POS['SEX'] == 2]
    F_POS
    F_NEG = NEG[NEG['SEX'] == 2]
    F_NEG

    out = dict()

    out['male'] = len(M_POS) / len(M_NEG)
    out['female'] = len(F_POS) / len(F_NEG)

    return out

#Question 4
import pandas as pd
import numpy as np
import scipy.stats as stats

def corr_chickenpox():
    df = pd.read_csv('assets/NISPUF17.csv', index_col = 0)

    df[['HAD_CPOX', 'P_NUMVRC']]
    df = df[df['HAD_CPOX'] < 3]
    df = df[df['P_NUMVRC'].notna()]


    corr, pval=stats.pearsonr(df['HAD_CPOX'], df['P_NUMVRC'])

    return corr
