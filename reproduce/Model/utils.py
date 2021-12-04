class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKRED='\033[0;31m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tokgreen(s):
    return bcolors.OKGREEN + s + bcolors.ENDC

def tokblue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC

def tokred(s):
    return bcolors.OKRED + s + bcolors.ENDC

def tokwaring(s):
    return bcolors.WARNING + s + bcolors.ENDC
