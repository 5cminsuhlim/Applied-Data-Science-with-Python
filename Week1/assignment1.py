#Question 1
import re

def names():
    simple_string = """Amy is 5 years old, and her sister Mary is 2 years old.
    Ruth and Peter, their parents, have 3 kids."""

    return re.findall('[A-Z][a-z]*', simple_string)


#Question 2
import re

def grades():
    with open ("assets/grades.txt", "r") as file:
        grades = file.read()

    lst = list()
    for item in re.findall('[A-Z][a-z]*\s[A-Z][a-z]*: B', grades):
        lst.append(item.split(':')[0])
    return lst


#Question 3
import re

def logs():
    with open("assets/logdata.txt", "r") as file:
        logdata = file.read()

    pattern = '''
    (?P<host>.*)
    (\s-\s)
    (?P<user_name>.*)
    (\s\[)
    (?P<time>.*)
    (\]\s\")
    (?P<request>.*)
    (\")
    '''
    lst = list()

    for item in re.finditer(pattern, logdata, re.VERBOSE):
        lst.append(item.groupdict())
    return lst
