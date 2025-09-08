import re
import sys

scp = sys.argv[1]
output = sys.argv[2]

def contains_english(text):
    return bool(re.search(r'[a-zA-Z]', text))

with open(output, 'w') as out:
    with open(scp, 'r') as file:
        line = file.readline()
        while line:
            content = line.split(" ")[1:]
            text = ""
            for c in content:
                text += c
            if not contains_english(text):
                out.write(line)
            line = file.readline()